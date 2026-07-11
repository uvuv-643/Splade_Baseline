import json
import os
import signal
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import psutil

WARN_PCT = float(os.environ.get("LAB_MEM_WARN_PCT", 80))
KILL_PCT = float(os.environ.get("LAB_MEM_KILL_PCT", 90))
GRACE_S = float(os.environ.get("LAB_MEM_GRACE_S", 15))
SAMPLE_S = 2.0
HISTORY_EVERY_S = 30.0

# CSR-индекс: float32 data + int32 indices на каждый ненулевой элемент.
BYTES_PER_NNZ = 8

CGROUP_V2 = Path("/sys/fs/cgroup")
CGROUP_V1 = Path("/sys/fs/cgroup/memory")


class Terminated(Exception):
    """SIGTERM, превращённый в исключение: except-путь runner'а успевает
    записать traceback в stdout.log, meta.json и status перед смертью."""


def install_sigterm_handler():
    if threading.current_thread() is not threading.main_thread():
        return

    def _raise(signum, frame):
        raise Terminated(f"получен SIGTERM (pid={os.getpid()})")

    signal.signal(signal.SIGTERM, _raise)


def _read_int(path: Path):
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if text == "max":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def cgroup_memory():
    """Лимит cgroup (контейнер/DataSphere): именно его превышение ловит OOM
    killer внутри контейнера, host-память может при этом быть свободна.
    v1 без лимита отдаёт PAGE_COUNTER_MAX — отсекается порогом 1<<60."""
    candidates = (
        (CGROUP_V2 / "memory.max", CGROUP_V2 / "memory.current"),
        (CGROUP_V1 / "memory.limit_in_bytes", CGROUP_V1 / "memory.usage_in_bytes"),
    )
    for limit_file, used_file in candidates:
        limit = _read_int(limit_file)
        used = _read_int(used_file)
        if limit is not None and used is not None and limit < (1 << 60):
            return {"limit": limit, "used": used,
                    "percent": round(used / limit * 100, 1)}
    return None


def user_memory():
    """Квота пользователя на общей машине без cgroup-лимита: лимит задаётся
    через LAB_MEM_TOTAL_GB (`lab worker ... --mem-gb`), занято — суммарный RSS
    процессов текущего пользователя. None, если лимит не задан/невалиден."""
    raw = os.environ.get("LAB_MEM_TOTAL_GB")
    if not raw:
        return None
    try:
        limit = int(float(raw) * 2**30)
    except ValueError:
        return None
    if limit <= 0:
        return None
    uid = os.getuid()
    rss = 0
    for p in psutil.process_iter(["uids", "memory_info"]):
        info = p.info
        if info["uids"] and info["uids"].real == uid and info["memory_info"]:
            rss += info["memory_info"].rss
    if rss <= 0:
        return None
    return {"limit": limit, "used": rss,
            "percent": round(rss / limit * 100, 1)}


def system_snapshot() -> dict:
    """Эффективное состояние памяти: базой служит квота пользователя
    (LAB_MEM_TOTAL_GB), если задана, иначе host (по MemAvailable); cgroup
    перекрывает базу, когда он более ограничен."""
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    host = {"total": vm.total, "used": vm.total - vm.available,
            "percent": round(vm.percent, 1)}
    cg = cgroup_memory()
    user = user_memory()
    if user:
        used, limit, percent, source = user["used"], user["limit"], user["percent"], "user"
    else:
        used, limit, percent, source = host["used"], host["total"], host["percent"], "host"
    if cg and cg["percent"] > percent:
        used, limit, percent, source = cg["used"], cg["limit"], cg["percent"], "cgroup"
    return {
        "t": datetime.now(timezone.utc).isoformat(),
        "percent": round(percent, 1),
        "used": used,
        "limit": limit,
        "source": source,
        "host": host,
        "cgroup": cg,
        "user": user,
        "swap": {"total": sw.total, "used": sw.used, "percent": sw.percent},
    }


def process_tree(pid=None) -> dict:
    try:
        proc = psutil.Process(pid)
        procs = [proc] + proc.children(recursive=True)
    except psutil.Error:
        return {"pid": pid, "rss": 0, "n_procs": 0}
    rss = 0
    for p in procs:
        try:
            rss += p.memory_info().rss
        except psutil.Error:
            continue
    return {"pid": proc.pid, "rss": rss, "n_procs": len(procs)}


def level(percent: float) -> str:
    if percent >= KILL_PCT:
        return "critical"
    if percent >= WARN_PCT:
        return "warning"
    return "ok"


def gb(n_bytes) -> str:
    return f"{n_bytes / 2**30:.1f}"


_encode_state = {}


def note_encode(kind: str, done: int, total: int, nnz_total: int) -> dict:
    """Прогресс энкода из eval: средний nnz вектора и прогноз размера полного
    CSR-индекса. Читается watchdog-потоком, попадает в memory.json и OOM-отчёт."""
    global _encode_state
    avg_nnz = nnz_total / max(1, done)
    _encode_state = {
        "kind": kind,
        "done": done,
        "total": total,
        "avg_nnz": round(avg_nnz, 1),
        "projected_bytes": int(avg_nnz * total * BYTES_PER_NNZ),
    }
    return _encode_state


def encode_state() -> dict:
    return dict(_encode_state)


def _write_json(path: Path, payload: dict):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(path)


def read_memory(run_dir) -> dict:
    try:
        return json.loads((Path(run_dir) / "memory.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def read_oom(run_dir) -> dict:
    try:
        return json.loads((Path(run_dir) / "oom_kill.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_oom_report(run_dir, killed_by: str, phase: str, snap: dict,
                     history=None, reason=None) -> Path:
    """Отчёт о принудительном убийстве — пишется ДО сигнала, чтобы пережить
    даже SIGKILL. Если отчёт этой попытки уже есть (in-process watchdog успел
    раньше внешнего) — не перетираем более детальную версию."""
    path = Path(run_dir) / "oom_kill.json"
    if path.exists():
        return path
    report = {
        "t": datetime.now(timezone.utc).isoformat(),
        "killed_by": killed_by,
        "phase": phase,
        "reason": reason or f"память {snap['percent']}% ≥ kill-порога {KILL_PCT:.0f}%",
        "thresholds": {"warn_pct": WARN_PCT, "kill_pct": KILL_PCT, "grace_s": GRACE_S},
        "snapshot": snap,
        "encode": encode_state() or None,
        "history": list(history or []),
    }
    _write_json(path, report)
    return path


def terminate_tree(pid: int, grace: float = None) -> bool:
    """Аккуратно гасит группу процессов: SIGTERM (шанс дописать логи/статус),
    после grace — SIGKILL. True, если процесс завершился сам по SIGTERM."""
    grace = GRACE_S if grace is None else grace
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGTERM)
    except OSError:
        return False
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return True
        time.sleep(0.5)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except OSError:
        pass
    return False


class MemWatch:
    """Watchdog внутри процесса-эксперимента. Каждые SAMPLE_S секунд пишет
    memory.json (текущее состояние для UI) и, реже, memory.jsonl (таймлайн для
    post-mortem). На warn-пороге предупреждает в stdout, на kill-пороге
    сохраняет oom_kill.json и гасит собственный процесс: SIGTERM → grace →
    SIGKILL всей группы."""

    def __init__(self, run_dir, phase: str):
        self.run_dir = Path(run_dir)
        self.phase = phase
        self.peak = {"percent": 0.0, "rss": 0}
        self.history = deque(maxlen=30)
        self._warned = False
        self._last_level = "ok"
        self._last_history_t = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="memwatch")

    def start(self):
        (self.run_dir / "oom_kill.json").unlink(missing_ok=True)
        snap = system_snapshot()
        print(f"[memwatch] {self.phase}: лимит {gb(snap['limit'])} GB "
              f"({snap['source']}), warn {WARN_PCT:.0f}%, kill {KILL_PCT:.0f}%, "
              f"сейчас {snap['percent']}%", flush=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=SAMPLE_S * 2)

    def _loop(self):
        while not self._stop.wait(SAMPLE_S):
            try:
                self._sample()
            except Exception as e:
                print(f"[memwatch] ошибка сэмпла: {e}", flush=True)

    def _sample(self):
        snap = system_snapshot()
        tree = process_tree()
        self.peak["percent"] = max(self.peak["percent"], snap["percent"])
        self.peak["rss"] = max(self.peak["rss"], tree["rss"])
        snap.update({
            "phase": self.phase,
            "proc": tree,
            "peak": dict(self.peak),
            "encode": encode_state() or None,
            "level": level(snap["percent"]),
            "thresholds": {"warn_pct": WARN_PCT, "kill_pct": KILL_PCT},
        })
        self.history.append({"t": snap["t"], "percent": snap["percent"],
                             "rss": tree["rss"]})
        _write_json(self.run_dir / "memory.json", snap)
        now = time.monotonic()
        if (snap["level"] != self._last_level
                or self._last_history_t is None
                or now - self._last_history_t >= HISTORY_EVERY_S):
            self._last_history_t = now
            with open(self.run_dir / "memory.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(snap, ensure_ascii=False) + "\n")
        self._last_level = snap["level"]

        if snap["level"] == "critical":
            self._kill(snap)
        elif snap["level"] == "warning" and not self._warned:
            self._warned = True
            print(f"[memwatch] ВНИМАНИЕ: память {snap['percent']}% ≥ "
                  f"{WARN_PCT:.0f}% (занято {gb(snap['used'])} из "
                  f"{gb(snap['limit'])} GB, RSS процесса {gb(tree['rss'])} GB)",
                  flush=True)
        elif snap["level"] == "ok":
            self._warned = False

    def _kill(self, snap):
        self._stop.set()
        write_oom_report(self.run_dir, killed_by="memwatch", phase=self.phase,
                         snap=snap, history=self.history)
        enc = snap.get("encode")
        print(f"[memwatch] ПРЕВЫШЕН ЛИМИТ: память {snap['percent']}% ≥ "
              f"{KILL_PCT:.0f}% (занято {gb(snap['used'])} из "
              f"{gb(snap['limit'])} GB, RSS процесса "
              f"{gb(snap['proc']['rss'])} GB)", flush=True)
        if enc:
            print(f"[memwatch] на момент убийства: энкод {enc['kind']} "
                  f"{enc['done']}/{enc['total']}, avg_nnz={enc['avg_nnz']}, "
                  f"прогноз полного индекса {gb(enc['projected_bytes'])} GB — "
                  f"похоже, векторы недостаточно разреженные", flush=True)
        print(f"[memwatch] шлю SIGTERM (grace {GRACE_S:.0f}s на дозапись "
              f"логов), отчёт в oom_kill.json", flush=True)
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(GRACE_S)
        print("[memwatch] процесс не завершился за grace — SIGKILL группы",
              flush=True)
        os.killpg(os.getpgid(0), signal.SIGKILL)
