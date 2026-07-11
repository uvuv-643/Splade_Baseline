import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

from . import config as config_mod
from . import memwatch
from . import queue as queue_mod
from . import runner, runs
from .paths import QUEUE_DIR, ROOT, WORKER_LOG, WORKER_PIDFILE

POLL_S = 5
MEM_KILL_STREAK = 3


def detect_gpus(spec=None) -> list:
    if spec:
        return [s.strip() for s in str(spec).split(",") if s.strip() != ""]
    try:
        import torch
        n = torch.cuda.device_count()
    except Exception:
        n = 0
    return [str(i) for i in range(n)] or [None]


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[worker {ts}] {msg}", flush=True)


def adopt_stale():
    for info in runs.list_runs():
        if info["status"] == "running":
            pid_file = info["dir"] / "pid"
            pid = int(pid_file.read_text()) if pid_file.exists() else None
            if pid is None or not _pid_alive(pid):
                runs.set_status(info["dir"], "failed")
                _log(f"осиротевший running-запуск {info['id']} -> failed")
    n = queue_mod.requeue_claimed()
    if n:
        _log(f"{n} claimed-джобов возвращено в очередь")


def start_job(job: dict, gpu):
    """Запускает джоб по его виду. Возвращает (proc, run_dir, kind)."""
    kind = job.get("kind", "train")
    run_dir = runs.run_dir(job["run_id"])
    if kind == "eval":
        proc = runner.start_eval_process(
            run_dir, job["datasets"], job.get("save_index", False), gpu)
        return proc, run_dir, kind
    # train: каталог обычно уже зарезервирован (runs.reserve_run) при enqueue.
    # Fallback для старых/ручных джобов, несущих config+snapshot внутри себя.
    if not (run_dir / "config.yaml").exists():
        run_dir = runs.create_run_dir(job["run_id"])
        (run_dir / "config.yaml").write_text(
            config_mod.dump(job["config"]), encoding="utf-8")
        (run_dir / "snapshot.json").write_text(
            json.dumps(job["snapshot"]), encoding="utf-8")
        runs.set_status(run_dir, "queued")
    proc = runner.start_run_process(run_dir, gpu)
    return proc, run_dir, kind


def _oom_note(run_dir) -> str:
    oom = memwatch.read_oom(run_dir)
    if not oom:
        return ""
    return (f" — убит по памяти ({oom['killed_by']}): {oom['reason']}, "
            f"подробности в {run_dir}/oom_kill.json")


def _enforce_memory(active, mem_state):
    """Внешняя страховка поверх in-process watchdog: если процесс застрял в
    нативном коде и свой watchdog не сработал, память всё равно освободим.
    Убиваем при KILL_PCT MEM_KILL_STREAK поллов подряд — джоб с максимальным
    RSS дерева процессов."""
    snap = memwatch.system_snapshot()
    lvl = memwatch.level(snap["percent"])
    if lvl != "critical":
        if lvl == "warning" and mem_state["level"] == "ok":
            _log(f"ВНИМАНИЕ: память {snap['percent']}% ≥ "
                 f"{memwatch.WARN_PCT:.0f}% (занято {memwatch.gb(snap['used'])} "
                 f"из {memwatch.gb(snap['limit'])} GB)")
        mem_state.update(level="ok" if lvl == "ok" else "warning", streak=0)
        return
    mem_state["level"] = "critical"
    mem_state["streak"] += 1
    _log(f"память {snap['percent']}% ≥ {memwatch.KILL_PCT:.0f}% "
         f"({mem_state['streak']}/{MEM_KILL_STREAK} поллов)")
    if mem_state["streak"] < MEM_KILL_STREAK or not active:
        return
    trees = {gpu: memwatch.process_tree(job[0].pid)
             for gpu, job in active.items() if job[0].poll() is None}
    if not trees:
        return
    gpu = max(trees, key=lambda g: trees[g]["rss"])
    proc, run_dir, job_path, kind = active[gpu]
    snap["proc"] = trees[gpu]
    memwatch.write_oom_report(
        run_dir, killed_by="worker", phase=kind, snap=snap,
        reason=f"память {snap['percent']}% ≥ {memwatch.KILL_PCT:.0f}% "
               f"{MEM_KILL_STREAK} поллов подряд, in-process watchdog "
               f"не сработал")
    _log(f"убиваю {run_dir.name} ({kind}, RSS дерева "
         f"{memwatch.gb(trees[gpu]['rss'])} GB): SIGTERM, grace "
         f"{memwatch.GRACE_S:.0f}s, затем SIGKILL")
    memwatch.terminate_tree(proc.pid)
    mem_state["streak"] = 0


def _finish_job(gpu, active):
    proc, run_dir, job_path, kind = active[gpu]
    code = proc.poll()
    if code is None:
        return False
    if kind == "eval":
        # eval не владеет статусом запуска: train-запуск как был 'done', так и
        # остаётся. Упавший eval-джоб уводим в failed/ для наглядности.
        if code != 0:
            queue_mod.fail_job(job_path, f"eval {run_dir.name}: exit={code}"
                                         f"{_oom_note(run_dir)}\n"
                                         f"см. {run_dir}/stdout.log и eval_log.jsonl")
            _log(f"{run_dir.name}: EVAL ПРОВАЛ (exit={code}, gpu={gpu})"
                 f"{_oom_note(run_dir)}")
        else:
            job_path.unlink(missing_ok=True)
            _log(f"{run_dir.name}: eval готов (gpu={gpu})")
    else:
        status = runs.get_status(run_dir)
        if status not in ("done", "failed"):
            runs.set_status(run_dir, "failed")
            status = "failed"
        job_path.unlink(missing_ok=True)
        _log(f"{run_dir.name}: {status} (exit={code}, gpu={gpu})"
             f"{_oom_note(run_dir)}")
    del active[gpu]
    return True


def _shutdown(active):
    for gpu, (proc, run_dir, job_path, kind) in active.items():
        _log(f"останавливаю {run_dir.name} (gpu={gpu}, {kind})")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except OSError:
                pass
        if kind == "eval":
            # eval идемпотентен (перестраивает индекс/переписывает метрики) —
            # возвращаем джоб в очередь, доедет при следующем старте воркера.
            job_path.rename(QUEUE_DIR / job_path.name)
        else:
            runs.set_status(run_dir, "failed")
            job_path.unlink(missing_ok=True)
    _log("остановлен")


def worker_loop(gpus: list):
    _log(f"старт, gpus={gpus}")
    snap = memwatch.system_snapshot()
    _log(f"memwatch: лимит {memwatch.gb(snap['limit'])} GB ({snap['source']}), "
         f"warn {memwatch.WARN_PCT:.0f}%, kill {memwatch.KILL_PCT:.0f}%")
    adopt_stale()
    active = {}
    mem_state = {"level": "ok", "streak": 0}
    stop = {"flag": False}

    def on_term(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    while True:
        for gpu in list(active):
            _finish_job(gpu, active)

        if stop["flag"]:
            _shutdown(active)
            return

        _enforce_memory(active, mem_state)

        for gpu in gpus:
            if gpu in active:
                continue
            job_path, job = queue_mod.claim_next()
            if job is None:
                break
            proc, run_dir, kind = start_job(job, gpu)
            active[gpu] = (proc, run_dir, job_path, kind)
            _log(f"{run_dir.name}: запущен {kind} на gpu={gpu} (pid={proc.pid})")

        time.sleep(POLL_S)


def daemon_pid():
    if not WORKER_PIDFILE.exists():
        return None
    try:
        pid = int(WORKER_PIDFILE.read_text().strip())
    except ValueError:
        return None
    return pid if _pid_alive(pid) else None


def start_daemon(gpus_spec=None):
    pid = daemon_pid()
    if pid:
        print(f"worker уже работает (pid={pid})")
        return
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "lab"), "worker", "run"]
    if gpus_spec:
        cmd += ["--gpus", str(gpus_spec)]
    log = open(WORKER_LOG, "a", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    log.close()
    WORKER_PIDFILE.write_text(str(proc.pid), encoding="utf-8")
    print(f"worker запущен в фоне: pid={proc.pid}, лог {WORKER_LOG}")
    print("он переживёт закрытие терминала/ssh — tmux не нужен")


def stop_daemon():
    pid = daemon_pid()
    if not pid:
        print("worker не запущен")
        WORKER_PIDFILE.unlink(missing_ok=True)
        return
    os.kill(pid, signal.SIGTERM)
    for _ in range(60):
        if not _pid_alive(pid):
            break
        time.sleep(1)
    else:
        os.kill(pid, signal.SIGKILL)
    WORKER_PIDFILE.unlink(missing_ok=True)
    print(f"worker (pid={pid}) остановлен, бежавшие запуски помечены failed")


def worker_state() -> dict:
    """Сводка состояния для UI/CLI: жив ли демон, что в очереди, что реально
    сейчас выполняется. Активные джобы берём из claimed/ (train — по status
    запуска 'running', eval — по eval.pid), плюс любые running-запуски."""
    pid = daemon_pid()
    jobs = queue_mod.list_jobs()
    n_wait = sum(1 for _, j in jobs if queue_mod.dep_state(j) == "waiting")
    active = []
    seen = set()
    for path, job in queue_mod.list_claimed():
        rid = job.get("run_id")
        kind = job.get("kind", "train")
        run_dir = runs.run_dir(rid)
        if kind == "eval":
            live = runs.eval_pid(run_dir) is not None
        else:
            live = runs.get_status(run_dir) == "running"
        active.append({"run_id": rid, "kind": kind,
                       "datasets": job.get("datasets", []), "live": live,
                       "mem": memwatch.read_memory(run_dir)})
        seen.add(rid)
    for r in runs.list_runs():
        if r["status"] == "running" and r["id"] not in seen:
            active.append({"run_id": r["id"], "kind": "train",
                           "datasets": [], "live": True,
                           "mem": memwatch.read_memory(runs.run_dir(r["id"]))})
    return {"pid": pid, "running": bool(pid), "n_jobs": len(jobs),
            "n_waiting": n_wait, "active": active,
            "mem": memwatch.system_snapshot()}


def worker_log_tail(n=200) -> str:
    if not WORKER_LOG.exists():
        return "(worker.log пуст)"
    return "\n".join(WORKER_LOG.read_text(
        encoding="utf-8", errors="replace").splitlines()[-n:])


def daemon_status():
    st = worker_state()
    if st["pid"]:
        print(f"worker: работает (pid={st['pid']})")
    else:
        print("worker: не запущен")
    waiting_note = f" (из них {st['n_waiting']} ждут зависимость)" if st["n_waiting"] else ""
    print(f"очередь: {st['n_jobs']} джобов{waiting_note}")
    mem = st["mem"]
    print(f"память: {mem['percent']}% (занято {memwatch.gb(mem['used'])} из "
          f"{memwatch.gb(mem['limit'])} GB, {mem['source']}), "
          f"warn {memwatch.WARN_PCT:.0f}% / kill {memwatch.KILL_PCT:.0f}%")
    for a in st["active"]:
        detail = f" {','.join(a['datasets'])}" if a["datasets"] else ""
        dead = "" if a["live"] else " (процесс мёртв!)"
        print(f"бежит: [{a['kind']}] {a['run_id']}{detail}{dead}")
