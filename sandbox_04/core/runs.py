import json
import os
import shutil
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import config as config_mod
from . import memwatch
from .paths import MODELS_REGISTRY, RUNS_DIR


def new_run_id(name: str, seed: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{name}-s{seed}-{ts}"


def run_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id


def create_run_dir(run_id: str) -> Path:
    d = RUNS_DIR / run_id
    n = 1
    while d.exists():
        n += 1
        d = RUNS_DIR / f"{run_id}-{n}"
    d.mkdir(parents=True)
    return d


def reserve_run(run_id: str, cfg: dict, snap_hash: str, snap_name: str) -> Path:
    """Создаёт каталог запуска заранее (при постановке в очередь): пишет
    config.yaml, snapshot.json и status='queued'. Возвращает каталог с
    фактическим id (если run_id занят — с суффиксом). Так запуск сразу виден в
    `lab status`/UI, а eval-джоб может ссылаться на него по имени каталога для
    цепочки train→eval."""
    d = create_run_dir(run_id)
    (d / "config.yaml").write_text(config_mod.dump(cfg), encoding="utf-8")
    (d / "snapshot.json").write_text(
        json.dumps({"hash": snap_hash, "name": snap_name}), encoding="utf-8")
    set_status(d, "queued")
    return d


def set_status(run_dir, status: str):
    path = Path(run_dir) / "status"
    tmp = path.with_name("status.tmp")
    tmp.write_text(status, encoding="utf-8")
    tmp.replace(path)


def get_status(run_dir) -> str:
    path = Path(run_dir) / "status"
    return path.read_text(encoding="utf-8").strip() if path.exists() else None


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_yaml(path):
    try:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except OSError:
        return {}


def eval_pid(run_dir) -> int:
    """PID бегущего до-eval этого запуска (файл eval.pid пишет runner.execute_eval),
    либо None, если eval не идёт или процесс уже мёртв."""
    path = Path(run_dir) / "eval.pid"
    if not path.exists():
        return None
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None
    return pid if _pid_alive(pid) else None


def run_info(d: Path) -> dict:
    cfg = _read_yaml(d / "config.yaml")
    meta = _read_json(d / "meta.json")
    metrics = _read_json(d / "metrics.json")
    snap = _read_json(d / "snapshot.json")
    key_metrics = {}
    for ds, agg in metrics.get("datasets", {}).items():
        for m in ("mrr@10", "ndcg@10"):
            if m in agg:
                key_metrics[f"{ds}/{m}"] = agg[m]
    status = get_status(d)
    eval_running = eval_pid(d) is not None
    return {
        "id": d.name,
        "dir": d,
        "name": cfg.get("name", ""),
        "seed": (cfg.get("train") or {}).get("seed"),
        "status": status,
        "eval_running": eval_running,
        "mem": memwatch.read_memory(d) if status == "running" or eval_running else {},
        "oom": memwatch.read_oom(d),
        "has_model": (d / "model").is_dir(),
        "has_index": (d / "index").is_dir(),
        "datasets": sorted(metrics.get("datasets", {})),
        "snapshot": snap.get("name") or snap.get("hash", ""),
        "snapshot_hash": snap.get("hash", ""),
        "core_hash": meta.get("core_hash", ""),
        "created": meta.get("started_at", ""),
        "duration_s": meta.get("duration_s"),
        "metrics": key_metrics,
    }


def list_runs(include_gate=False) -> list:
    if not RUNS_DIR.exists():
        return []
    out = []
    for d in RUNS_DIR.iterdir():
        if not d.is_dir():
            continue
        if d.name.startswith("_") and not include_gate:
            continue
        if not (d / "config.yaml").exists():
            continue
        out.append(run_info(d))
    return sorted(out, key=lambda r: r["created"] or r["id"], reverse=True)


def resolve_run(ref: str) -> Path:
    d = RUNS_DIR / ref
    if (d / "config.yaml").exists():
        return d
    registry = _read_json(MODELS_REGISTRY)
    if ref in registry:
        return RUNS_DIR / registry[ref]["run_id"]
    matches = [p for p in RUNS_DIR.iterdir()
               if p.is_dir() and p.name.startswith(ref) and (p / "config.yaml").exists()]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"запуск {ref!r} не найден"
                            + (f" (кандидатов: {len(matches)})" if matches else ""))


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def kill_run(ref: str) -> bool:
    d = resolve_run(ref)
    pid_file = d / "pid"
    if not pid_file.exists():
        return False
    pid = int(pid_file.read_text())
    if not _pid_alive(pid):
        return False
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except OSError:
        os.kill(pid, signal.SIGTERM)
    for _ in range(20):
        if not _pid_alive(pid):
            break
        time.sleep(0.5)
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except OSError:
            pass
    set_status(d, "failed")
    return True


def kill_eval(ref: str) -> bool:
    """Останавливает бегущий до-eval запуска (по eval.pid). Статус самого
    train-запуска не трогаем — он остаётся 'done'. Файл eval.pid снимается
    самим процессом в finally, но подчистим и здесь на случай SIGKILL."""
    d = resolve_run(ref)
    pid = eval_pid(d)
    if pid is None:
        return False
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except OSError:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    for _ in range(20):
        if not _pid_alive(pid):
            break
        time.sleep(0.5)
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except OSError:
            pass
    (d / "eval.pid").unlink(missing_ok=True)
    return True


def delete_run(ref: str) -> bool:
    """Полностью удаляет каталог запуска. Отказывается, если запуск сейчас
    бежит (train-статус 'running' или идёт eval) — сначала останови его."""
    d = resolve_run(ref)
    if get_status(d) == "running" or eval_pid(d) is not None:
        raise RuntimeError(f"{d.name} сейчас выполняется — сначала останови (kill)")
    shutil.rmtree(d)
    return True


def requeue_run(ref: str) -> str:
    """«Оживляет» упавший (или иным образом завершённый) запуск: ставит его
    train-джоб обратно в очередь, ссылаясь на тот же каталог. Каталог уже несёт
    config.yaml + snapshot.json, поэтому worker.start_job/runner.execute просто
    перезапустят обучение с теми же параметрами и снапшотом кода — id запуска и
    его история (stdout.log и т.п.) сохраняются, метрики/модель перезапишутся.

    Отказывается, если запуск сейчас бежит или уже стоит в очереди (status
    'running'/'queued' или идёт eval) — оживлять нечего. Возвращает run_id."""
    from . import queue as queue_mod
    d = resolve_run(ref)
    status = get_status(d)
    if status == "running" or eval_pid(d) is not None:
        raise RuntimeError(f"{d.name} сейчас выполняется — оживлять нечего")
    if status == "queued":
        raise RuntimeError(f"{d.name} уже в очереди")
    cfg = _read_yaml(d / "config.yaml")
    if not cfg:
        raise FileNotFoundError(f"в {d} нет config.yaml — нечего перезапускать")
    snap = _read_json(d / "snapshot.json")
    set_status(d, "queued")
    queue_mod.enqueue_train(cfg, snap.get("hash", ""), snap.get("name", ""), d.name)
    return d.name


def name_model(run_ref: str, name: str, message=""):
    d = resolve_run(run_ref)
    if not (d / "model").is_dir():
        raise FileNotFoundError(f"в {d} нет model/")
    registry = _read_json(MODELS_REGISTRY)
    registry[name] = {"run_id": d.name, "description": message,
                      "created": datetime.now(timezone.utc).isoformat()}
    MODELS_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    MODELS_REGISTRY.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")


def list_models() -> dict:
    return _read_json(MODELS_REGISTRY)
