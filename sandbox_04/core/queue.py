import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .paths import CLAIMED_DIR, FAILED_JOBS_DIR, QUEUE_DIR


def _prefix() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")


def _write_job(job: dict, file_stub: str) -> Path:
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    path = QUEUE_DIR / f"{_prefix()}__{file_stub}.yaml"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(yaml.safe_dump(job, allow_unicode=True, sort_keys=False),
                   encoding="utf-8")
    tmp.rename(path)
    return path


def enqueue_train(cfg_single_seed: dict, snap_hash: str, snap_name: str,
                  run_id: str) -> Path:
    """Джоб обучения: worker прогонит train→index→eval из снапшота.
    run_dir с этим run_id уже зарезервирован (runs.reserve_run) — джоб лишь
    указывает на него."""
    job = {"kind": "train",
           "run_id": run_id,
           "snapshot": {"hash": snap_hash, "name": snap_name},
           "created": datetime.now(timezone.utc).isoformat(),
           "config": cfg_single_seed}
    return _write_job(job, run_id)


def enqueue_eval(run_id: str, datasets, depends_on=None, save_index=False) -> Path:
    """Джоб до-eval: построить индекс + посчитать метрики на новых датасетах
    для уже обученного run_id, прямо в его каталоге. Если depends_on задан —
    worker подождёт, пока тот запуск не станет 'done' (цепочка train→eval)."""
    job = {"kind": "eval",
           "run_id": run_id,
           "datasets": list(datasets),
           "depends_on": depends_on,
           "save_index": bool(save_index),
           "created": datetime.now(timezone.utc).isoformat()}
    return _write_job(job, f"{run_id}-eval")


# обратная совместимость со старым именем
enqueue = enqueue_train


def list_jobs() -> list:
    if not QUEUE_DIR.exists():
        return []
    out = []
    for path in sorted(QUEUE_DIR.glob("*.yaml")):
        try:
            job = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        out.append((path, job))
    return out


def dep_state(job: dict) -> str:
    """Готов ли джоб к запуску по своей зависимости:
    'ready' — можно брать; 'waiting' — ждём завершения; 'dep_failed' — зависимость упала."""
    dep = job.get("depends_on")
    if not dep:
        return "ready"
    from . import runs
    status = runs.get_status(runs.run_dir(dep))
    if status == "done":
        return "ready"
    if status == "failed":
        return "dep_failed"
    return "waiting"


def claim_next():
    """Атомарно забирает первый готовый джоб. Джобы, ждущие зависимость
    (eval после ещё не завершённого train), пропускаются — очередь не встаёт
    в голову: следующий независимый джоб будет взят."""
    CLAIMED_DIR.mkdir(parents=True, exist_ok=True)
    for path, job in list_jobs():
        if dep_state(job) == "waiting":
            continue
        target = CLAIMED_DIR / path.name
        try:
            os.rename(path, target)
        except OSError:
            continue
        return target, job
    return None, None


def requeue_claimed():
    if not CLAIMED_DIR.exists():
        return 0
    n = 0
    for path in sorted(CLAIMED_DIR.glob("*.yaml")):
        path.rename(QUEUE_DIR / path.name)
        n += 1
    return n


def _find(job_ref: str):
    for path, job in list_jobs():
        if job.get("run_id") == job_ref or path.name == job_ref:
            return path, job
    raise FileNotFoundError(f"job {job_ref!r} не найден в очереди")


def remove(job_ref: str):
    path, _ = _find(job_ref)
    path.unlink()


def move(job_ref: str, direction: int):
    jobs = list_jobs()
    names = [p.name for p, _ in jobs]
    path, _ = _find(job_ref)
    i = names.index(path.name)
    j = i + direction
    if not (0 <= j < len(jobs)):
        return
    a, b = jobs[i][0], jobs[j][0]
    pa, ra = a.name.split("__", 1)
    pb, rb = b.name.split("__", 1)
    tmp = a.with_name(a.name + ".swap")
    a.rename(tmp)
    b.rename(b.with_name(f"{pa}__{rb}"))
    tmp.rename(a.with_name(f"{pb}__{ra}"))


def fail_job(job_path: Path, reason: str):
    FAILED_JOBS_DIR.mkdir(parents=True, exist_ok=True)
    target = FAILED_JOBS_DIR / job_path.name
    job_path.rename(target)
    target.with_suffix(".log").write_text(reason, encoding="utf-8")
