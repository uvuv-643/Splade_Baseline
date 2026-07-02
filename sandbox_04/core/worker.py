import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import config as config_mod
from . import queue as queue_mod
from . import runner, runs
from .paths import QUEUE_DIR, ROOT, RUNS_DIR, WORKER_LOG, WORKER_PIDFILE

POLL_S = 5


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
    run_dir = runs.create_run_dir(job["run_id"])
    (run_dir / "config.yaml").write_text(
        config_mod.dump(job["config"]), encoding="utf-8")
    (run_dir / "snapshot.json").write_text(
        json.dumps(job["snapshot"]), encoding="utf-8")
    runs.set_status(run_dir, "queued")
    proc = runner.start_run_process(run_dir, gpu)
    return proc, run_dir


def worker_loop(gpus: list):
    _log(f"старт, gpus={gpus}")
    adopt_stale()
    active = {}
    stop = {"flag": False}

    def on_term(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    while True:
        for gpu in list(active):
            proc, run_dir, job_path = active[gpu]
            code = proc.poll()
            if code is None:
                continue
            status = runs.get_status(run_dir)
            if status not in ("done", "failed"):
                runs.set_status(run_dir, "failed")
                status = "failed"
            job_path.unlink(missing_ok=True)
            _log(f"{run_dir.name}: {status} (exit={code}, gpu={gpu})")
            del active[gpu]

        if stop["flag"]:
            for gpu, (proc, run_dir, job_path) in active.items():
                _log(f"останавливаю {run_dir.name} (gpu={gpu})")
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    proc.wait(timeout=30)
                except (OSError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except OSError:
                        pass
                runs.set_status(run_dir, "failed")
                job_path.unlink(missing_ok=True)
            _log("остановлен")
            return

        for gpu in gpus:
            if gpu in active:
                continue
            job_path, job = queue_mod.claim_next()
            if job is None:
                break
            proc, run_dir = start_job(job, gpu)
            active[gpu] = (proc, run_dir, job_path)
            _log(f"{run_dir.name}: запущен на gpu={gpu} (pid={proc.pid})")

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


def daemon_status():
    pid = daemon_pid()
    n_queued = len(queue_mod.list_jobs())
    running = [r for r in runs.list_runs() if r["status"] == "running"]
    if pid:
        print(f"worker: работает (pid={pid})")
    else:
        print("worker: не запущен")
    print(f"очередь: {n_queued} джобов")
    for r in running:
        print(f"бежит: {r['id']}")
