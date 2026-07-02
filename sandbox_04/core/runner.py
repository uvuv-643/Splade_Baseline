import json
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from . import contract, data, runs
from . import eval as eval_mod
from .context import RunContext
from .hashing import core_hash, sha256_text
from .paths import CACHE_DIR, ROOT, SNAPSHOTS_DIR


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _versions() -> dict:
    import scipy
    import transformers
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__}


def _write_meta(run_dir: Path, meta: dict):
    (run_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def execute(run_dir) -> None:
    run_dir = Path(run_dir).resolve()
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    snap = json.loads((run_dir / "snapshot.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str(SNAPSHOTS_DIR / snap["hash"]))

    (run_dir / "pid").write_text(str(os.getpid()), encoding="utf-8")
    runs.set_status(run_dir, "running")
    device = pick_device()
    t0 = time.time()
    meta = {
        "run_id": run_dir.name,
        "name": cfg["name"],
        "seed": cfg["train"]["seed"],
        "snapshot": snap,
        "core_hash": core_hash(),
        "eval_data_hash": data.eval_data_hash(cfg["eval"]["datasets"]),
        "git_sha": _git_sha(),
        "host": platform.node(),
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "config_sha256": sha256_text((run_dir / "config.yaml").read_text(encoding="utf-8")),
        "packages": _versions(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_meta(run_dir, meta)
    print(f"[run] {run_dir.name} device={device} snapshot={snap['hash']}", flush=True)

    try:
        set_seed(cfg["train"]["seed"])
        ctx = RunContext(run_dir, device, CACHE_DIR)
        import exp.train
        t_train = time.time()
        encoder = exp.train.train(cfg, ctx)
        meta["train_s"] = round(time.time() - t_train, 1)

        contract.check_encoder_contract(encoder)
        result, per_query = eval_mod.run_eval(
            encoder, cfg["eval"]["datasets"], cfg["eval"], device, run_dir)
        per_query.to_parquet(run_dir / "per_query.parquet", index=False)
        (run_dir / "metrics.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

        meta["finished_at"] = datetime.now(timezone.utc).isoformat()
        meta["duration_s"] = round(time.time() - t0, 1)
        _write_meta(run_dir, meta)
        runs.set_status(run_dir, "done")
        print(f"[run] готово за {meta['duration_s']}s", flush=True)
    except Exception:
        traceback.print_exc()
        meta["finished_at"] = datetime.now(timezone.utc).isoformat()
        meta["duration_s"] = round(time.time() - t0, 1)
        _write_meta(run_dir, meta)
        runs.set_status(run_dir, "failed")
        raise SystemExit(1)


def start_run_process(run_dir, gpu=None) -> subprocess.Popen:
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log = open(Path(run_dir) / "stdout.log", "a", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "core.runner", str(run_dir)],
        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True)
    log.close()
    return proc


def run_foreground(run_dir) -> int:
    env = os.environ.copy()
    proc = subprocess.Popen(
        [sys.executable, "-m", "core.runner", str(run_dir)],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True)
    with open(Path(run_dir) / "stdout.log", "a", encoding="utf-8") as log:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
    return proc.wait()


def wait_run_process(run_dir, gpu=None, timeout=None) -> int:
    proc = start_run_process(run_dir, gpu)
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        import signal as _signal
        try:
            os.killpg(os.getpgid(proc.pid), _signal.SIGKILL)
        except OSError:
            pass
        proc.wait()
        return -1


def main():
    execute(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
