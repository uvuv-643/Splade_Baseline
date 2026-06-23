"""Артефакты прогона: outputs/<version>/<run_id>/{config.json,metrics.json,meta.json,model/}."""
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import OUTPUTS_DIR, ROOT


def create_run_dir(version: str, run_id: str = None) -> Path:
    base = run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = OUTPUTS_DIR / version / base
    n = 1
    while run_dir.exists():  # повторный запуск клетки в ту же секунду — не падаем
        n += 1
        run_dir = OUTPUTS_DIR / version / f"{base}-{n}"
    run_dir.mkdir(parents=True)
    return run_dir


def save_config(run_dir: Path, cfg: dict):
    (run_dir / "config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def save_metrics(run_dir: Path, metrics: dict):
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _pkg_versions() -> dict:
    import numpy
    import scipy
    import torch
    import transformers
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
    }


def start_meta(run_dir: Path, cfg: dict, device) -> dict:
    import torch
    cfg_text = (run_dir / "config.json").read_text(encoding="utf-8")
    return {
        "version": cfg["version"],
        "mode": cfg["mode"],
        "run_id": run_dir.name,
        "seed": cfg["train"]["seed"],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": _git_commit(),
        "config_sha256": hashlib.sha256(cfg_text.encode()).hexdigest(),
        "packages": _pkg_versions(),
        "_t0": time.time(),
    }


def finish_meta(run_dir: Path, meta: dict):
    meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    meta["duration_s"] = round(time.time() - meta.pop("_t0"), 1)
    (run_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
