import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from . import config as config_mod
from . import data, runner, runs, snapshots
from .paths import GATE_RUNS_DIR

GATE_TIMEOUT_S = 900
GATE_KEEP = 5

GATE_OVERRIDES = {
    "data": {"train_triples": 256},
    "train": {"max_steps": 8, "warmup_steps": 2, "flops_warmup_steps": 4,
              "batch_size": 8, "log_every": 2},
    "eval": {"datasets": ["gate"], "max_queries": 50, "save_index": False},
}


def gate_config(cfg: dict) -> dict:
    g = config_mod.deep_merge(cfg, GATE_OVERRIDES)
    g["train"].pop("seeds", None)
    g["train"]["seed"] = 1
    g["name"] = cfg["name"] + "-gate"
    return g


def _prune_old():
    if not GATE_RUNS_DIR.exists():
        return
    dirs = sorted((d for d in GATE_RUNS_DIR.iterdir() if d.is_dir()),
                  key=lambda d: d.name)
    for d in dirs[:-GATE_KEEP]:
        shutil.rmtree(d, ignore_errors=True)


def run_gate(cfg: dict, snap_hash: str, snap_name: str):
    """Микро-прогон полного пайплайна train→index→search→metrics из снапшота.
    Возвращает (ok, run_dir). Полный корпус здесь не кодируется никогда —
    eval только на замороженном наборе 'gate' (1k доков / 50 запросов)."""
    missing = data.datasets_prepared(["gate"])
    if missing:
        raise RuntimeError(f"gate-данные не готовы: {missing} — "
                           f"запустите ./lab data prepare")
    gcfg = gate_config(cfg)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = GATE_RUNS_DIR / f"{cfg['name']}-{ts}"
    n = 1
    while run_dir.exists():
        n += 1
        run_dir = GATE_RUNS_DIR / f"{cfg['name']}-{ts}-{n}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(config_mod.dump(gcfg), encoding="utf-8")
    (run_dir / "snapshot.json").write_text(
        json.dumps({"hash": snap_hash, "name": snap_name}), encoding="utf-8")
    runs.set_status(run_dir, "queued")
    code = runner.wait_run_process(run_dir, timeout=GATE_TIMEOUT_S)
    ok = code == 0
    snapshots.mark_gate(snap_hash, ok)
    _prune_old()
    return ok, run_dir
