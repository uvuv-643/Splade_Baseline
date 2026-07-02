from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "configs"
SNAPSHOTS_DIR = ROOT / "snapshots"
QUEUE_DIR = ROOT / "queue"
CLAIMED_DIR = QUEUE_DIR / "claimed"
FAILED_JOBS_DIR = QUEUE_DIR / "failed"
RUNS_DIR = ROOT / "runs"
GATE_RUNS_DIR = RUNS_DIR / "_gate"
DATA_DIR = ROOT / "data"
EVAL_DIR = DATA_DIR / "eval"
TRAIN_DIR = DATA_DIR / "train"
CACHE_DIR = DATA_DIR / "cache"
REPORTS_DIR = ROOT / "reports"
MODELS_REGISTRY = RUNS_DIR / "models.json"
WORKER_PIDFILE = QUEUE_DIR / "worker.pid"
WORKER_LOG = QUEUE_DIR / "worker.log"
