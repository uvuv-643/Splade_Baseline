import json
import time
from pathlib import Path

from .guards import check_finite
from .paths import ROOT


class RunContext:
    def __init__(self, run_dir, device, cache_dir):
        self.root = ROOT
        self.run_dir = Path(run_dir)
        self.model_dir = self.run_dir / "model"
        self.cache_dir = Path(cache_dir)
        self.device = device
        self._log_path = self.run_dir / "train_log.jsonl"

    def log(self, step, **values):
        for key, val in values.items():
            if isinstance(val, (int, float)):
                check_finite(f"{key} (шаг {step})", float(val))
        record = {"step": int(step), **values, "t": round(time.time(), 3)}
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
