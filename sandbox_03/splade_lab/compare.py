"""Сводная таблица метрик всех прогонов из outputs/."""
import json
from pathlib import Path

import pandas as pd

from .config import OUTPUTS_DIR

METRIC_COLS = ["mrr@10", "recall@10", "recall@100", "recall@1000",
               "avg_nnz_doc", "avg_nnz_query", "final_train_loss",
               "train_steps", "n_eval_queries", "n_corpus_docs", "duration_s"]


def collect_runs(outputs_dir=OUTPUTS_DIR) -> pd.DataFrame:
    rows = []
    for metrics_path in sorted(Path(outputs_dir).glob("*/*/metrics.json")):
        run_dir = metrics_path.parent
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        meta_path = run_dir / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        rows.append({
            "version": meta.get("version", run_dir.parent.name),
            "run_id": run_dir.name,
            "mode": meta.get("mode"),
            "seed": meta.get("seed"),
            "duration_s": meta.get("duration_s"),
            **metrics,
        })
    return pd.DataFrame(rows)


def compare_runs(outputs_dir=OUTPUTS_DIR, save_csv=None) -> pd.DataFrame:
    df = collect_runs(outputs_dir)
    if df.empty:
        print(f"Нет прогонов в {outputs_dir}")
        return df
    cols = ["version", "run_id", "mode", "seed"] + [c for c in METRIC_COLS if c in df.columns]
    df = df[cols].sort_values(["mode", "version", "run_id"]).reset_index(drop=True)
    save_csv = Path(save_csv) if save_csv else Path(outputs_dir) / "comparison.csv"
    save_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_csv, index=False)
    print(df.to_string(index=False))
    print(f"\n[compare] CSV: {save_csv}")
    return df
