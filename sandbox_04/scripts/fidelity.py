import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd
import yaml
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]


def collect(exp: str, metric: str, lite: str, full: str) -> pd.DataFrame:
    rows = []
    for d in sorted((ROOT / "runs").iterdir()):
        cfg_path, pq_path, status_path = d / "config.yaml", d / "per_query.parquet", d / "status"
        if not (cfg_path.exists() and pq_path.exists() and status_path.exists()):
            continue
        if status_path.read_text(encoding="utf-8").strip() != "done":
            continue
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        name = cfg["name"]
        if name != exp and not name.startswith(exp + "@"):
            continue
        df = pd.read_parquet(pq_path)
        means = df[df["metric"] == metric].groupby("dataset")["value"].mean()
        if lite not in means.index or full not in means.index:
            continue
        rows.append({"run": d.name, "system": name, "seed": cfg["train"]["seed"],
                     "lite": means[lite], "full": means[full]})
    return pd.DataFrame(rows)


def pair_deltas(df: pd.DataFrame) -> pd.DataFrame:
    pairs = []
    for (_, a), (_, b) in combinations(df.iterrows(), 2):
        pairs.append({"same_system": a["system"] == b["system"],
                      "d_lite": a["lite"] - b["lite"],
                      "d_full": a["full"] - b["full"]})
    return pd.DataFrame(pairs)


def main():
    ap = argparse.ArgumentParser(description="fidelity lite↔full по готовым runs")
    ap.add_argument("--exp", default="data-scaling")
    ap.add_argument("--metric", default="mrr@10")
    ap.add_argument("--lite", default="msmarco-dev-lite")
    ap.add_argument("--full", default="msmarco-dev")
    ap.add_argument("--min-delta", type=float, default=0.005)
    args = ap.parse_args()

    df = collect(args.exp, args.metric, args.lite, args.full)
    if len(df) < 3:
        raise SystemExit(f"runs с обоими датасетами: {len(df)} — мало, нужно >= 3")

    df = df.sort_values("lite", ascending=False).reset_index(drop=True)
    df["bias"] = df["full"] - df["lite"]
    print(f"=== {args.metric}: {args.lite} vs {args.full} ({len(df)} runs) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    tau, tau_p = sps.kendalltau(df["lite"], df["full"])
    print(f"\nKendall tau по runs:               {tau:.3f} (p={tau_p:.4f})")
    sys_df = df.groupby("system", as_index=False)[["lite", "full"]].mean()
    if len(sys_df) >= 3:
        stau, stau_p = sps.kendalltau(sys_df["lite"], sys_df["full"])
        print(f"Kendall tau по системам (mean сидов): {stau:.3f} (p={stau_p:.4f})")
    print(f"смещение full-lite: mean {df['bias'].mean():+.4f}, std {df['bias'].std():.4f}")

    pairs = pair_deltas(df)
    cross = pairs[~pairs["same_system"]]
    if len(cross):
        slope = (cross["d_lite"] * cross["d_full"]).sum() / (cross["d_lite"] ** 2).sum()
        big = cross[cross["d_lite"].abs() >= args.min_delta]
        print(f"\nмежсистемные пары: {len(cross)}")
        print(f"наклон d_full ~ d_lite (через 0): {slope:.3f}")
        if len(big):
            agree = (big["d_lite"] * big["d_full"] > 0).mean()
            print(f"совпадение знака при |d_lite| >= {args.min_delta}: {agree:.0%} ({len(big)} пар)")

    same = pairs[pairs["same_system"]]
    if len(same):
        print(f"\nшумовой пол ({len(same)} пар сидов внутри систем): "
              f"mean |d_lite| {same['d_lite'].abs().mean():.4f}, "
              f"mean |d_full| {same['d_full'].abs().mean():.4f}")


if __name__ == "__main__":
    main()
