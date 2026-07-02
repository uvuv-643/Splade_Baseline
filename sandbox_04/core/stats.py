import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import optimize, special
from scipy import stats as sps

N_PERMUTATIONS = 10_000
N_BOOTSTRAP = 10_000
ALPHA = 0.05
EXACT_LIMIT = 20
DISPLAY_METRICS = ("mrr@10", "ndcg@10", "recall@10", "recall@100", "recall@1000")


def primary_metrics(dataset: str) -> tuple:
    if dataset.startswith("trec-dl"):
        return ("ndcg@10",)
    if dataset.startswith("msmarco-dev"):
        return ("mrr@10", "ndcg@10")
    return ("mrr@10",)


def fisher_randomization(diff, n_perm=N_PERMUTATIONS, seed=0) -> float:
    """Парный рандомизационный тест Фишера, двусторонний: под H0 знак каждой
    парной разности случаен, статистика |mean|. При n <= EXACT_LIMIT
    перебираются все 2^n расстановок знаков (точный тест), иначе Monte-Carlo
    с несмещённой оценкой p = (1+k)/(1+B)."""
    d = np.asarray(diff, dtype=np.float64)
    n = d.size
    if n == 0:
        return float("nan")
    observed = abs(d.mean())
    if n <= EXACT_LIMIT:
        bits = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(np.int8)
        means = np.abs((2 * bits - 1) @ d) / n
        return float((means >= observed - 1e-12).mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n))
    means = np.abs(signs @ d) / n
    return float((1 + (means >= observed - 1e-12).sum()) / (1 + n_perm))


def paired_ttest(diff) -> float:
    d = np.asarray(diff, dtype=np.float64)
    if d.size < 2:
        return float("nan")
    res = sps.ttest_1samp(d, 0.0)
    return float(res.pvalue)


def holm(pvalues: dict, alpha=ALPHA) -> dict:
    """Поправка Холма: p_(i) умножается на (m-i+1), с монотонизацией."""
    items = sorted(((k, p) for k, p in pvalues.items() if p == p),
                   key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for rank, (key, p) in enumerate(items):
        p_adj = max(prev, min(1.0, (m - rank) * p))
        prev = p_adj
        out[key] = {"p": p, "p_adj": p_adj, "reject": bool(p_adj < alpha)}
    for key, p in pvalues.items():
        if p != p:
            out[key] = {"p": p, "p_adj": float("nan"), "reject": False}
    return out


def bca_interval(diff, n_boot=N_BOOTSTRAP, alpha=ALPHA, seed=0):
    """BCa-интервал (Efron) для mean(diff): z0 корректирует смещение
    (доля bootstrap-средних ниже наблюдаемого), a — ускорение из
    jackknife-асимметрии: a = Σu³ / (6 (Σu²)^{3/2}), u_i = θ̄_jack − θ_(i)."""
    d = np.asarray(diff, dtype=np.float64)
    n = d.size
    theta = float(d.mean()) if n else float("nan")
    if n < 2 or np.allclose(d, d[0]):
        return theta, theta, theta
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    prop = np.clip((boot < theta).mean(), 1 / (n_boot + 1), n_boot / (n_boot + 1))
    z0 = special.ndtri(prop)
    jack = (d.sum() - d) / (n - 1)
    u = jack.mean() - jack
    denom = float((u ** 2).sum()) ** 1.5
    a = float((u ** 3).sum()) / (6 * denom) if denom > 0 else 0.0
    z = special.ndtri(np.array([alpha / 2, 1 - alpha / 2]))
    q = special.ndtr(z0 + (z0 + z) / (1 - a * (z0 + z)))
    lo, hi = np.quantile(boot, q)
    return theta, float(lo), float(hi)


def load_per_query(run_dir) -> pd.DataFrame:
    return pd.read_parquet(Path(run_dir) / "per_query.parquet")


def seed_average(dfs: list) -> pd.DataFrame:
    """Per-query метрика усредняется по сидам (выравнивание по dataset+qid+metric).
    Требование протокола: одинаковые наборы qid у всех сидов."""
    keys = [frozenset(map(tuple, df[["dataset", "qid", "metric"]].itertuples(index=False)))
            for df in dfs]
    if len(set(keys)) > 1:
        raise ValueError("qid-сетки сидов не совпадают — eval-данные разные")
    return (pd.concat(dfs)
            .groupby(["dataset", "qid", "metric"], as_index=False)["value"].mean())


def _config_leaves(cfg, prefix=""):
    out = {}
    for key, value in cfg.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_config_leaves(value, path))
        elif key not in ("name", "description", "seed", "seeds"):
            out[path] = value
    return out


def varying_numeric_param(configs: dict):
    leaves = {system: _config_leaves(cfg) for system, cfg in configs.items()}
    all_paths = set().union(*(set(l) for l in leaves.values()))
    varying = []
    for path in sorted(all_paths):
        values = {system: l.get(path) for system, l in leaves.items()}
        if len({repr(v) for v in values.values()}) > 1:
            varying.append((path, values))
    numeric = [(p, v) for p, v in varying
               if all(isinstance(x, (int, float)) and not isinstance(x, bool)
                      for x in v.values())]
    if len(numeric) == 1 and len(varying) == 1:
        return numeric[0]
    return None


def fit_scaling(x, y):
    """Насыщающийся степенной закон m(N) = a − b·N^(−γ): a — асимптотика
    метрики при N→∞, γ — скорость выхода на плато. При фиксированном γ модель
    линейна по (a, b), поэтому γ профилируется по сетке (точное МНК-решение
    в каждой точке), лучший старт полируется bounded curve_fit — это убирает
    локальные минимумы слабо идентифицируемой трёхпараметрической модели."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(np.unique(x)) < 3:
        return None
    best = None
    for g0 in np.geomspace(0.02, 2.0, 60):
        design = np.column_stack([np.ones_like(x), -(x ** (-g0))])
        (a0, b0), residual, *_ = np.linalg.lstsq(design, y, rcond=None)
        if b0 < 0 or not (0 <= a0 <= 1):
            continue
        sse = float(((design @ [a0, b0] - y) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, float(a0), float(b0), float(g0))
    if best is None:
        return None
    _, a0, b0, g0 = best
    try:
        popt, _ = optimize.curve_fit(
            lambda n, a, b, g: a - b * n ** (-g), x, y,
            p0=(a0, max(b0, 1e-9), g0),
            bounds=([0.0, 0.0, 0.01], [1.0, np.inf, 2.0]), maxfev=20000)
        a, b, g = (float(v) for v in popt)
    except (RuntimeError, ValueError):
        a, b, g = a0, b0, g0
    pred = a - b * x ** (-g)
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"a": a, "b": b, "gamma": g,
            "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")}


def kendall_trend(x, y):
    tau, p = sps.kendalltau(x, y)
    return {"tau": float(tau), "p": float(p)}


def _read_run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    metrics_js = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    return {"dir": run_dir, "id": run_dir.name, "config": cfg, "meta": meta,
            "metrics": metrics_js, "per_query": load_per_query(run_dir)}


def compare_systems(system_runs: dict, datasets=None,
                    alpha=ALPHA, n_perm=N_PERMUTATIONS, n_boot=N_BOOTSTRAP) -> dict:
    """system_runs: {имя_системы: [run_dir, ...]} — сиды одной системы вместе.
    Вся статистика — только из per_query.parquet (без GPU и индекса)."""
    systems = {name: [_read_run(d) for d in dirs]
               for name, dirs in system_runs.items()}
    if len(systems) < 1:
        raise ValueError("нечего сравнивать")

    warnings = []
    core_hashes = {r["meta"].get("core_hash")
                   for runs in systems.values() for r in runs}
    if len(core_hashes) > 1:
        warnings.append(f"разные core_hash: {sorted(core_hashes)} — запуски из разных «эпох» ядра")

    common_datasets = None
    for runs in systems.values():
        for r in runs:
            ds = set(r["metrics"]["datasets"])
            common_datasets = ds if common_datasets is None else common_datasets & ds
    datasets = sorted(common_datasets if datasets is None
                      else set(datasets) & common_datasets)
    for ds in datasets:
        hashes = {r["metrics"]["datasets"][ds].get("eval_data_hash")
                  for runs in systems.values() for r in runs}
        if len(hashes) > 1:
            warnings.append(f"{ds}: разные eval_data_hash {sorted(hashes)} — сравнение невалидно")

    averaged = {}
    for name, runs in systems.items():
        try:
            averaged[name] = seed_average([r["per_query"] for r in runs])
        except ValueError as e:
            warnings.append(f"{name}: {e}")
            averaged[name] = pd.concat([r["per_query"] for r in runs]).groupby(
                ["dataset", "qid", "metric"], as_index=False)["value"].mean()

    report = {"alpha": alpha, "warnings": warnings, "datasets": {},
              "systems": {name: {"runs": [r["id"] for r in runs],
                                 "seeds": [r["config"]["train"]["seed"] for r in runs]}
                          for name, runs in systems.items()}}

    all_pvalues = {}
    for ds in datasets:
        ds_report = {"aggregates": {}, "pairwise": []}
        present_metrics = [m for m in DISPLAY_METRICS
                           if all(((avg["dataset"] == ds) & (avg["metric"] == m)).any()
                                  for avg in averaged.values())]
        vectors = {}
        for metric in present_metrics:
            per_system = {}
            for name, avg in averaged.items():
                sub = avg[(avg["dataset"] == ds) & (avg["metric"] == metric)]
                per_system[name] = dict(zip(sub["qid"], sub["value"]))
            common_qids = sorted(set.intersection(*(set(v) for v in per_system.values())))
            dropped = max(len(v) for v in per_system.values()) - len(common_qids)
            if dropped:
                warnings.append(f"{ds}/{metric}: {dropped} qid вне пересечения систем")
            vectors[metric] = {name: np.array([per_system[name][q] for q in common_qids])
                               for name in systems}

        for name, runs in systems.items():
            agg = {}
            for metric in present_metrics:
                per_seed = [pq[(pq["dataset"] == ds) &
                               (pq["metric"] == metric)]["value"].mean()
                            for pq in (r["per_query"] for r in runs)]
                agg[metric] = {"mean": float(np.mean(per_seed)),
                               "std": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0,
                               "n_seeds": len(per_seed)}
            ds_report["aggregates"][name] = agg

        for a_name, b_name in itertools.combinations(systems, 2):
            for metric in present_metrics:
                a, b = vectors[metric][a_name], vectors[metric][b_name]
                d = b - a
                delta, lo, hi = bca_interval(d, n_boot=n_boot, alpha=alpha)
                p_fisher = fisher_randomization(d, n_perm=n_perm)
                p_t = paired_ttest(d)
                key = f"{ds}|{metric}|{a_name}|{b_name}"
                all_pvalues[key] = p_fisher
                ds_report["pairwise"].append({
                    "a": a_name, "b": b_name, "metric": metric,
                    "n_queries": int(d.size), "delta": delta,
                    "ci95_low": lo, "ci95_high": hi,
                    "p_fisher": p_fisher, "p_ttest": p_t,
                })
        report["datasets"][ds] = ds_report

    adjusted = holm(all_pvalues, alpha=alpha)
    for ds in datasets:
        for pw in report["datasets"][ds]["pairwise"]:
            key = f"{ds}|{pw['metric']}|{pw['a']}|{pw['b']}"
            pw["p_holm"] = adjusted[key]["p_adj"]
            pw["significant"] = adjusted[key]["reject"]

    configs = {name: runs[0]["config"] for name, runs in systems.items()}
    param = varying_numeric_param(configs)
    if param is not None and len(systems) >= 3:
        path, values = param
        scaling = {"param": path, "values": values, "datasets": {}}
        for ds in datasets:
            per_metric = {}
            for metric in primary_metrics(ds):
                xs, ys = [], []
                for name, runs in systems.items():
                    for r in runs:
                        pq = r["per_query"]
                        sub = pq[(pq["dataset"] == ds) & (pq["metric"] == metric)]
                        if len(sub):
                            xs.append(float(values[name]))
                            ys.append(float(sub["value"].mean()))
                if len(xs) >= 3:
                    per_metric[metric] = {
                        "points": sorted(zip(xs, ys)),
                        "fit": fit_scaling(xs, ys),
                        "trend": kendall_trend(xs, ys),
                    }
            if per_metric:
                scaling["datasets"][ds] = per_metric
        report["scaling"] = scaling
    return report


def _fmt_p(p):
    if p != p:
        return "—"
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"


def render_markdown(report: dict) -> str:
    lines = ["# Сравнение систем", ""]
    if report["warnings"]:
        lines.append("## Предупреждения")
        lines.extend(f"- ⚠️ {w}" for w in report["warnings"])
        lines.append("")
    lines.append("## Системы")
    for name, info in report["systems"].items():
        lines.append(f"- **{name}**: сиды {info['seeds']}, запуски {info['runs']}")
    lines.append("")
    for ds, ds_rep in report["datasets"].items():
        lines.append(f"## {ds}")
        lines.append("")
        prim = set(primary_metrics(ds))
        metrics_list = list(next(iter(ds_rep["aggregates"].values())).keys())
        header = "| система | " + " | ".join(
            f"**{m}**" if m in prim else m for m in metrics_list) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(metrics_list) + 1))
        for name, agg in ds_rep["aggregates"].items():
            cells = [f"{agg[m]['mean']:.4f} ± {agg[m]['std']:.4f}" for m in metrics_list]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
        if ds_rep["pairwise"]:
            lines.append("| пара (B−A) | метрика | Δ | 95% CI (BCa) | p Fisher | p t-test | p Holm | значимо |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for pw in ds_rep["pairwise"]:
                sig = "**да**" if pw["significant"] else "нет"
                lines.append(
                    f"| {pw['b']} − {pw['a']} | {pw['metric']} | {pw['delta']:+.4f} "
                    f"| [{pw['ci95_low']:+.4f}, {pw['ci95_high']:+.4f}] "
                    f"| {_fmt_p(pw['p_fisher'])} | {_fmt_p(pw['p_ttest'])} "
                    f"| {_fmt_p(pw['p_holm'])} | {sig} |")
            lines.append("")
    scaling = report.get("scaling")
    if scaling:
        lines.append(f"## Скейлинг по `{scaling['param']}`")
        lines.append("")
        for ds, per_metric in scaling["datasets"].items():
            for metric, info in per_metric.items():
                lines.append(f"### {ds} / {metric}")
                fit = info["fit"]
                if fit:
                    lines.append(
                        f"- аппроксимация m(N) = a − b·N^(−γ): "
                        f"a={fit['a']:.4f}, b={fit['b']:.4f}, γ={fit['gamma']:.3f}, "
                        f"R²={fit['r2']:.3f}")
                tr = info["trend"]
                lines.append(f"- тренд Кендалла: τ={tr['tau']:.3f}, p={_fmt_p(tr['p'])}")
                lines.append("- точки (N, метрика): " + ", ".join(
                    f"({int(x)}, {y:.4f})" for x, y in info["points"]))
                lines.append("")
    return "\n".join(lines)


def save_scaling_plots(report: dict, out_dir: Path) -> list:
    scaling = report.get("scaling")
    if not scaling:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for ds, per_metric in scaling["datasets"].items():
        for metric, info in per_metric.items():
            fig, ax = plt.subplots(figsize=(6, 4))
            xs = np.array([p[0] for p in info["points"]])
            ys = np.array([p[1] for p in info["points"]])
            ax.scatter(xs, ys, s=25, alpha=0.7, label="запуски (сиды)")
            fit = info["fit"]
            if fit:
                grid = np.geomspace(xs.min(), xs.max(), 200)
                ax.plot(grid, fit["a"] - fit["b"] * grid ** (-fit["gamma"]),
                        "r-", label=f"a−b·N^(−γ), γ={fit['gamma']:.2f}, R²={fit['r2']:.2f}")
            ax.set_xscale("log")
            ax.set_xlabel(scaling["param"])
            ax.set_ylabel(metric)
            ax.set_title(ds)
            ax.legend()
            fig.tight_layout()
            path = out_dir / f"scaling_{ds}_{metric.replace('@', '')}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            saved.append(path)
    return saved
