"""Статистическая значимость различий между версиями модели на одном наборе.

Зачем
-----
Хочется не просто видеть, что у версии B nDCG@10 выше, чем у A, а понимать,
**значимо** ли это или в пределах шума. Для IR это стандартная практика: метрику
считают по каждому запросу, а версии сравнивают **парно** (на одних и тех же
запросах того же набора), что убирает разброс «лёгкие/трудные запросы» и сильно
повышает чувствительность теста.

Какие тесты
-----------
Для двух моделей на N запросах сравниваем парные векторы метрики (например
nDCG@10) тремя взаимодополняющими способами:

1. **Парный bootstrap** по запросам (10k ресэмплов) — даёт доверительный
   интервал для средней разницы Δ и двусторонний p (доля ресэмплов, где знак Δ
   меняется). Непараметрический, не требует нормальности, корректно работает на
   маленьких наборах (SciFact ~300 запросов). Это де-факто стандарт в IR
   (Sakai; пакет ranx использует ровно его).
2. **Парный t-тест** (scipy.stats.ttest_rel) — классический, чувствителен при
   близком к нормальному распределении разниц.
3. **Wilcoxon signed-rank** (scipy.stats.wilcoxon) — непараметрическая
   альтернатива t-тесту, на случай тяжёлых хвостов / выбросов.

Если все три согласованно дают p < alpha — вывод об улучшении надёжен.
Сравнения многих версий стоит поправлять на множественность (Холм–Бонферрони,
``holm_correction``).

Зависимости: numpy (есть) + scipy (есть в requirements). Bootstrap реализован
сам и от scipy не зависит — работает и без него.
"""
import numpy as np

try:
    from scipy import stats as _scipy_stats
    _HAVE_SCIPY = True
except Exception:  # scipy опционален: bootstrap работает и без него
    _HAVE_SCIPY = False


def _as_pair(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Векторы метрик разной длины: {a.shape} vs {b.shape}. "
                         "Сравнивать можно только парно, на одних и тех же запросах.")
    if a.ndim != 1:
        raise ValueError("Ожидаются одномерные векторы (метрика по запросам).")
    return a, b


def align_per_query(pq_a: dict, pq_b: dict, metric: str):
    """Выравнивает две раскладки per_query по общим запросам (по qid).

    benchmark.per_query_metrics возвращает 'qids' и векторы метрик в том же
    порядке. Здесь берём пересечение qid и возвращаем парные векторы в едином
    порядке — чтобы парные тесты сравнивали один и тот же запрос с самим собой.
    """
    qa, qb = pq_a["qids"], pq_b["qids"]
    if metric not in pq_a or metric not in pq_b:
        raise KeyError(f"Нет метрики {metric!r} в per_query (есть: "
                       f"{sorted(set(pq_a) & set(pq_b) - {'qids'})})")
    idx_b = {q: i for i, q in enumerate(qb)}
    va, vb, qids = [], [], []
    for i, q in enumerate(qa):
        j = idx_b.get(q)
        if j is not None:
            va.append(pq_a[metric][i])
            vb.append(pq_b[metric][j])
            qids.append(q)
    return np.asarray(va), np.asarray(vb), qids


def paired_bootstrap(a, b, n_boot=10000, alpha=0.05, seed=0) -> dict:
    """Парный bootstrap по запросам для разницы средних Δ = mean(b) - mean(a).

    Возвращает наблюдаемую Δ, доверительный интервал уровня (1-alpha) и
    двусторонний p-value (по методу «доли ресэмплов с противоположным знаком»).
    """
    a, b = _as_pair(a, b)
    diff = b - a
    n = diff.shape[0]
    if n == 0:
        return {"delta": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "p_value": float("nan"),
                "n": 0, "n_boot": n_boot}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)
    observed = float(diff.mean())
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    # двусторонний p: насколько часто bootstrap-среднее по знаку противоречит
    # наблюдаемому (центрируем на 0 — гипотеза об отсутствии разницы)
    centered = boot_means - observed
    p_one = np.mean(centered >= abs(observed)) if observed >= 0 else np.mean(centered <= -abs(observed))
    p_value = float(min(1.0, 2.0 * p_one))
    return {"delta": observed, "ci_low": lo, "ci_high": hi,
            "p_value": p_value, "n": n, "n_boot": n_boot}


def paired_ttest(a, b) -> dict:
    """Парный t-тест (scipy.stats.ttest_rel). Δ = mean(b) - mean(a)."""
    a, b = _as_pair(a, b)
    if not _HAVE_SCIPY:
        return {"delta": float(np.mean(b - a)), "t_stat": None,
                "p_value": None, "note": "scipy недоступен"}
    res = _scipy_stats.ttest_rel(b, a)
    return {"delta": float(np.mean(b - a)),
            "t_stat": float(res.statistic), "p_value": float(res.pvalue)}


def wilcoxon_test(a, b) -> dict:
    """Парный Wilcoxon signed-rank (scipy.stats.wilcoxon)."""
    a, b = _as_pair(a, b)
    if not _HAVE_SCIPY:
        return {"delta": float(np.mean(b - a)), "stat": None,
                "p_value": None, "note": "scipy недоступен"}
    diff = b - a
    if np.allclose(diff, 0):
        return {"delta": 0.0, "stat": None, "p_value": 1.0,
                "note": "все разницы нулевые"}
    res = _scipy_stats.wilcoxon(b, a, zero_method="wilcox", correction=False,
                                alternative="two-sided", mode="auto")
    return {"delta": float(np.mean(diff)),
            "stat": float(res.statistic), "p_value": float(res.pvalue)}


def compare_pair(pq_a: dict, pq_b: dict, metric: str = "ndcg@10",
                 name_a: str = "A", name_b: str = "B",
                 alpha: float = 0.05, n_boot: int = 10000, seed: int = 0) -> dict:
    """Полное парное сравнение двух версий по одной метрике (все три теста).

    Возвращает сводку: средние, Δ, относительный прирост и p-value каждого теста
    плюс флаг significant (по bootstrap-критерию, как наиболее принятому в IR).
    """
    a, b, qids = align_per_query(pq_a, pq_b, metric)
    boot = paired_bootstrap(a, b, n_boot=n_boot, alpha=alpha, seed=seed)
    tt = paired_ttest(a, b)
    wil = wilcoxon_test(a, b)
    mean_a, mean_b = float(np.mean(a)) if a.size else float("nan"), \
                     float(np.mean(b)) if b.size else float("nan")
    rel = (boot["delta"] / mean_a * 100.0) if mean_a not in (0.0, float("nan")) else float("nan")
    return {
        "metric": metric,
        "name_a": name_a, "name_b": name_b,
        "n_queries": len(qids),
        "mean_a": mean_a, "mean_b": mean_b,
        "delta": boot["delta"], "rel_improvement_pct": rel,
        "ci95_low": boot["ci_low"], "ci95_high": boot["ci_high"],
        "p_bootstrap": boot["p_value"],
        "p_ttest": tt["p_value"],
        "p_wilcoxon": wil["p_value"],
        "alpha": alpha,
        "significant": (boot["p_value"] is not None and boot["p_value"] < alpha),
    }


def holm_correction(pvalues: dict, alpha: float = 0.05) -> dict:
    """Поправка Холма–Бонферрони на множественные сравнения.

    pvalues — {имя_сравнения: p}. Возвращает {имя: {p, p_adj, reject}} —
    скорректированные пороги при нескольких версиях против базовой.
    """
    items = [(k, v) for k, v in pvalues.items() if v is not None]
    m = len(items)
    items.sort(key=lambda kv: kv[1])
    out, prev = {}, 0.0
    for rank, (key, p) in enumerate(items):
        p_adj = min(1.0, (m - rank) * p)
        p_adj = max(p_adj, prev)  # монотонность скорректированных p
        prev = p_adj
        out[key] = {"p": p, "p_adj": p_adj, "reject": p_adj < alpha}
    for key, p in pvalues.items():  # тесты без p (scipy off) — отметим отдельно
        if p is None:
            out[key] = {"p": None, "p_adj": None, "reject": None}
    return out
