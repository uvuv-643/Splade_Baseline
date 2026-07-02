import itertools

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from core import stats


def test_fisher_exact_matches_enumeration():
    d = np.array([0.1, -0.2, 0.3])
    observed = abs(d.mean())
    count = sum(1 for signs in itertools.product([-1, 1], repeat=3)
                if abs((d * signs).mean()) >= observed - 1e-12)
    expected = count / 8
    assert stats.fisher_randomization(d) == pytest.approx(expected)


def test_fisher_detects_shift():
    rng = np.random.default_rng(0)
    d = rng.normal(0.5, 0.1, size=100)
    assert stats.fisher_randomization(d) < 0.001


def test_fisher_null_uniformish():
    rng = np.random.default_rng(1)
    d = rng.normal(0.0, 1.0, size=100)
    p = stats.fisher_randomization(d)
    assert 0.01 < p <= 1.0


def test_ttest_matches_scipy():
    rng = np.random.default_rng(2)
    d = rng.normal(0.1, 0.5, size=30)
    assert stats.paired_ttest(d) == pytest.approx(
        float(sps.ttest_1samp(d, 0.0).pvalue))


def test_holm_textbook():
    out = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03}, alpha=0.05)
    assert out["a"]["p_adj"] == pytest.approx(0.03)
    assert out["c"]["p_adj"] == pytest.approx(0.06)
    assert out["b"]["p_adj"] == pytest.approx(0.06)
    assert out["a"]["reject"] and not out["b"]["reject"] and not out["c"]["reject"]


def test_bca_matches_scipy():
    rng = np.random.default_rng(0)
    d = rng.normal(0.05, 0.1, size=40)
    theta, lo, hi = stats.bca_interval(d, n_boot=20000, seed=1)
    res = sps.bootstrap((d,), np.mean, confidence_level=0.95,
                        n_resamples=20000, method="BCa",
                        random_state=np.random.default_rng(2))
    assert theta == pytest.approx(d.mean())
    assert lo == pytest.approx(res.confidence_interval.low, abs=0.01)
    assert hi == pytest.approx(res.confidence_interval.high, abs=0.01)
    assert lo < theta < hi


def test_bca_degenerate():
    theta, lo, hi = stats.bca_interval(np.full(10, 0.25))
    assert theta == lo == hi == 0.25


def test_scaling_fit_recovers_params():
    n = np.array([30000, 120000, 480000, 1920000], dtype=float)
    n = np.repeat(n, 3)
    rng = np.random.default_rng(3)
    y = 0.40 - 0.5 * n ** (-0.30) + rng.normal(0, 1e-6, size=n.size)
    fit = stats.fit_scaling(n, y)
    assert fit is not None
    assert fit["a"] == pytest.approx(0.40, abs=1e-3)
    assert fit["gamma"] == pytest.approx(0.30, abs=0.01)
    assert fit["r2"] > 0.999


def test_scaling_fit_noisy_still_describes():
    # при реалистичном шуме сидов γ слабо идентифицируем (плоская SSE-поверхность),
    # поэтому проверяется только качество описания, а не точечное значение γ
    n = np.repeat(np.array([30000, 120000, 480000, 1920000], dtype=float), 3)
    rng = np.random.default_rng(3)
    y = 0.40 - 0.5 * n ** (-0.30) + rng.normal(0, 5e-4, size=n.size)
    fit = stats.fit_scaling(n, y)
    assert fit is not None
    assert fit["r2"] > 0.95
    assert 0.35 <= fit["a"] <= 0.45


def test_kendall_trend_monotone():
    tr = stats.kendall_trend([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    assert tr["tau"] == pytest.approx(1.0)


def _pq(dataset, values):
    return pd.DataFrame([{"dataset": dataset, "qid": f"q{i}",
                          "metric": "mrr@10", "value": v}
                         for i, v in enumerate(values)])


def test_seed_average():
    a = _pq("ds", [0.2, 0.4])
    b = _pq("ds", [0.4, 0.6])
    avg = stats.seed_average([a, b])
    assert sorted(avg["value"]) == pytest.approx([0.3, 0.5])


def test_seed_average_mismatch_raises():
    a = _pq("ds", [0.2, 0.4])
    b = _pq("ds", [0.4, 0.6, 0.8])
    with pytest.raises(ValueError):
        stats.seed_average([a, b])


def test_varying_numeric_param():
    base = {"name": "x", "data": {"train_triples": 100, "train_pool": "p"},
            "train": {"lr": 1e-5}}
    other = {"name": "y", "data": {"train_triples": 200, "train_pool": "p"},
             "train": {"lr": 1e-5}}
    found = stats.varying_numeric_param({"x": base, "y": other})
    assert found is not None
    path, values = found
    assert path == "data.train_triples"
    assert values == {"x": 100, "y": 200}


def test_varying_param_none_when_multiple():
    a = {"name": "x", "data": {"train_triples": 100}, "train": {"lr": 1e-5}}
    b = {"name": "y", "data": {"train_triples": 200}, "train": {"lr": 2e-5}}
    assert stats.varying_numeric_param({"x": a, "y": b}) is None
