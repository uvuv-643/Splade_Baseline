import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from core import guards


def _csr(dense):
    return sp.csr_matrix(np.asarray(dense, dtype=np.float64))


def test_check_finite():
    guards.check_finite("loss", 1.0)
    with pytest.raises(guards.GuardError):
        guards.check_finite("loss", float("nan"))
    with pytest.raises(guards.GuardError):
        guards.check_finite("loss", float("inf"))


def test_check_csr_ok():
    guards.check_csr(_csr([[0, 1.0], [2.0, 0]]), 2, "doc")


def test_check_csr_zero_row():
    with pytest.raises(guards.GuardError, match="нулевых"):
        guards.check_csr(_csr([[0, 1.0], [0, 0]]), 2, "doc")


def test_check_csr_negative():
    with pytest.raises(guards.GuardError, match="отрицательные"):
        guards.check_csr(_csr([[0, -1.0], [1.0, 0]]), 2, "doc")


def test_check_csr_wrong_rows():
    with pytest.raises(guards.GuardError):
        guards.check_csr(_csr([[0, 1.0]]), 2, "doc")


def test_check_index_size():
    guards.check_index_size(10, 10)
    with pytest.raises(guards.GuardError):
        guards.check_index_size(9, 10)


def _df(rows):
    return pd.DataFrame(rows, columns=["dataset", "qid", "metric", "value"])


def test_per_query_bounds():
    df = _df([("ds", "q1", "mrr@10", 1.5)])
    with pytest.raises(guards.GuardError, match=r"\[0,1\]"):
        guards.check_per_query(df, {})


def test_per_query_row_count():
    df = _df([("ds", "q1", "mrr@10", 0.5)])
    guards.check_per_query(df, {"ds": {"mrr@10": 1}})
    with pytest.raises(guards.GuardError, match="строк per_query"):
        guards.check_per_query(df, {"ds": {"mrr@10": 2}})


def test_recall_monotonic():
    good = _df([("ds", "q1", "recall@10", 0.2), ("ds", "q1", "recall@100", 0.5)])
    guards.check_recall_monotonic(good)
    bad = _df([("ds", "q1", "recall@10", 0.6), ("ds", "q1", "recall@100", 0.5)])
    with pytest.raises(guards.GuardError):
        guards.check_recall_monotonic(bad)
