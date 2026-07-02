import numpy as np
import pytest
import scipy.sparse as sp

from core.contract import check_encoder_contract
from core.guards import GuardError

VOCAB = 100


def _det_csr(texts, value_fn):
    rows, cols, vals = [], [], []
    for i, text in enumerate(texts):
        col = sum(ord(c) for c in text) % VOCAB
        rows.append(i)
        cols.append(col)
        vals.append(value_fn(text))
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(texts), VOCAB))


class GoodEncoder:
    def encode_queries(self, texts):
        return _det_csr(texts, lambda t: float(len(t)))

    def encode_docs(self, texts):
        return _det_csr(texts, lambda t: float(len(t)) + 1.0)


class DenseEncoder(GoodEncoder):
    def encode_docs(self, texts):
        return np.ones((len(texts), VOCAB))


class NegativeEncoder(GoodEncoder):
    def encode_docs(self, texts):
        return _det_csr(texts, lambda t: -1.0)


class RandomEncoder(GoodEncoder):
    def __init__(self):
        self.rng = np.random.default_rng()

    def encode_docs(self, texts):
        return _det_csr(texts, lambda t: float(self.rng.random()) + 0.1)


class NanEncoder(GoodEncoder):
    def encode_docs(self, texts):
        return _det_csr(texts, lambda t: float("nan"))


def test_good_encoder_passes():
    check_encoder_contract(GoodEncoder())


def test_dense_output_rejected():
    with pytest.raises(GuardError):
        check_encoder_contract(DenseEncoder())


def test_negative_values_rejected():
    with pytest.raises(GuardError):
        check_encoder_contract(NegativeEncoder())


def test_nondeterminism_rejected():
    with pytest.raises(GuardError):
        check_encoder_contract(RandomEncoder())


def test_nan_rejected():
    with pytest.raises(GuardError):
        check_encoder_contract(NanEncoder())


def test_missing_method_rejected():
    class NoDocs:
        def encode_queries(self, texts):
            return _det_csr(texts, lambda t: 1.0)

    with pytest.raises(GuardError):
        check_encoder_contract(NoDocs())
