import numpy as np
import scipy.sparse as sp

from .guards import GuardError

PROBE_TEXTS = ["what is the capital of france", "splade sparse retrieval model"]


def _check_matrix(m, method: str):
    if not sp.issparse(m) or m.format != "csr":
        raise GuardError(f"{method}: ожидается scipy.sparse.csr_matrix, получено {type(m)}")
    if m.shape[0] != len(PROBE_TEXTS):
        raise GuardError(f"{method}: {m.shape[0]} строк на {len(PROBE_TEXTS)} текстов")
    if m.nnz == 0:
        raise GuardError(f"{method}: пустой выход (nnz=0)")
    if not np.isfinite(m.data).all():
        raise GuardError(f"{method}: NaN/inf в значениях")
    if (m.data < 0).any():
        raise GuardError(f"{method}: отрицательные веса")


def check_encoder_contract(encoder):
    dims = {}
    for method in ("encode_queries", "encode_docs"):
        fn = getattr(encoder, method, None)
        if fn is None:
            raise GuardError(f"у энкодера нет метода {method}")
        m1 = fn(list(PROBE_TEXTS))
        m2 = fn(list(PROBE_TEXTS))
        _check_matrix(m1, method)
        _check_matrix(m2, method)
        if (m1 != m2).nnz != 0:
            raise GuardError(f"{method}: недетерминизм в eval-режиме")
        dims[method] = m1.shape[1]
    if dims["encode_queries"] != dims["encode_docs"]:
        raise GuardError(f"размерности словаря query/doc не совпадают: {dims}")
