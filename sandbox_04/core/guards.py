import numpy as np


class GuardError(RuntimeError):
    pass


def check_finite(name: str, value: float):
    if not np.isfinite(value):
        raise GuardError(f"{name} не конечен: {value}")


def check_csr(mat, n_rows: int, kind: str):
    if mat.shape[0] != n_rows:
        raise GuardError(f"{kind}: {mat.shape[0]} строк, ожидалось {n_rows}")
    if mat.nnz == 0:
        raise GuardError(f"{kind}: пустая матрица (nnz=0)")
    if not np.isfinite(mat.data).all():
        raise GuardError(f"{kind}: NaN/inf в значениях")
    if (mat.data < 0).any():
        raise GuardError(f"{kind}: отрицательные веса")
    zero_rows = int((mat.getnnz(axis=1) == 0).sum())
    if zero_rows:
        raise GuardError(f"{kind}: {zero_rows} нулевых векторов")


def check_index_size(n_indexed: int, n_corpus: int):
    if n_indexed != n_corpus:
        raise GuardError(f"в индексе {n_indexed} документов, в корпусе {n_corpus}")


def check_per_query(df, expected: dict):
    if len(df) == 0:
        raise GuardError("per_query пуст")
    bad = df[(df["value"] < -1e-12) | (df["value"] > 1 + 1e-12)]
    if len(bad):
        raise GuardError(f"метрики вне [0,1]: {len(bad)} строк, "
                         f"например {bad.iloc[0].to_dict()}")
    for (ds, metric), grp in df.groupby(["dataset", "metric"]):
        want = expected.get(ds, {}).get(metric)
        if want is not None and len(grp) != want:
            raise GuardError(f"{ds}/{metric}: {len(grp)} строк per_query, "
                             f"ожидалось {want}")


def check_recall_monotonic(df):
    rec = df[df["metric"].str.startswith("recall@")]
    if len(rec) == 0:
        return
    piv = rec.pivot_table(index=["dataset", "qid"], columns="metric", values="value")
    ks = sorted(int(c.split("@")[1]) for c in piv.columns)
    for lo, hi in zip(ks, ks[1:]):
        a, b = piv[f"recall@{lo}"], piv[f"recall@{hi}"]
        mask = a.notna() & b.notna()
        if (a[mask] > b[mask] + 1e-12).any():
            raise GuardError(f"recall@{lo} > recall@{hi} у части запросов")
