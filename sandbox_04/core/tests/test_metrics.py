import math

import numpy as np
import pytest

from core import metrics


def _rows_dict(rows):
    return {(qid, m): v for qid, m, v in rows}


def test_hand_reference_threshold_2():
    pids = ["d1", "d2", "d3", "d4", "d5"]
    ranks = np.array([[1, 3, 0, 2, 4]])
    qrels = {"q1": {"d1": 3, "d2": 1, "d3": 0}}
    rows = _rows_dict(metrics.per_query_metrics(
        ranks, ["q1"], pids, qrels, rel_threshold=2,
        recall_ks=(2, 3), mrr_k=10, ndcg_k=3))
    assert rows[("q1", "mrr@10")] == pytest.approx(1 / 3)
    assert rows[("q1", "recall@2")] == 0.0
    assert rows[("q1", "recall@3")] == 1.0
    dcg = 1 / math.log2(2) + 0 / math.log2(3) + 3 / math.log2(4)
    idcg = 3 / math.log2(2) + 1 / math.log2(3)
    assert rows[("q1", "ndcg@3")] == pytest.approx(dcg / idcg, abs=1e-12)


def test_hand_reference_threshold_1():
    pids = ["d1", "d2", "d3", "d4", "d5"]
    ranks = np.array([[1, 3, 0, 2, 4]])
    qrels = {"q1": {"d1": 3, "d2": 1, "d3": 0}}
    rows = _rows_dict(metrics.per_query_metrics(
        ranks, ["q1"], pids, qrels, rel_threshold=1,
        recall_ks=(2, 3), mrr_k=10, ndcg_k=3))
    assert rows[("q1", "mrr@10")] == 1.0
    assert rows[("q1", "recall@2")] == 0.5
    assert rows[("q1", "recall@3")] == 1.0


def test_query_without_qrels_skipped():
    pids = ["d1", "d2"]
    ranks = np.array([[0, 1], [1, 0]])
    qrels = {"q1": {"d1": 1}}
    rows = metrics.per_query_metrics(ranks, ["q1", "q2"], pids, qrels,
                                     rel_threshold=1, recall_ks=(2,))
    assert {q for q, _, _ in rows} == {"q1"}


def test_recall_monotonic_random():
    rng = np.random.default_rng(7)
    n_docs = 200
    pids = [f"d{i}" for i in range(n_docs)]
    qids = [f"q{i}" for i in range(30)]
    ranks = np.stack([rng.permutation(n_docs) for _ in qids])
    qrels = {q: {pids[j]: 1 for j in rng.choice(n_docs, 5, replace=False)}
             for q in qids}
    rows = _rows_dict(metrics.per_query_metrics(
        ranks, qids, pids, qrels, rel_threshold=1, recall_ks=(10, 50, 100)))
    for q in qids:
        assert rows[(q, "recall@10")] <= rows[(q, "recall@50")] <= rows[(q, "recall@100")]


def test_cross_check_ir_measures():
    ir_measures = pytest.importorskip("ir_measures")
    rng = np.random.default_rng(42)
    n_q, n_docs = 100, 50
    pids = [f"d{i}" for i in range(n_docs)]
    qids = [f"q{i}" for i in range(n_q)]
    qrels = {}
    for q in qids:
        judged = rng.choice(n_docs, size=10, replace=False)
        rels = rng.integers(0, 4, size=10)
        if not (rels > 0).any():
            rels[0] = 1
        qrels[q] = {pids[j]: int(r) for j, r in zip(judged, rels)}
    ranks = np.stack([rng.permutation(n_docs) for _ in qids])
    ours = _rows_dict(metrics.per_query_metrics(
        ranks, qids, pids, qrels, rel_threshold=1,
        recall_ks=(5, 20), mrr_k=n_docs, ndcg_k=10))

    qrels_graded = [ir_measures.Qrel(q, p, r)
                    for q, m in qrels.items() for p, r in m.items()]
    qrels_binary = [ir_measures.Qrel(q, p, int(r >= 1))
                    for q, m in qrels.items() for p, r in m.items()]
    run = [ir_measures.ScoredDoc(qids[i], pids[j], float(n_docs - rank))
           for i in range(n_q) for rank, j in enumerate(ranks[i])]

    checks = [("RR", f"mrr@{n_docs}", qrels_binary),
              ("nDCG@10", "ndcg@10", qrels_graded),
              ("R@5", "recall@5", qrels_binary),
              ("R@20", "recall@20", qrels_binary)]
    for measure_str, our_key, qr in checks:
        measure = ir_measures.parse_measure(measure_str)
        n_checked = 0
        for m in ir_measures.iter_calc([measure], qr, run):
            assert abs(ours[(m.query_id, our_key)] - m.value) < 1e-9, \
                f"{measure_str} расходится на {m.query_id}"
            n_checked += 1
        assert n_checked == n_q
