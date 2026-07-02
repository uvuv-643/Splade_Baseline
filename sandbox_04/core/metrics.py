import numpy as np

RECALL_KS = (10, 100, 1000)
MRR_K = 10
NDCG_K = 10


def dcg(gains) -> float:
    """DCG с линейными gains: sum_i rel_i / log2(i+1) — как в trec_eval/ir_measures."""
    gains = np.asarray(gains, dtype=np.float64)
    if gains.size == 0:
        return 0.0
    return float((gains / np.log2(np.arange(2, gains.size + 2))).sum())


def per_query_metrics(ranks, qids, pids, qrels, rel_threshold,
                      recall_ks=RECALL_KS, mrr_k=MRR_K, ndcg_k=NDCG_K) -> list:
    """-> [(qid, metric, value)].

    Бинарные метрики (MRR, Recall) считаются по rel >= rel_threshold
    (для TREC-DL стандарт: порог 2); nDCG — по градуированным rel > 0.
    Запрос без релевантных для данного вида метрики пропускается — как в trec_eval.
    """
    rows = []
    ks = [k for k in recall_ks if k <= ranks.shape[1]]
    for i, qid in enumerate(qids):
        rel_map = qrels.get(qid)
        if not rel_map:
            continue
        ranked = [pids[j] for j in ranks[i]]
        binary = {p for p, r in rel_map.items() if r >= rel_threshold}
        graded = {p: r for p, r in rel_map.items() if r > 0}
        if binary:
            rr = 0.0
            for rank, pid in enumerate(ranked[:mrr_k], start=1):
                if pid in binary:
                    rr = 1.0 / rank
                    break
            rows.append((qid, f"mrr@{mrr_k}", rr))
            for k in ks:
                hit = len(set(ranked[:k]) & binary)
                rows.append((qid, f"recall@{k}", hit / len(binary)))
        if graded:
            ideal = sorted(graded.values(), reverse=True)[:ndcg_k]
            gains = [graded.get(pid, 0) for pid in ranked[:ndcg_k]]
            idcg = dcg(ideal)
            rows.append((qid, f"ndcg@{ndcg_k}",
                         dcg(gains) / idcg if idcg > 0 else 0.0))
    return rows


def aggregate(rows: list) -> dict:
    by_metric = {}
    for _, metric, value in rows:
        by_metric.setdefault(metric, []).append(value)
    return {m: float(np.mean(v)) for m, v in sorted(by_metric.items())}
