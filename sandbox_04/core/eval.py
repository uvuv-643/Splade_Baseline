import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from . import data, guards, metrics

ENCODE_CHUNK = 32768


def _encode_matrix(encode_fn, texts) -> sp.csr_matrix:
    mats = [encode_fn(texts[s:s + ENCODE_CHUNK])
            for s in range(0, len(texts), ENCODE_CHUNK)]
    return sp.vstack(mats, format="csr") if len(mats) > 1 else mats[0]


def search(q_mat, d_mat, topk, device, batch_size=64) -> np.ndarray:
    topk = min(topk, d_mat.shape[0])
    if device.type == "cuda":
        return _search_cuda(q_mat, d_mat, topk, device, batch_size)
    return _search_cpu(q_mat, d_mat, topk, batch_size)


def _search_cpu(q_mat, d_mat, topk, batch_size):
    dT = d_mat.T.tocsc()
    out = np.zeros((q_mat.shape[0], topk), dtype=np.int64)
    for s in range(0, q_mat.shape[0], batch_size):
        scores = (q_mat[s:s + batch_size] @ dT).toarray()
        idx = np.argpartition(-scores, topk - 1, axis=1)[:, :topk]
        part = np.take_along_axis(scores, idx, axis=1)
        order = np.argsort(-part, axis=1)
        out[s:s + batch_size] = np.take_along_axis(idx, order, axis=1)
    return out


def _search_cuda(q_mat, d_mat, topk, device, batch_size):
    d = d_mat.tocsr()
    D = torch.sparse_csr_tensor(
        torch.from_numpy(d.indptr.astype(np.int64)),
        torch.from_numpy(d.indices.astype(np.int64)),
        torch.from_numpy(d.data),
        size=d.shape, device=device)
    out = np.zeros((q_mat.shape[0], topk), dtype=np.int64)
    for s in range(0, q_mat.shape[0], batch_size):
        qb = torch.from_numpy(q_mat[s:s + batch_size].toarray()).to(device)
        scores = torch.matmul(D, qb.t())
        top = torch.topk(scores.t(), k=topk, dim=1).indices
        out[s:s + batch_size] = top.cpu().numpy()
    return out


def expected_counts(dataset_names, max_queries, corpus_sizes: dict) -> dict:
    expected = {}
    for name in dataset_names:
        spec = data.DATASETS[name]
        qids, _ = data.load_queries(spec["queryset"], max_queries)
        qrels = data.load_qrels(spec["queryset"])
        thr = spec["rel_threshold"]
        n_binary = sum(1 for q in qids
                       if any(r >= thr for r in qrels.get(q, {}).values()))
        n_graded = sum(1 for q in qids
                       if any(r > 0 for r in qrels.get(q, {}).values()))
        counts = {f"mrr@{metrics.MRR_K}": n_binary,
                  f"ndcg@{metrics.NDCG_K}": n_graded}
        for k in metrics.RECALL_KS:
            if k <= corpus_sizes[spec["corpus"]]:
                counts[f"recall@{k}"] = n_binary
        expected[name] = counts
    return expected


def run_eval(encoder, dataset_names, eval_cfg, device, run_dir=None):
    groups = {}
    for name in dataset_names:
        groups.setdefault(data.DATASETS[name]["corpus"], []).append(name)

    rows = []
    result = {"datasets": {}, "corpora": {}}
    corpus_sizes = {}
    for corpus_name, names in sorted(groups.items()):
        pids, texts = data.load_corpus(corpus_name)
        corpus_sizes[corpus_name] = len(pids)
        index_file = (Path(run_dir) / "index" / f"{corpus_name}.npz") if run_dir else None
        t0 = time.time()
        if index_file is not None and index_file.exists():
            print(f"[eval] индекс {corpus_name} из {index_file}")
            d_mat = sp.load_npz(index_file)
        else:
            print(f"[eval] энкод корпуса {corpus_name}: {len(texts)} пассажей")
            d_mat = _encode_matrix(encoder.encode_docs, texts)
        encode_s = time.time() - t0
        guards.check_csr(d_mat, len(texts), f"doc:{corpus_name}")
        guards.check_index_size(d_mat.shape[0], len(pids))
        if eval_cfg["save_index"] and index_file is not None and not index_file.exists():
            index_file.parent.mkdir(parents=True, exist_ok=True)
            sp.save_npz(index_file, d_mat)
            (index_file.with_suffix(".pids.txt")).write_text(
                "\n".join(pids), encoding="utf-8")
        result["corpora"][corpus_name] = {
            "n_docs": len(pids),
            "avg_nnz_doc": float(d_mat.getnnz(axis=1).mean()),
            "encode_s": round(encode_s, 1),
        }
        for ds_name in names:
            spec = data.DATASETS[ds_name]
            qids, q_texts = data.load_queries(spec["queryset"], eval_cfg["max_queries"])
            qrels = data.load_qrels(spec["queryset"])
            q_mat = _encode_matrix(encoder.encode_queries, q_texts)
            guards.check_csr(q_mat, len(qids), f"query:{ds_name}")
            t0 = time.time()
            ranks = search(q_mat, d_mat, eval_cfg["topk"], device,
                           eval_cfg["batch_size_search"])
            search_s = time.time() - t0
            ds_rows = metrics.per_query_metrics(ranks, qids, pids, qrels,
                                                spec["rel_threshold"])
            rows.extend((ds_name, qid, metric, value)
                        for qid, metric, value in ds_rows)
            agg = metrics.aggregate(ds_rows)
            agg.update({
                "n_queries": len({q for q, _, _ in ds_rows}),
                "avg_nnz_query": float(q_mat.getnnz(axis=1).mean()),
                "search_s": round(search_s, 1),
                "eval_data_hash": data.eval_data_hash([ds_name]),
            })
            result["datasets"][ds_name] = agg
            print(f"[eval] {ds_name}: " + " ".join(
                f"{k}={v:.4f}" for k, v in sorted(agg.items())
                if isinstance(v, float) and "@" in k))
    df = pd.DataFrame(rows, columns=["dataset", "qid", "metric", "value"])
    guards.check_per_query(df, expected_counts(dataset_names,
                                               eval_cfg["max_queries"], corpus_sizes))
    guards.check_recall_monotonic(df)
    return result, df
