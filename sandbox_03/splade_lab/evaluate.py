"""Кодирование в разреженные вектора (CSR), полнотекстовый поиск, метрики IR."""
import array

import numpy as np
import scipy.sparse as sp
import torch

from . import data
from .model import tok
from .progress import tqdm  # файловый прогресс вместо tqdm (пишет в лог, не в блокнот)


@torch.no_grad()
def encode_sparse(model, tokenizer, texts, max_len, batch_size, device, kind="doc") -> sp.csr_matrix:
    """-> csr_matrix (len(texts), vocab_size).

    Ненулевые элементы копятся в непрерывные буферы array.array, а не в списки
    мелких ndarray: нет пооверхеда на строку (~100 Б × nnz_строк) и нет удвоения
    памяти на финальном np.concatenate — итоговые массивы CSR строятся zero-copy
    через np.frombuffer. Пик RAM ≈ размер самой матрицы.
    """
    model.eval()
    vals = array.array("f")          # данные CSR, float32
    cols = array.array("i")          # индексы столбцов, int32
    indptr = array.array("q", [0])   # указатели строк, int64
    use_amp = device.type == "cuda"
    for start in tqdm(range(0, len(texts), batch_size), desc=f"encode:{kind}"):
        chunk = texts[start:start + batch_size]
        enc = tok(tokenizer, chunk, max_len, device, special=(kind == "query"))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            if kind == "query":
                reps = model.encode_queries(enc["input_ids"], enc["attention_mask"],
                                            enc.get("special_tokens_mask"))
            else:
                reps = model.encode_docs(enc["input_ids"], enc["attention_mask"])
        reps = reps.float().cpu()
        nz = reps.nonzero(as_tuple=False)  # (k,2): row,col в построчном порядке (= порядок CSR)
        rows = nz[:, 0]
        cols.frombytes(nz[:, 1].to(torch.int32).numpy().tobytes())
        vals.frombytes(reps[rows, nz[:, 1]].numpy().astype(np.float32, copy=False).tobytes())
        counts = torch.bincount(rows, minlength=reps.size(0))  # nnz по каждой строке батча
        cum = (torch.cumsum(counts, 0) + indptr[-1]).numpy().astype(np.int64)
        indptr.frombytes(cum.tobytes())
    return sp.csr_matrix(
        (np.frombuffer(vals, dtype=np.float32),
         np.frombuffer(cols, dtype=np.int32),
         np.frombuffer(indptr, dtype=np.int64)),
        shape=(len(texts), model.vocab_size))


def search(q_mat, d_mat, topk, device, batch_size=64) -> np.ndarray:
    """-> (n_queries, topk) позиции документов в корпусе, по убыванию score."""
    topk = min(topk, d_mat.shape[0])
    if device.type == "cuda":
        return _search_torch_cuda(q_mat, d_mat, topk, device, batch_size)
    return _search_scipy(q_mat, d_mat, topk, batch_size)


def _search_scipy(q_mat, d_mat, topk, batch_size):
    dT = d_mat.T.tocsc()
    out = np.zeros((q_mat.shape[0], topk), dtype=np.int64)
    for s in tqdm(range(0, q_mat.shape[0], batch_size), desc="search:cpu"):
        scores = (q_mat[s:s + batch_size] @ dT).toarray()
        idx = np.argpartition(-scores, topk - 1, axis=1)[:, :topk]
        part = np.take_along_axis(scores, idx, axis=1)
        order = np.argsort(-part, axis=1)
        out[s:s + batch_size] = np.take_along_axis(idx, order, axis=1)
    return out


def _search_torch_cuda(q_mat, d_mat, topk, device, batch_size):
    d = d_mat.tocsr()
    D = torch.sparse_csr_tensor(
        torch.from_numpy(d.indptr.astype(np.int64)),
        torch.from_numpy(d.indices.astype(np.int64)),
        torch.from_numpy(d.data),
        size=d.shape, device=device)
    out = np.zeros((q_mat.shape[0], topk), dtype=np.int64)
    for s in tqdm(range(0, q_mat.shape[0], batch_size), desc="search:gpu"):
        qb = torch.from_numpy(q_mat[s:s + batch_size].toarray()).to(device)  # b,V
        scores = torch.matmul(D, qb.t())                                     # N,b
        top = torch.topk(scores.t(), k=topk, dim=1).indices                  # b,topk
        out[s:s + batch_size] = top.cpu().numpy()
    return out


def compute_metrics(ranks, qids, pids, qrels, ks=(10, 100, 1000)) -> dict:
    ks = [k for k in ks if k <= ranks.shape[1]] or [ranks.shape[1]]
    mrr, recalls = [], {k: [] for k in ks}
    for i, qid in enumerate(qids):
        rel = qrels.get(qid, set())
        if not rel:
            continue
        ranked_pids = [pids[j] for j in ranks[i]]
        rr = 0.0
        for rank, pid in enumerate(ranked_pids[:10], start=1):
            if pid in rel:
                rr = 1.0 / rank
                break
        mrr.append(rr)
        for k in ks:
            recalls[k].append(len(set(ranked_pids[:k]) & rel) / len(rel))
    out = {"mrr@10": float(np.mean(mrr))}
    for k in ks:
        out[f"recall@{k}"] = float(np.mean(recalls[k]))
    out["n_eval_queries"] = len(mrr)
    return out


def evaluate_retrieval(model, tokenizer, ds_dir, cfg, device) -> dict:
    ecfg, mcfg = cfg["eval"], cfg["model"]
    pids, doc_texts = data.load_collection(ds_dir)
    qids, q_texts = data.load_queries(ds_dir)
    qrels = data.load_qrels(ds_dir)

    d_mat = encode_sparse(model, tokenizer, doc_texts, mcfg["max_len_doc"],
                          ecfg["batch_size_docs"], device, kind="doc")
    q_mat = encode_sparse(model, tokenizer, q_texts, mcfg["max_len_query"],
                          ecfg["batch_size_queries"], device, kind="query")

    ks = tuple(ecfg["recall_ks"])
    ranks = search(q_mat, d_mat, max(ks), device, ecfg["batch_size_search"])
    metrics = compute_metrics(ranks, qids, pids, qrels, ks)
    metrics["n_corpus_docs"] = len(pids)
    metrics["avg_nnz_doc"] = float(d_mat.getnnz(axis=1).mean())
    metrics["avg_nnz_query"] = float(q_mat.getnnz(axis=1).mean())
    return metrics
