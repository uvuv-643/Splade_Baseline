"""Бенчмарк сохранённых моделей SPLADE на маленьких наборах (zero-shot).

Что делает
----------
Берёт уже обученную (на MS MARCO) модель из ``outputs/<version>/<run_id>/`` и
прогоняет инференс+поиск по небольшому набору (например SciFact, ~5K докум.):
кодирует корпус и запросы в разреженные вектора, ищет top-k, считает метрики
IR — агрегатно и **по каждому запросу** (нужно для статистики значимости).

Почему быстро
-------------
Узкое место полного MS MARCO — кодирование 8.8M пассажей и построение индекса
(~1.5 ч). У BEIR-наборов корпус на 3-4 порядка меньше, поэтому весь прогон —
секунды/минуты. Логика кодирования и поиска НЕ дублируется: переиспользуются
``evaluate.encode_sparse`` и ``evaluate.search`` из основного пайплайна без
изменений — те же оптимизированные CSR/GPU-пути, что и в обучении на MS MARCO.

Метрики
-------
Главная для BEIR — **nDCG@10** (учитывает градуированную релевантность qrels).
Также считаем Recall@k и MRR@10 — для сопоставимости с таблицами MS MARCO.
Все метрики возвращаются и как среднее, и как вектор по запросам.

Важно про qrels
---------------
``data.load_qrels`` отдаёт только множество релевантных pid (rel>0) — этого
достаточно для recall/MRR, но не для nDCG с градациями. Поэтому здесь qrels
читаются отдельно, с сохранением оценок релевантности (BEIR их даёт).
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import data, datasets, evaluate, modeling
from .config import OUTPUTS_DIR, resolve_path
from .progress import tqdm  # файловый прогресс вместо tqdm (пишет в лог, не в блокнот)
from .train import pick_device


# ---------- qrels с градациями ----------

def load_graded_qrels(ds_dir) -> dict:
    """{qid: {pid: rel}} с сохранением градаций релевантности (rel>0)."""
    qrels = {}
    path = Path(ds_dir) / "qrels.tsv"
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            qid, _, pid, rel = parts
            rel = int(rel)
            if rel > 0:
                qrels.setdefault(qid, {})[pid] = rel
    return qrels


# ---------- метрики по запросам ----------

def _dcg(gains) -> float:
    gains = np.asarray(gains, dtype=np.float64)
    if gains.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    return float((gains * discounts).sum())


def per_query_metrics(ranks, qids, pids, qrels, ks=(10, 100, 1000),
                      ndcg_ks=(10,), mrr_k=10) -> dict:
    """Метрики по каждому запросу.

    ranks — (n_queries, topk) позиции документов в корпусе (из evaluate.search);
    qrels — {qid: {pid: rel}} с градациями.
    Возвращает dict: имя метрики -> np.ndarray длины n_eval (по запросам с qrels)
    плюс 'qids' (порядок запросов) для парных тестов.
    """
    max_avail = ranks.shape[1]
    ks = [k for k in ks if k <= max_avail] or [max_avail]
    ndcg_ks = [k for k in ndcg_ks if k <= max_avail] or [min(10, max_avail)]

    out_qids = []
    mrr_vals = []
    recall_vals = {k: [] for k in ks}
    ndcg_vals = {k: [] for k in ndcg_ks}

    for i, qid in enumerate(qids):
        rel_map = qrels.get(qid)
        if not rel_map:
            continue
        out_qids.append(qid)
        ranked_pids = [pids[j] for j in ranks[i]]
        rel_set = set(rel_map)

        # MRR@mrr_k
        rr = 0.0
        for rank, pid in enumerate(ranked_pids[:mrr_k], start=1):
            if pid in rel_set:
                rr = 1.0 / rank
                break
        mrr_vals.append(rr)

        # Recall@k
        n_rel = len(rel_set)
        for k in ks:
            hit = len(set(ranked_pids[:k]) & rel_set)
            recall_vals[k].append(hit / n_rel if n_rel else 0.0)

        # nDCG@k (градуированный, idealDCG по сортировке оценок qrels)
        ideal_gains = sorted(rel_map.values(), reverse=True)
        for k in ndcg_ks:
            gains = [rel_map.get(pid, 0) for pid in ranked_pids[:k]]
            idcg = _dcg(ideal_gains[:k])
            ndcg_vals[k].append(_dcg(gains) / idcg if idcg > 0 else 0.0)

    res = {"qids": out_qids,
           f"mrr@{mrr_k}": np.asarray(mrr_vals, dtype=np.float64)}
    for k in ks:
        res[f"recall@{k}"] = np.asarray(recall_vals[k], dtype=np.float64)
    for k in ndcg_ks:
        res[f"ndcg@{k}"] = np.asarray(ndcg_vals[k], dtype=np.float64)
    return res


def aggregate(per_query: dict) -> dict:
    """Средние по запросам (то, что обычно показывают в таблицах)."""
    agg = {}
    for key, val in per_query.items():
        if key == "qids":
            continue
        agg[key] = float(np.mean(val)) if len(val) else float("nan")
    agg["n_eval_queries"] = len(per_query.get("qids", []))
    return agg


# ---------- основной прогон ----------

def benchmark_model(model, tokenizer, ds_dir, device, *,
                    max_len_query, max_len_doc,
                    batch_size_docs=256, batch_size_queries=64,
                    batch_size_search=64,
                    recall_ks=(10, 100, 1000), ndcg_ks=(10,)) -> dict:
    """Полный прогон одной модели по набору. Возвращает {aggregate, per_query, timing, sparsity}.

    Кодирование/поиск — через evaluate.* (без изменений основного кода).
    """
    pids, doc_texts = data.load_collection(ds_dir)
    qids, q_texts = data.load_queries(ds_dir)
    qrels = load_graded_qrels(ds_dir)

    t0 = time.time()
    d_mat = evaluate.encode_sparse(model, tokenizer, doc_texts, max_len_doc,
                                   batch_size_docs, device, kind="doc")
    t_index = time.time() - t0

    t0 = time.time()
    q_mat = evaluate.encode_sparse(model, tokenizer, q_texts, max_len_query,
                                   batch_size_queries, device, kind="query")
    topk = max(max(recall_ks), max(ndcg_ks))
    ranks = evaluate.search(q_mat, d_mat, topk, device, batch_size_search)
    t_search = time.time() - t0

    pq = per_query_metrics(ranks, qids, pids, qrels,
                           ks=tuple(recall_ks), ndcg_ks=tuple(ndcg_ks))
    agg = aggregate(pq)
    agg["n_corpus_docs"] = len(pids)
    agg["avg_nnz_doc"] = float(d_mat.getnnz(axis=1).mean())
    agg["avg_nnz_query"] = float(q_mat.getnnz(axis=1).mean())
    return {
        "aggregate": agg,
        "per_query": pq,
        "timing": {"index_s": round(t_index, 2),
                   "search_s": round(t_search, 2)},
    }


# ---------- высокоуровневый прогон (одна модель × один набор) ----------

# Куда складывать результаты бенчмарка (отдельно от outputs/<version>/, чтобы не
# мешать артефактам обучения и не попадать в compare_runs() основного блокнота).
BENCH_OUTPUTS = OUTPUTS_DIR / "benchmark"


def _save_per_query_csv(path: Path, pq: dict):
    """Метрики по запросам в CSV (qid + по столбцу на метрику) — для перепроверки."""
    metric_keys = [k for k in pq if k != "qids"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("qid," + ",".join(metric_keys) + "\n")
        for i, qid in enumerate(pq["qids"]):
            row = [str(qid)] + [f"{float(pq[k][i]):.6f}" for k in metric_keys]
            f.write(",".join(row) + "\n")


def run_benchmark(version=None, data_cfg=None, *, run_id=None, run_dir=None,
                  query_encoder=None, device=None,
                  recall_ks=(10, 100, 1000), ndcg_ks=(10,),
                  batch_overrides=None, save=True, label=None) -> dict:
    """Прогоняет сохранённую модель по набору данных и (опц.) пишет артефакты.

    version/run_id/run_dir — какая модель (см. modeling.resolve_model_run);
    data_cfg — конфиг набора из datasets.beir_config(...).
    Возвращает {label, version, run_id, dataset, aggregate, per_query, timing}.
    Артефакты: outputs/benchmark/<dataset>/<label>__<src_run>/...
    """
    if data_cfg is None:
        raise ValueError("Нужен data_cfg (datasets.beir_config(...))")
    device = device or pick_device()
    src_run = modeling.resolve_model_run(version=version, run_id=run_id,
                                          run_dir=run_dir)
    model, tokenizer, cfg = modeling.load_model(src_run, device, query_encoder)
    ver = cfg.get("version", version or src_run.parent.name)

    ds_dir = datasets.dataset_dir(data_cfg)
    if not datasets.is_prepared(data_cfg):
        raise RuntimeError(f"Набор не готов: {ds_dir}. Сначала "
                           f"datasets.prepare_beir(DATA).")

    bo = batch_overrides or {}
    print(f"[bench] {ver} (src={src_run.name}) на {data_cfg['name']} "
          f"qe={query_encoder or cfg['model']['query_encoder']} device={device}")
    res = benchmark_model(
        model, tokenizer, ds_dir, device,
        max_len_query=cfg["model"]["max_len_query"],
        max_len_doc=cfg["model"]["max_len_doc"],
        batch_size_docs=bo.get("batch_size_docs", 256),
        batch_size_queries=bo.get("batch_size_queries", 64),
        batch_size_search=bo.get("batch_size_search", 64),
        recall_ks=recall_ks, ndcg_ks=ndcg_ks)

    out = {
        "label": label or ver,
        "version": ver,
        "run_id": src_run.name,
        "dataset": data_cfg["name"],
        "split": data_cfg.get("split"),
        "query_encoder": query_encoder or cfg["model"]["query_encoder"],
        "aggregate": res["aggregate"],
        "per_query": res["per_query"],
        "timing": res["timing"],
    }
    print(f"[bench] {data_cfg['name']} | " + " ".join(
        f"{k}={v:.4f}" for k, v in res["aggregate"].items()
        if isinstance(v, float)) + f" | index={res['timing']['index_s']}s "
        f"search={res['timing']['search_s']}s")

    if save:
        bench_dir = (BENCH_OUTPUTS / data_cfg["name"] /
                     f"{out['label']}__{src_run.name}")
        bench_dir.mkdir(parents=True, exist_ok=True)
        summary = {k: v for k, v in out.items() if k != "per_query"}
        summary["created_at"] = datetime.now(timezone.utc).isoformat()
        (bench_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        _save_per_query_csv(bench_dir / "per_query.csv", res["per_query"])
        # per_query в npz — чтобы статистика читала векторы без повторного прогона
        np.savez(bench_dir / "per_query.npz",
                 qids=np.asarray(out["per_query"]["qids"]),
                 **{k: v for k, v in out["per_query"].items() if k != "qids"})
        out["bench_dir"] = str(bench_dir)
        print(f"[bench] артефакты: {bench_dir}")
    return out


def load_per_query(bench_dir) -> dict:
    """Читает сохранённый per_query.npz обратно в dict (для stats.compare_pair)."""
    bench_dir = Path(bench_dir)
    npz = np.load(bench_dir / "per_query.npz", allow_pickle=True)
    out = {"qids": list(npz["qids"])}
    for key in npz.files:
        if key != "qids":
            out[key] = npz[key]
    return out
