"""SPLADE-Distil (SPLADE v2, arXiv:2109.10086, раздел про дистилляцию).

Студент — тот же SPLADE-max (model.py, query_encoder="mlm"). Учитель —
кросс-энкодер cross-encoder/ms-marco-MiniLM-L6-v2 (одна релевантность на пару
query-doc). Дистилляция по MarginMSE: студент учится повторять разрыв оценок
учителя между позитивом и негативом каждого триплета, плюс FLOPS-регуляризация.

Конвейер на одном GPU (без двух моделей в памяти одновременно):
  этап 1 — учитель считает margins по всем триплетам и кешируется на диск,
           затем выгружается из памяти;
  этап 2 — студент обучается на готовых margins;
  этап 3 — оценка retrieval (evaluate.py, без изменений).
"""
import gc
import json
import resource
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from . import artifacts, data, evaluate
from .config import validate_config
from .model import Splade, flops_loss, tok
from .progress import tqdm
from .train import pick_device, set_seed

DISTIL_KEYS = ("teacher", "max_len", "batch_size")


def validate_distil(cfg: dict):
    if "distil" not in cfg:
        raise KeyError("В конфиге нет секции 'distil' (teacher/max_len/batch_size)")
    missing = [k for k in DISTIL_KEYS if k not in cfg["distil"]]
    if missing:
        raise KeyError(f"В cfg['distil'] не хватает ключей: {missing}")


def _mem(device) -> str:
    parts = []
    if device.type == "cuda":
        parts.append(f"gpu={torch.cuda.memory_allocated() / 1e9:.1f}GB")
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_gb = kb / 1e6 if sys.platform.startswith("linux") else kb / 1e9
    parts.append(f"ram={rss_gb:.1f}GB")
    return " ".join(parts)


def _free(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def teacher_cache_path(ds_dir, teacher: str) -> Path:
    return Path(ds_dir) / f"teacher_margins_{teacher.replace('/', '__')}.npz"


@torch.no_grad()
def compute_teacher_margins(ds_dir, cfg: dict, device) -> np.ndarray:
    dcfg = cfg["distil"]
    triples = data.load_triples(ds_dir)
    if not triples:
        raise RuntimeError(f"Пустой {ds_dir}/triples.tsv")
    path = teacher_cache_path(ds_dir, dcfg["teacher"])
    if path.exists():
        cached = np.load(path)["margin"]
        if len(cached) == len(triples):
            print(f"[distil] кеш учителя: {path} ({len(cached)} margins)")
            return cached.astype(np.float32)
        print(f"[distil] кеш устарел ({len(cached)} != {len(triples)}), пересчёт")

    tokenizer = AutoTokenizer.from_pretrained(dcfg["teacher"])
    teacher = AutoModelForSequenceClassification.from_pretrained(dcfg["teacher"]).to(device).eval()
    use_amp = device.type == "cuda"
    bs, max_len = dcfg["batch_size"], dcfg["max_len"]
    queries = [t[0] for t in triples]

    def score(docs, desc) -> np.ndarray:
        out = np.empty(len(triples), dtype=np.float32)
        pbar = tqdm(range(0, len(triples), bs), desc=desc)
        for s in pbar:
            enc = tokenizer(queries[s:s + bs], docs[s:s + bs], padding=True,
                            truncation=True, max_length=max_len, return_tensors="pt").to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = teacher(**enc).logits[:, 0]
            out[s:s + bs] = logits.float().cpu().numpy()
            pbar.set_postfix(mem=_mem(device))
        return out

    s_pos = score([t[1] for t in triples], "teacher:pos")
    s_neg = score([t[2] for t in triples], "teacher:neg")
    margins = (s_pos - s_neg).astype(np.float32)
    np.savez(path, margin=margins)
    print(f"[distil] учитель посчитан и закеширован: {path} | {_mem(device)}")

    del teacher, tokenizer
    _free(device)
    return margins


def train_loop_distil(model, tokenizer, ds_dir, cfg, device, margins) -> list:
    tcfg, mcfg = cfg["train"], cfg["model"]
    triples = data.load_triples(ds_dir)
    margins_t = torch.from_numpy(np.asarray(margins, dtype=np.float32))

    rng = np.random.default_rng(tcfg["seed"])
    order = rng.permutation(len(triples))
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"])
    max_steps, warmup = tcfg["max_steps"], tcfg["warmup_steps"]

    def lr_lambda(step):  # линейный разогрев + линейное затухание (как в статье)
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return max(0.0, (max_steps - step) / max(1, max_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    bs = tcfg["batch_size"]
    use_amp = device.type == "cuda"
    model.train()
    losses, ptr = [], 0

    pbar = tqdm(range(max_steps), desc=f"distil:{cfg['version']}", unit=" шаг")
    for step in pbar:
        if ptr + bs > len(order):
            order = rng.permutation(len(triples))
            ptr = 0
        idx = order[ptr:ptr + bs]
        ptr += bs
        batch = [triples[i] for i in idx]
        queries = [t[0] for t in batch]
        t_margin = margins_t[torch.from_numpy(idx)].to(device)

        q_enc = tok(tokenizer, queries, mcfg["max_len_query"], device, special=True)
        p_enc = tok(tokenizer, [t[1] for t in batch], mcfg["max_len_doc"], device)
        n_enc = tok(tokenizer, [t[2] for t in batch], mcfg["max_len_doc"], device)

        # квадратичный разгон лямбд до flops_warmup_steps (SPLADE v2)
        lam_scale = min(1.0, ((step + 1) / max(1, tcfg["flops_warmup_steps"])) ** 2)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            q = model.encode_queries(q_enc["input_ids"], q_enc["attention_mask"],
                                     q_enc.get("special_tokens_mask"))
            dp = model.encode_docs(p_enc["input_ids"], p_enc["attention_mask"])
            dn = model.encode_docs(n_enc["input_ids"], n_enc["attention_mask"])
            s_margin = (q * dp).sum(dim=1) - (q * dn).sum(dim=1)  # разрыв оценок студента
            loss = F.mse_loss(s_margin.float(), t_margin)         # повторяем margins учителя
            if model.query_encoder == "mlm" and tcfg["lambda_q"] > 0:
                loss = loss + lam_scale * tcfg["lambda_q"] * flops_loss(q)
            if tcfg["lambda_d"] > 0:
                loss = loss + lam_scale * tcfg["lambda_d"] * flops_loss(torch.cat([dp, dn]))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        if step == 0 or (step + 1) % tcfg["log_every"] == 0:
            window = losses[-tcfg["log_every"]:]
            pbar.set_postfix(loss=f"{np.mean(window):.4f}",
                             lr=f"{scheduler.get_last_lr()[0]:.2e}", mem=_mem(device))
    return losses


def run_experiment_distil(cfg: dict, data_cfg: dict, run_id: str = None):
    validate_config(cfg)
    validate_distil(cfg)
    version, mode = cfg["version"], cfg["mode"]
    ds_dir = data.dataset_dir(data_cfg, mode)
    if not data.is_prepared(data_cfg, mode):
        raise RuntimeError(f"Данные не готовы: {ds_dir}. "
                           f"Сначала запустите клетку подготовки данных (mode={mode}).")

    set_seed(cfg["train"]["seed"])
    device = pick_device()
    run_dir = artifacts.create_run_dir(version, run_id)
    artifacts.save_config(run_dir, cfg)
    meta = artifacts.start_meta(run_dir, cfg, device)
    print(f"[distil] {version} mode={mode} device={device} -> {run_dir}")

    print(f"[distil] этап 1/3: учитель {cfg['distil']['teacher']} (только учитель на GPU)")
    margins = compute_teacher_margins(ds_dir, cfg, device)

    print("[distil] этап 2/3: обучение студента (только студент на GPU)")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["hf_model"])
    model = Splade(cfg["model"]["hf_model"], cfg["model"]["query_encoder"]).to(device)
    losses = train_loop_distil(model, tokenizer, ds_dir, cfg, device, margins)
    (run_dir / "losses.json").write_text(json.dumps(losses), encoding="utf-8")

    model_dir = run_dir / "model"
    model.mlm.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    print("[distil] этап 3/3: оценка retrieval")
    metrics = evaluate.evaluate_retrieval(model, tokenizer, ds_dir, cfg, device)
    metrics["final_train_loss"] = float(np.mean(losses[-10:]))
    metrics["train_steps"] = cfg["train"]["max_steps"]
    metrics["teacher"] = cfg["distil"]["teacher"]
    metrics["peak_mem"] = _mem(device)
    artifacts.save_metrics(run_dir, metrics)
    artifacts.finish_meta(run_dir, meta)
    print(f"[distil] готово: {run_dir}")
    print(f"[distil] метрики: {metrics}")
    return run_dir, metrics
