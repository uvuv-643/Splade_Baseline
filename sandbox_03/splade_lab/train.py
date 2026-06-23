"""Прогон эксперимента: обучение SPLADE (InfoNCE + FLOPS-рег.) -> eval -> артефакты.

Интерфейс под ноутбук: run_experiment(cfg: dict, data_cfg: dict) — оба конфига
задаются в ноутбуке питоновскими словарями.
"""
import random

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from .progress import tqdm  # файловый прогресс вместо tqdm (пишет в лог, не в блокнот)

from . import artifacts, data, evaluate
from .config import validate_config
from .model import Splade, flops_loss, tok


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # детерминизм важнее скорости
    torch.backends.cudnn.benchmark = False


def train_loop(model, tokenizer, ds_dir, cfg, device) -> list:
    tcfg, mcfg = cfg["train"], cfg["model"]
    triples = data.load_triples(ds_dir)
    if not triples:
        raise RuntimeError(f"Пустой {ds_dir}/triples.tsv")

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

    pbar = tqdm(range(max_steps), desc=f"train:{cfg['version']}", unit=" шаг")
    for step in pbar:
        if ptr + bs > len(order):
            order = rng.permutation(len(triples))
            ptr = 0
        batch = [triples[i] for i in order[ptr:ptr + bs]]
        ptr += bs
        queries = [t[0] for t in batch]
        docs = [t[1] for t in batch] + [t[2] for t in batch]  # positives, затем hard negatives

        q_enc = tok(tokenizer, queries, mcfg["max_len_query"], device, special=True)
        d_enc = tok(tokenizer, docs, mcfg["max_len_doc"], device)

        # квадратичный разгон лямбд до flops_warmup_steps (SPLADE v2)
        lam_scale = min(1.0, ((step + 1) / max(1, tcfg["flops_warmup_steps"])) ** 2)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            q = model.encode_queries(q_enc["input_ids"], q_enc["attention_mask"],
                                     q_enc.get("special_tokens_mask"))
            d = model.encode_docs(d_enc["input_ids"], d_enc["attention_mask"])
            scores = q @ d.t()  # B x 2B: позитив запроса i — в столбце i (in-batch + hard neg)
            labels = torch.arange(len(batch), device=device)
            loss = F.cross_entropy(scores, labels)
            if model.query_encoder == "mlm" and tcfg["lambda_q"] > 0:
                loss = loss + lam_scale * tcfg["lambda_q"] * flops_loss(q)
            if tcfg["lambda_d"] > 0:
                loss = loss + lam_scale * tcfg["lambda_d"] * flops_loss(d)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        if step == 0 or (step + 1) % tcfg["log_every"] == 0:
            window = losses[-tcfg["log_every"]:]
            pbar.set_postfix(loss=f"{np.mean(window):.4f}",
                             lr=f"{scheduler.get_last_lr()[0]:.2e}")
    return losses


def run_experiment(cfg: dict, data_cfg: dict, run_id: str = None):
    """Полный прогон одной версии: train -> save model -> eval -> артефакты.

    cfg      — конфиг эксперимента (модель/обучение/eval + version, mode);
    data_cfg — конфиг данных (urls, data_dir, размеры smoke/full).
    """
    validate_config(cfg)
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
    print(f"[run] {version} mode={mode} device={device} -> {run_dir}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["hf_model"])
    model = Splade(cfg["model"]["hf_model"], cfg["model"]["query_encoder"]).to(device)

    losses = train_loop(model, tokenizer, ds_dir, cfg, device)

    model_dir = run_dir / "model"
    model.mlm.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    metrics = evaluate.evaluate_retrieval(model, tokenizer, ds_dir, cfg, device)
    metrics["final_train_loss"] = float(np.mean(losses[-10:]))
    metrics["train_steps"] = cfg["train"]["max_steps"]
    artifacts.save_metrics(run_dir, metrics)
    artifacts.finish_meta(run_dir, meta)
    print(f"[run] готово: {run_dir}")
    print(f"[run] метрики: {metrics}")
    return run_dir, metrics
