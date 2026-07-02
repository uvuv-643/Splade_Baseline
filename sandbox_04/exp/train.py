import numpy as np
import torch
from transformers import AutoTokenizer

from .data import sample_triples
from .loss import flops, flops_scale, info_nce
from .model import Splade, SpladeEncoder, tokenize


def _lr_lambda(step, warmup, max_steps):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    return max(0.0, (max_steps - step) / max(1, max_steps - warmup))


def train(cfg: dict, ctx) -> SpladeEncoder:
    mcfg, tcfg, dcfg = cfg["model"], cfg["train"], cfg["data"]
    tokenizer = AutoTokenizer.from_pretrained(mcfg["hf_model"])
    model = Splade(mcfg["hf_model"], mcfg["query_encoder"]).to(ctx.device)

    sample_seed = tcfg["seed"] if dcfg["sample_seed"] == "auto" else int(dcfg["sample_seed"])
    pool = ctx.root / dcfg["train_pool"]
    triples = sample_triples(pool, dcfg["train_triples"], sample_seed)
    print(f"[train] {len(triples)} триплетов из {pool.name} (seed выборки {sample_seed})",
          flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: _lr_lambda(s, tcfg["warmup_steps"], tcfg["max_steps"]))

    rng = np.random.default_rng(tcfg["seed"])
    order = rng.permutation(len(triples))
    bs = tcfg["batch_size"]
    use_amp = ctx.device.type == "cuda"
    model.train()
    ptr = 0
    window = []
    for step in range(tcfg["max_steps"]):
        if ptr + bs > len(order):
            order = rng.permutation(len(triples))
            ptr = 0
        batch = [triples[i] for i in order[ptr:ptr + bs]]
        ptr += bs
        queries = [t[0] for t in batch]
        docs = [t[1] for t in batch] + [t[2] for t in batch]
        q_enc = tokenize(tokenizer, queries, mcfg["max_len_query"], ctx.device, special=True)
        d_enc = tokenize(tokenizer, docs, mcfg["max_len_doc"], ctx.device)
        lam = flops_scale(step, tcfg["flops_warmup_steps"])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            q = model.encode_queries(q_enc["input_ids"], q_enc["attention_mask"],
                                     q_enc.get("special_tokens_mask"))
            d = model.encode_docs(d_enc["input_ids"], d_enc["attention_mask"])
            loss_rank = info_nce(q, d)
            loss_fq = flops(q) if model.query_encoder == "mlm" \
                else torch.zeros((), device=ctx.device)
            loss_fd = flops(d)
            loss = loss_rank + lam * (tcfg["lambda_q"] * loss_fq
                                      + tcfg["lambda_d"] * loss_fd)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        window.append(float(loss))
        if step == 0 or (step + 1) % tcfg["log_every"] == 0:
            ctx.log(step + 1,
                    loss=float(np.mean(window)),
                    loss_rank=float(loss_rank),
                    loss_flops_q=float(loss_fq),
                    loss_flops_d=float(loss_fd),
                    lr=float(scheduler.get_last_lr()[0]),
                    flops_lambda_scale=round(lam, 4))
            window = []

    model.mlm.save_pretrained(ctx.model_dir)
    tokenizer.save_pretrained(ctx.model_dir)
    return SpladeEncoder(model, tokenizer, cfg, ctx.device)


def load(model_dir, cfg: dict, device) -> SpladeEncoder:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = Splade(str(model_dir), cfg["model"]["query_encoder"])
    return SpladeEncoder(model, tokenizer, cfg, device)
