import math

import numpy as np
import torch
from transformers import AutoTokenizer

from .data import TriplePool, sample_indices
from .loss import DualController, NegativeBank, flops, flops_scale, info_nce
from .model import Splade, SpladeEncoder, tokenize

COLLAPSE_PATIENCE = 100


def _lr_lambda(step, warmup, max_steps):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    return max(0.0, (max_steps - step) / max(1, max_steps - warmup))


def train(cfg: dict, ctx) -> SpladeEncoder:
    mcfg, tcfg, dcfg = cfg["model"], cfg["train"], cfg["data"]
    tokenizer = AutoTokenizer.from_pretrained(mcfg["hf_model"])
    model = Splade(mcfg["hf_model"], mcfg["query_encoder"]).to(ctx.device)

    sample_seed = tcfg["seed"] if dcfg["sample_seed"] == "auto" else int(dcfg["sample_seed"])
    pool = TriplePool(ctx.root / dcfg["train_pool"])
    n_triples = len(pool) if dcfg["train_triples"] == "full" \
        else int(dcfg["train_triples"])
    sample = sample_indices(len(pool), n_triples, sample_seed)
    # max_steps=auto: компьют пропорционален данным — ровно один проход по выборке
    max_steps = math.ceil(n_triples / tcfg["batch_size"]) \
        if tcfg["max_steps"] == "auto" else tcfg["max_steps"]
    print(f"[train] {n_triples} триплетов из {pool.path.name} "
          f"(seed выборки {sample_seed}), {max_steps} шагов", flush=True)

    # Бюджетный режим (PD-SPLADE): λ ведут dual-контроллеры к целевому L0,
    # квадратичный flops-прогрев остаётся внешним конвертом (защита от раннего
    # коллапса), при lr < 10% пика λ замораживается (anti-windup).
    budgeted = tcfg.get("budget_q", 0) > 0 and tcfg.get("budget_d", 0) > 0
    if budgeted:
        eta = tcfg.get("dual_eta", 0.02)
        ctrl_q = DualController(tcfg["lambda_q"], tcfg["budget_q"], eta)
        ctrl_d = DualController(tcfg["lambda_d"], tcfg["budget_d"], eta)
    bank = NegativeBank(tcfg.get("xbm_size", 0))
    xbm_start = tcfg.get("xbm_start_step", 0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: _lr_lambda(s, tcfg["warmup_steps"], max_steps))

    rng = np.random.default_rng(tcfg["seed"])
    order = rng.permutation(n_triples)
    bs = tcfg["batch_size"]
    use_amp = ctx.device.type == "cuda"
    model.train()
    ptr = 0
    window = []
    collapse_streak = 0
    for step in range(max_steps):
        if ptr + bs > len(order):
            order = rng.permutation(n_triples)
            ptr = 0
        batch = pool.read(sample[order[ptr:ptr + bs]])
        ptr += bs
        queries = [t[0] for t in batch]
        docs = [t[1] for t in batch] + [t[2] for t in batch]
        q_enc = tokenize(tokenizer, queries, mcfg["max_len_query"], ctx.device, special=True)
        d_enc = tokenize(tokenizer, docs, mcfg["max_len_doc"], ctx.device)
        lam = flops_scale(step, tcfg["flops_warmup_steps"])
        lr_now = scheduler.get_last_lr()[0]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            q = model.encode_queries(q_enc["input_ids"], q_enc["attention_mask"],
                                     q_enc.get("special_tokens_mask"))
            d = model.encode_docs(d_enc["input_ids"], d_enc["attention_mask"])
            extra = bank.get() if bank.size and step >= xbm_start else None
            loss_rank = info_nce(q, d, extra)
            loss_fq = flops(q) if model.query_encoder == "mlm" \
                else torch.zeros((), device=ctx.device)
            loss_fd = flops(d)
            if budgeted:
                l0_q = float((q > 0).sum(dim=1).float().mean())
                l0_d = float((d > 0).sum(dim=1).float().mean())
                # anti-windup с двух сторон: λ не интегрируется, пока конверт
                # глушит актуатор (lam<0.5) и когда lr уже затух (<10% пика)
                allow = lam >= 0.5 and lr_now >= 0.1 * tcfg["lr"]
                lam_q = ctrl_q.update(l0_q, allow=allow)
                lam_d = ctrl_d.update(l0_d, allow=allow)
            else:
                lam_q, lam_d = tcfg["lambda_q"], tcfg["lambda_d"]
            loss = loss_rank + lam * (lam_q * loss_fq + lam_d * loss_fd)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if bank.size:
            bank.push(d)
        if budgeted:
            # коллапс — поглощающее состояние с конечным лоссом: check_finite
            # его не видит, поэтому следим за EMA(L0) документов напрямую
            collapse_streak = collapse_streak + 1 if ctrl_d.ema < 1.0 else 0
            if collapse_streak >= COLLAPSE_PATIENCE:
                raise RuntimeError(
                    f"коллапс представлений: EMA(L0_d) < 1 последние "
                    f"{COLLAPSE_PATIENCE} шагов (шаг {step + 1})")
        window.append(float(loss))
        if step == 0 or (step + 1) % tcfg["log_every"] == 0:
            extras = {}
            if budgeted:
                extras = {"l0_q": round(ctrl_q.ema, 1), "l0_d": round(ctrl_d.ema, 1),
                          "lambda_q_dual": ctrl_q.lam, "lambda_d_dual": ctrl_d.lam}
            if bank.size:
                extras["xbm_filled"] = bank.filled
            ctx.log(step + 1,
                    loss=float(np.mean(window)),
                    loss_rank=float(loss_rank),
                    loss_flops_q=float(loss_fq),
                    loss_flops_d=float(loss_fd),
                    lr=float(lr_now),
                    flops_lambda_scale=round(lam, 4),
                    **extras)
            window = []

    model.mlm.save_pretrained(ctx.model_dir)
    tokenizer.save_pretrained(ctx.model_dir)
    return SpladeEncoder(model, tokenizer, cfg, ctx.device)


def load(model_dir, cfg: dict, device) -> SpladeEncoder:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = Splade(str(model_dir), cfg["model"]["query_encoder"])
    return SpladeEncoder(model, tokenizer, cfg, device)
