"""SPLADE поверх MLM-головы (SPLADE v2, arXiv:2109.10086).

Представление: w_j = max_i log(1 + ReLU(logits_ij))  (SPLADE-max, eq. F6).
Запрос: "mlm" — тот же энкодер (SPLADE-max); "bow" — бинарный мешок токенов
запроса без инференса (SPLADE-doc).
"""
import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM

QUERY_ENCODERS = ("mlm", "bow")


class Splade(nn.Module):
    def __init__(self, hf_model: str, query_encoder: str = "mlm"):
        super().__init__()
        if query_encoder not in QUERY_ENCODERS:
            raise ValueError(f"query_encoder={query_encoder!r}, ожидается {QUERY_ENCODERS}")
        self.mlm = AutoModelForMaskedLM.from_pretrained(hf_model)
        self.query_encoder = query_encoder
        self.vocab_size = self.mlm.config.vocab_size

    def _splade_rep(self, input_ids, attention_mask):
        logits = self.mlm(input_ids=input_ids, attention_mask=attention_mask).logits  # B,L,V
        sat = torch.log1p(torch.relu(logits))
        sat = sat * attention_mask.unsqueeze(-1).to(sat.dtype)  # паддинг не участвует
        return sat.max(dim=1).values  # B,V

    def encode_docs(self, input_ids, attention_mask):
        return self._splade_rep(input_ids, attention_mask)

    def encode_queries(self, input_ids, attention_mask, special_tokens_mask=None):
        if self.query_encoder == "mlm":
            return self._splade_rep(input_ids, attention_mask)
        # bow: 1 на позициях токенов запроса, спецтокены вырезаны
        keep = attention_mask
        if special_tokens_mask is not None:
            keep = keep * (1 - special_tokens_mask)
        rep = torch.zeros(input_ids.size(0), self.vocab_size,
                          device=input_ids.device, dtype=torch.float32)
        rep.scatter_(1, input_ids, keep.float())
        return rep


def flops_loss(reps: torch.Tensor) -> torch.Tensor:
    """l_FLOPS = sum_j (mean_i a_ij)^2 (eq. F4)."""
    return (reps.mean(dim=0) ** 2).sum()


def tok(tokenizer, texts, max_len: int, device, special: bool = False) -> dict:
    enc = tokenizer(list(texts), padding=True, truncation=True, max_length=max_len,
                    return_tensors="pt", return_special_tokens_mask=special)
    return {k: v.to(device) for k, v in enc.items()}
