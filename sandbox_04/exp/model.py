import array

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM

QUERY_ENCODERS = ("mlm", "bow")


class Splade(nn.Module):
    """SPLADE-max (arXiv:2109.10086): w_j = max_i log(1 + ReLU(logit_ij))."""

    def __init__(self, hf_model, query_encoder="mlm"):
        super().__init__()
        if query_encoder not in QUERY_ENCODERS:
            raise ValueError(f"query_encoder={query_encoder!r}, ожидается {QUERY_ENCODERS}")
        self.mlm = AutoModelForMaskedLM.from_pretrained(hf_model)
        self.query_encoder = query_encoder
        self.vocab_size = self.mlm.config.vocab_size

    def _rep(self, input_ids, attention_mask):
        logits = self.mlm(input_ids=input_ids, attention_mask=attention_mask).logits
        sat = torch.log1p(torch.relu(logits))
        sat = sat * attention_mask.unsqueeze(-1).to(sat.dtype)
        return sat.max(dim=1).values

    def encode_docs(self, input_ids, attention_mask):
        return self._rep(input_ids, attention_mask)

    def encode_queries(self, input_ids, attention_mask, special_tokens_mask=None):
        if self.query_encoder == "mlm":
            return self._rep(input_ids, attention_mask)
        keep = attention_mask
        if special_tokens_mask is not None:
            keep = keep * (1 - special_tokens_mask)
        rep = torch.zeros(input_ids.size(0), self.vocab_size,
                          device=input_ids.device, dtype=torch.float32)
        rep.scatter_(1, input_ids, keep.float())
        return rep


def tokenize(tokenizer, texts, max_len, device, special=False):
    enc = tokenizer(list(texts), padding=True, truncation=True, max_length=max_len,
                    return_tensors="pt", return_special_tokens_mask=special)
    return {k: v.to(device) for k, v in enc.items()}


class SpladeEncoder:
    """Контракт §6: encode_queries/encode_docs(texts) -> csr_matrix (n, vocab)."""

    def __init__(self, model, tokenizer, cfg, device):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.max_len_query = cfg["model"]["max_len_query"]
        self.max_len_doc = cfg["model"]["max_len_doc"]
        self.batch_queries = cfg["model"]["encode_batch_queries"]
        self.batch_docs = cfg["model"]["encode_batch_docs"]
        self.vocab_size = model.vocab_size

    @torch.no_grad()
    def _encode(self, texts, max_len, batch_size, kind) -> sp.csr_matrix:
        # nnz копятся в плоские буферы array.array и собираются в CSR zero-copy
        # через np.frombuffer — пик RAM ~ размер итоговой матрицы
        vals, cols = array.array("f"), array.array("i")
        indptr = array.array("q", [0])
        use_amp = self.device.type == "cuda"
        for start in range(0, len(texts), batch_size):
            enc = tokenize(self.tokenizer, texts[start:start + batch_size],
                           max_len, self.device, special=(kind == "query"))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                if kind == "query":
                    reps = self.model.encode_queries(
                        enc["input_ids"], enc["attention_mask"],
                        enc.get("special_tokens_mask"))
                else:
                    reps = self.model.encode_docs(enc["input_ids"], enc["attention_mask"])
            reps = reps.float().cpu()
            nz = reps.nonzero(as_tuple=False)
            rows = nz[:, 0]
            cols.frombytes(nz[:, 1].to(torch.int32).numpy().tobytes())
            vals.frombytes(reps[rows, nz[:, 1]].numpy().astype(np.float32, copy=False).tobytes())
            counts = torch.bincount(rows, minlength=reps.size(0))
            cum = (torch.cumsum(counts, 0) + indptr[-1]).numpy().astype(np.int64)
            indptr.frombytes(cum.tobytes())
        return sp.csr_matrix(
            (np.frombuffer(vals, dtype=np.float32),
             np.frombuffer(cols, dtype=np.int32),
             np.frombuffer(indptr, dtype=np.int64)),
            shape=(len(texts), self.vocab_size))

    def encode_queries(self, texts) -> sp.csr_matrix:
        return self._encode(texts, self.max_len_query, self.batch_queries, "query")

    def encode_docs(self, texts) -> sp.csr_matrix:
        return self._encode(texts, self.max_len_doc, self.batch_docs, "doc")
