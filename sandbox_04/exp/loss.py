import torch
import torch.nn.functional as F


def info_nce(q_reps, doc_reps):
    """InfoNCE с in-batch + hard негативами (SPLADE v2): doc_reps = [позитивы; негативы]
    размера 2B, позитив запроса i — столбец i матрицы score B×2B."""
    scores = q_reps @ doc_reps.t()
    labels = torch.arange(q_reps.size(0), device=q_reps.device)
    return F.cross_entropy(scores, labels)


def margin_mse(q_reps, pos_reps, neg_reps, teacher_margin):
    """MarginMSE (Hofstätter et al., 2020): студент повторяет разрыв
    оценок учителя s(q,d+) − s(q,d−)."""
    student = (q_reps * pos_reps).sum(-1) - (q_reps * neg_reps).sum(-1)
    return F.mse_loss(student.float(), teacher_margin)


def flops(reps):
    """l_FLOPS = Σ_j (mean_i a_ij)² (Paria et al., 2020) — гладкая оценка
    числа умножений в инвертированном индексе."""
    return (reps.mean(dim=0) ** 2).sum()


def flops_scale(step, warmup_steps):
    """Квадратичный разгон λ с 0 до 1 за warmup_steps (SPLADE v2)."""
    return min(1.0, ((step + 1) / max(1, warmup_steps)) ** 2)
