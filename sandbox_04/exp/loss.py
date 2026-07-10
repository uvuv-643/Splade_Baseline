import math

import torch
import torch.nn.functional as F


def info_nce(q_reps, doc_reps, extra_negatives=None):
    """InfoNCE с in-batch + hard негативами (SPLADE v2): doc_reps = [позитивы; негативы]
    размера 2B, позитив запроса i — столбец i матрицы score B×2B.
    extra_negatives — детачнутые строки банка прошлых батчей (XBM), добавляются
    столбцами негативов. Логиты приводятся к fp32: bf16-квантование (ulp ~0.5 при
    |score|~64) ломает softmax на тысячах близких столбцов."""
    scores = q_reps @ doc_reps.t()
    if extra_negatives is not None:
        scores = torch.cat([scores, q_reps @ extra_negatives.t()], dim=1)
    labels = torch.arange(q_reps.size(0), device=q_reps.device)
    return F.cross_entropy(scores.float(), labels)


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


class DualController:
    """Множитель Лагранжа для бюджета активности (GECO-стиль): λ движется
    мультипликативно по клипованному лог-отношению EMA(L0) к бюджету.
    Лог-отношение безразмерно и симметрично (равные скорости роста/спада, макс.
    ~exp(0.01)≈1% за шаг при eta=0.02); деадбенд ±5% гасит предельные циклы."""

    def __init__(self, lam_init, budget, eta=0.02, lam_min=1e-6, lam_max=3e-2):
        self.lam = float(lam_init)
        self.budget = float(budget)
        self.eta = float(eta)
        self.lam_min = float(lam_min)
        self.lam_max = float(lam_max)
        self.ema = None

    def update(self, l0, allow=True):
        self.ema = float(l0) if self.ema is None else 0.9 * self.ema + 0.1 * float(l0)
        drive = math.log(max(self.ema, 1e-3) / self.budget)
        drive = min(max(drive, -0.5), 0.5)
        if allow and abs(drive) >= 0.05:
            self.lam = min(max(self.lam * math.exp(self.eta * drive),
                               self.lam_min), self.lam_max)
        return self.lam


class NegativeBank:
    """Кольцевой банк доковых представлений прошлых батчей (XBM, Wang et al. 2020;
    negative cache, Lindgren et al. 2021). Хранит детачнутые dense-активации:
    дополнительные негативы для InfoNCE почти бесплатны по компьюту, градиент
    через банк не течёт (негативы толкают только запрос)."""

    def __init__(self, size):
        self.size = int(size)
        self.buf = None
        self.ptr = 0
        self.filled = 0

    def push(self, reps):
        reps = reps.detach()
        if self.buf is None:
            self.buf = torch.zeros(self.size, reps.shape[1],
                                   dtype=reps.dtype, device=reps.device)
        n = min(reps.shape[0], self.size)
        idx = torch.arange(self.ptr, self.ptr + n, device=reps.device) % self.size
        self.buf[idx] = reps[:n]
        self.ptr = (self.ptr + n) % self.size
        self.filled = min(self.filled + n, self.size)

    def get(self):
        return self.buf[:self.filled] if self.filled else None
