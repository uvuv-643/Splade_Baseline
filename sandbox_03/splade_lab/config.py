"""Общие пути и слияние конфигов. Сами конфиги экспериментов задаются в ноутбуке (dict)."""
import copy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "outputs"


def merge_config(base: dict, override: dict) -> dict:
    """Рекурсивное слияние: override поверх base, без мутации аргументов."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_config(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve_path(p) -> Path:
    """Относительные пути считаются от корня репозитория (где лежит ноутбук)."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def validate_config(cfg: dict):
    """Ранняя проверка ключей — чтобы падать с понятной ошибкой до обучения."""
    for section, keys in {
        "model": ("hf_model", "query_encoder", "max_len_query", "max_len_doc"),
        "train": ("seed", "lr", "batch_size", "max_steps", "warmup_steps",
                  "flops_warmup_steps", "lambda_q", "lambda_d", "log_every"),
        "eval": ("batch_size_docs", "batch_size_queries", "batch_size_search", "recall_ks"),
    }.items():
        if section not in cfg:
            raise KeyError(f"В конфиге нет секции {section!r}")
        missing = [k for k in keys if k not in cfg[section]]
        if missing:
            raise KeyError(f"В cfg[{section!r}] не хватает ключей: {missing}")
    for key in ("version", "mode"):
        if key not in cfg:
            raise KeyError(f"В конфиге нет ключа {key!r} (его добавляет build_config в ноутбуке)")
