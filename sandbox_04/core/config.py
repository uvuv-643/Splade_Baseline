import copy
import itertools
from pathlib import Path

import yaml

from .paths import CONFIGS_DIR

SCHEMA = {
    "model": ("hf_model", "query_encoder", "max_len_query", "max_len_doc",
              "encode_batch_docs", "encode_batch_queries"),
    "data": ("train_pool", "train_triples", "sample_seed"),
    "train": ("lr", "batch_size", "max_steps", "warmup_steps",
              "flops_warmup_steps", "lambda_q", "lambda_d", "log_every"),
    "eval": ("datasets", "topk", "max_queries", "batch_size_search", "save_index"),
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _resolve_cfg_path(path, base_dir=None) -> Path:
    path = Path(path)
    candidates = [path]
    if base_dir is not None:
        candidates.append(Path(base_dir) / path)
    candidates.append(CONFIGS_DIR / path)
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(f"конфиг не найден: {path}")


def load_config(path, base_dir=None) -> dict:
    p = _resolve_cfg_path(path, base_dir)
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    parent = cfg.pop("extends", None)
    if parent:
        cfg = deep_merge(load_config(parent, base_dir=p.parent), cfg)
    return cfg


def validate(cfg: dict):
    if not cfg.get("name"):
        raise KeyError("в конфиге нет ключа 'name'")
    for section, keys in SCHEMA.items():
        if section not in cfg:
            raise KeyError(f"в конфиге нет секции {section!r}")
        missing = [k for k in keys if k not in cfg[section]]
        if missing:
            raise KeyError(f"в cfg[{section!r}] не хватает ключей: {missing}")
    train = cfg["train"]
    if "seeds" not in train and "seed" not in train:
        raise KeyError("в cfg['train'] нужен 'seeds' (список) или 'seed'")
    if train.get("seeds") is not None and not isinstance(train["seeds"], list):
        raise KeyError("cfg['train']['seeds'] должен быть списком")
    from . import data
    unknown = [d for d in cfg["eval"]["datasets"] if d not in data.DATASETS]
    if unknown:
        raise KeyError(f"неизвестные eval-датасеты: {unknown} "
                       f"(доступны: {sorted(data.DATASETS)})")


def expand_seeds(cfg: dict) -> list:
    seeds = cfg["train"].get("seeds")
    if seeds is None:
        seeds = [cfg["train"]["seed"]]
    out = []
    for s in seeds:
        c = copy.deepcopy(cfg)
        c["train"].pop("seeds", None)
        c["train"]["seed"] = int(s)
        out.append(c)
    return out


def get_path(cfg: dict, dotted: str, default=None):
    node = cfg
    for k in dotted.split("."):
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def set_path(cfg: dict, dotted: str, value):
    node = cfg
    keys = dotted.split(".")
    for k in keys[:-1]:
        if not isinstance(node.get(k), dict):
            raise KeyError(f"нет секции {k!r} в пути {dotted!r}")
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(f"нет ключа {dotted!r} в конфиге")
    node[keys[-1]] = value


def parse_sweep(tokens: list) -> dict:
    sweep = {}
    for tok in tokens:
        if "=" not in tok:
            raise ValueError(f"ожидается key=v1,v2 — получено {tok!r}")
        key, raw = tok.split("=", 1)
        sweep[key.strip()] = [yaml.safe_load(v) for v in raw.split(",")]
    return sweep


def expand_sweep(cfg: dict, sweep: dict) -> list:
    if not sweep:
        return [copy.deepcopy(cfg)]
    keys = list(sweep)
    out = []
    for combo in itertools.product(*(sweep[k] for k in keys)):
        c = copy.deepcopy(cfg)
        for k, v in zip(keys, combo):
            set_path(c, k, v)
        suffix = ",".join(f"{k.split('.')[-1]}={v}" for k, v in zip(keys, combo))
        c["name"] = f"{cfg['name']}@{suffix}"
        out.append(c)
    return out


def dump(cfg: dict) -> str:
    return yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
