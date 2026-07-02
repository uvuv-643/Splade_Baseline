import json

from .hashing import params_key
from .paths import CACHE_DIR


def cache_dir(kind: str, params: dict):
    return CACHE_DIR / f"{kind}-{params_key(params)}"


def get_or_compute(kind: str, params: dict, compute):
    d = cache_dir(kind, params)
    marker = d / "DONE"
    if marker.exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    compute(d)
    (d / "params.json").write_text(
        json.dumps(params, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    marker.write_text("")
    return d
