import hashlib
import json
from pathlib import Path


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dir_hash(root, exclude_parts=("__pycache__", ".ipynb_checkpoints")) -> str:
    root = Path(root)
    h = hashlib.sha256()
    files = sorted(p for p in root.rglob("*")
                   if p.is_file() and not any(part in p.parts for part in exclude_parts))
    for p in files:
        h.update(str(p.relative_to(root)).encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def core_hash() -> str:
    from .paths import ROOT
    core_dir = ROOT / "core"
    h = hashlib.sha256()
    files = sorted(p for p in core_dir.rglob("*.py")
                   if "__pycache__" not in p.parts and "tests" not in p.parts)
    for p in files:
        h.update(str(p.relative_to(core_dir)).encode())
        h.update(b"\0")
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def params_key(params: dict) -> str:
    payload = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
    return sha256_text(payload)[:16]
