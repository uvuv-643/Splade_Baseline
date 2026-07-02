import difflib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .hashing import dir_hash
from .paths import ROOT, SNAPSHOTS_DIR

EXP_DIR = ROOT / "exp"
IGNORE = shutil.ignore_patterns("__pycache__", ".ipynb_checkpoints", "*.pyc")


def _index_path(snapshots_dir) -> Path:
    return Path(snapshots_dir) / "index.json"


def load_index(snapshots_dir=SNAPSHOTS_DIR) -> dict:
    path = _index_path(snapshots_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_index(idx, snapshots_dir):
    path = _index_path(snapshots_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(idx, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _latest_hash(idx) -> str:
    if not idx:
        return None
    return max(idx.items(), key=lambda kv: kv[1]["created"])[0]


def save(name=None, message="", exp_dir=EXP_DIR, snapshots_dir=SNAPSHOTS_DIR) -> str:
    exp_dir = Path(exp_dir)
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"нет каталога {exp_dir}")
    h = dir_hash(exp_dir)[:12]
    idx = load_index(snapshots_dir)
    dest = Path(snapshots_dir) / h / "exp"
    if not dest.exists():
        shutil.copytree(exp_dir, dest, ignore=IGNORE)
    entry = idx.get(h, {})
    idx[h] = {
        "name": name or entry.get("name") or h,
        "description": message or entry.get("description", ""),
        "created": entry.get("created") or datetime.now(timezone.utc).isoformat(),
        "parent": entry.get("parent", _latest_hash({k: v for k, v in idx.items() if k != h})),
        "gate_ok": entry.get("gate_ok"),
    }
    _write_index(idx, snapshots_dir)
    return h


def mark_gate(h: str, ok: bool, snapshots_dir=SNAPSHOTS_DIR):
    idx = load_index(snapshots_dir)
    if h in idx:
        idx[h]["gate_ok"] = bool(ok)
        _write_index(idx, snapshots_dir)


def snapshot_dir(h: str, snapshots_dir=SNAPSHOTS_DIR) -> Path:
    d = Path(snapshots_dir) / h
    if not (d / "exp").is_dir():
        raise FileNotFoundError(f"снапшот {h} не найден в {snapshots_dir}")
    return d


def resolve(ref: str, snapshots_dir=SNAPSHOTS_DIR) -> str:
    idx = load_index(snapshots_dir)
    if ref in idx:
        return ref
    by_name = [h for h, e in idx.items() if e["name"] == ref]
    if len(by_name) == 1:
        return by_name[0]
    if len(by_name) > 1:
        newest = max(by_name, key=lambda h: idx[h]["created"])
        return newest
    by_prefix = [h for h in idx if h.startswith(ref)]
    if len(by_prefix) == 1:
        return by_prefix[0]
    raise KeyError(f"снапшот {ref!r} не найден (ни хэш, ни имя)")


def list_snapshots(snapshots_dir=SNAPSHOTS_DIR) -> list:
    idx = load_index(snapshots_dir)
    return sorted(({"hash": h, **e} for h, e in idx.items()),
                  key=lambda e: e["created"], reverse=True)


def diff_snapshots(hash_a: str, hash_b: str, snapshots_dir=SNAPSHOTS_DIR) -> str:
    dir_a = snapshot_dir(hash_a, snapshots_dir) / "exp"
    dir_b = snapshot_dir(hash_b, snapshots_dir) / "exp"
    files = sorted({str(p.relative_to(dir_a)) for p in dir_a.rglob("*") if p.is_file()} |
                   {str(p.relative_to(dir_b)) for p in dir_b.rglob("*") if p.is_file()})
    chunks = []
    for rel in files:
        a, b = dir_a / rel, dir_b / rel
        lines_a = a.read_text(encoding="utf-8").splitlines(keepends=True) if a.exists() else []
        lines_b = b.read_text(encoding="utf-8").splitlines(keepends=True) if b.exists() else []
        diff = list(difflib.unified_diff(lines_a, lines_b,
                                         fromfile=f"{hash_a}/exp/{rel}",
                                         tofile=f"{hash_b}/exp/{rel}"))
        if diff:
            chunks.append("".join(diff))
    return "\n".join(chunks) if chunks else "(код идентичен)"
