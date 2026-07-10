from pathlib import Path

import numpy as np


def count_lines(path) -> int:
    with open(path, "rb") as f:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1 << 24), b""))


def line_offsets(pool_path) -> np.ndarray:
    """Байтовые смещения начала каждой строки пула; считаются одним проходом
    и кэшируются рядом с файлом (<имя>.offsets.npy)."""
    pool_path = Path(pool_path)
    cache = pool_path.with_name(pool_path.name + ".offsets.npy")
    if cache.exists() and cache.stat().st_mtime >= pool_path.stat().st_mtime:
        return np.load(cache)
    offsets = []
    pos = 0
    with open(pool_path, "rb") as f:
        for line in f:
            offsets.append(pos)
            pos += len(line)
    arr = np.asarray(offsets, dtype=np.int64)
    np.save(cache, arr)
    return arr


class TriplePool:
    """Пул триплетов с ленивым чтением строк по смещениям: в памяти только
    offsets (8 байт/строка), поэтому размер пула не ограничен RAM."""

    def __init__(self, path):
        self.path = Path(path)
        self.offsets = line_offsets(self.path)
        self._file = open(self.path, "rb")

    def __len__(self):
        return len(self.offsets)

    def read(self, indices) -> list:
        triples = []
        for i in indices:
            self._file.seek(int(self.offsets[i]))
            parts = self._file.readline().decode("utf-8").rstrip("\n").split("\t")
            if len(parts) != 3:
                raise RuntimeError(f"битая строка {i} в {self.path.name}")
            triples.append(tuple(parts))
        return triples


def sample_indices(total: int, n: int, seed) -> np.ndarray:
    """Равномерная выборка n индексов пула без возвращения; детерминирована по seed."""
    if n > total:
        raise ValueError(f"нужно {n} триплетов, в пуле только {total}")
    rng = np.random.default_rng(seed)
    return rng.choice(total, size=n, replace=False)
