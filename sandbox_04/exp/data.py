import numpy as np


def count_lines(path) -> int:
    with open(path, "rb") as f:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1 << 24), b""))


def sample_triples(pool_path, n, seed) -> list:
    """Равномерная выборка n триплетов без возвращения из пула;
    детерминирована по seed, один проход по файлу."""
    total = count_lines(pool_path)
    if n > total:
        raise ValueError(f"нужно {n} триплетов, в пуле {pool_path} только {total}")
    rng = np.random.default_rng(seed)
    picked = np.sort(rng.choice(total, size=n, replace=False))
    triples = []
    it = iter(picked.tolist())
    target = next(it, None)
    with open(pool_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if target is None:
                break
            if i == target:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 3:
                    triples.append(tuple(parts))
                target = next(it, None)
    if len(triples) != n:
        raise RuntimeError(f"выбрано {len(triples)} из {n} — битые строки в пуле")
    return triples
