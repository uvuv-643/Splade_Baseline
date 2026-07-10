#!/usr/bin/env python3
"""Готовит полный train-пул: все триплеты triples.train.small (~39.8M строк,
~27GB на диске) в data/train/triples-full.tsv. Скачивание — стримом, без
промежуточного tar на диске. По готовности печатает точную команду свипа
compute-scaling (верхняя точка сетки = весь пул)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from core.data import URLS, _count_lines, _stream_tar_member_lines
from core.paths import TRAIN_DIR

DEST = TRAIN_DIR / "triples-full.tsv"
GRID = [100_000, 200_000, 400_000, 800_000, 1_600_000, 3_200_000,
        6_400_000, 12_800_000, 25_600_000]


def prepare() -> int:
    if DEST.exists():
        print(f"[data] есть {DEST}, пропуск (пересоздать: удалите файл)")
        return _count_lines(DEST)
    DEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = DEST.with_name(DEST.name + ".part")
    written = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for line in tqdm(_stream_tar_member_lines(URLS["triples"], ".tsv"),
                         desc="triples-full", unit=" строк", unit_scale=True):
            if len(line.split("\t")) == 3:
                f.write(line + "\n")
                written += 1
    tmp.rename(DEST)
    return written


def main():
    total = prepare()
    print(f"[data] полный пул: {total} триплетов -> {DEST}")
    values = ",".join(str(v) for v in GRID + [total])
    print("\nкоманда свипа (10 точек, верхняя = весь пул):")
    print(f"./lab sweep configs/compute_scaling.yaml data.train_triples={values}")


if __name__ == "__main__":
    main()
