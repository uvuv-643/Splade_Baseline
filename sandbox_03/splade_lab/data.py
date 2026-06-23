"""MS MARCO: скачивание, подготовка smoke/full, загрузка с диска.

Единая схема каталога data/msmarco/<mode>/ для обоих режимов:
  collection.tsv  pid \\t text
  queries.tsv     qid \\t text
  qrels.tsv       qid \\t 0 \\t pid \\t 1
  triples.tsv     query \\t positive \\t negative   (сырые тексты, для обучения)

Smoke строится из первых строк triples.train.small (в файле есть полные тексты):
первые N строк -> train, следующие уникальные запросы -> eval, их positives +
прочие пассажи -> корпус. Детерминированно, полная коллекция не скачивается.
"""
import io
import shutil
import tarfile
from pathlib import Path

import requests

from .config import resolve_path
from .progress import tqdm  # файловый прогресс вместо tqdm (пишет в лог, не в блокнот)

FILES = ("collection.tsv", "queries.tsv", "qrels.tsv", "triples.tsv")


def dataset_dir(data_cfg: dict, mode: str) -> Path:
    return resolve_path(data_cfg["data_dir"]) / mode


def is_prepared(data_cfg: dict, mode: str) -> bool:
    d = dataset_dir(data_cfg, mode)
    return all((d / f).exists() for f in FILES)


def dataset_stats(ds_dir: Path) -> dict:
    """Сколько строк в каждом файле датасета — чтобы видеть, с чем работаем."""
    ds_dir = Path(ds_dir)
    stats = {}
    for name in FILES:
        path = ds_dir / name
        if not path.exists():
            stats[name] = 0
            continue
        with open(path, encoding="utf-8") as f:
            stats[name] = sum(1 for _ in f)
    return stats


# ---------- скачивание ----------

def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[data] есть {dest}, пропуск")
        return dest
    tmp = dest.with_name(dest.name + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total or None, unit="B",
                                        unit_scale=True, desc=dest.name) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)
    return dest


def _iter_text_lines(fileobj, encoding="utf-8", errors="replace",
                     chunk_size=1 << 20):
    """Декодирует бинарный поток в текстовые строки без io.TextIOWrapper.

    В режиме tarfile 'r|gz' tar.extractfile() отдаёт объект поверх внутреннего
    несеекаемого _Stream, у которого нет метода .seekable(). io.TextIOWrapper
    вызывает .seekable() при инициализации и падает с AttributeError. Здесь мы
    читаем сырые байты только вперёд и режем их по переводу строки, сохраняя
    '\\n' в конце (как делал бы TextIOWrapper при итерации)."""
    tail = b""
    while True:
        chunk = fileobj.read(chunk_size)
        if not chunk:
            break
        tail += chunk
        parts = tail.split(b"\n")
        tail = parts.pop()  # возможный неполный «хвост» последней строки
        for part in parts:
            yield part.decode(encoding, errors) + "\n"
    if tail:
        yield tail.decode(encoding, errors)


def _stream_tar_member_lines(url: str, suffix: str):
    """Стримит строки члена tar.gz по HTTP без сохранения архива.
    Ранний break у потребителя = ранний обрыв скачивания (экономия трафика)."""
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        r.raw.decode_content = False  # gz распаковывает tarfile, не requests
        with tarfile.open(fileobj=r.raw, mode="r|gz") as tar:
            for member in tar:
                if not member.name.endswith(suffix):
                    continue
                f = tar.extractfile(member)
                # NB: в потоковом режиме 'r|gz' f нельзя оборачивать в
                # io.TextIOWrapper — внутренний _Stream не поддерживает
                # seekable(); читаем и декодируем построчно вручную.
                yield from _iter_text_lines(f, encoding="utf-8", errors="replace")
                return
    raise RuntimeError(f"В {url} не найден член архива *{suffix}")


def _parse_triple(line: str):
    parts = line.rstrip("\n").split("\t")
    return parts if len(parts) == 3 else None


# ---------- smoke ----------

def prepare_smoke(data_cfg: dict, force: bool = False) -> Path:
    out = dataset_dir(data_cfg, "smoke")
    if is_prepared(data_cfg, "smoke") and not force:
        print(f"[data] smoke уже готов: {out}")
        return out
    p = data_cfg["smoke"]
    n_train, n_eval, n_corpus = p["num_train_triples"], p["num_eval_queries"], p["num_corpus_docs"]
    out.mkdir(parents=True, exist_ok=True)
    print(f"[data] smoke: стрим первых строк triples "
          f"({n_train} train, {n_eval} eval-запросов, корпус {n_corpus})")

    train_triples = []   # (q, pos, neg)
    eval_items = []      # (q, pos)
    corpus = {}          # text -> pid (дедупликация пассажей)
    train_queries, eval_queries = set(), set()

    def add_doc(text: str) -> int:
        if text not in corpus:
            corpus[text] = len(corpus)
        return corpus[text]

    stream = _stream_tar_member_lines(data_cfg["urls"]["triples"], ".tsv")
    with tqdm(desc="smoke: triples", unit=" строк") as bar:
        for line in stream:
            bar.update(1)
            triple = _parse_triple(line)
            if triple is None:
                continue
            q, pos, neg = triple
            if len(train_triples) < n_train:
                train_triples.append(triple)
                train_queries.add(q)
                continue
            if len(eval_items) < n_eval:
                if q not in train_queries and q not in eval_queries:
                    eval_queries.add(q)
                    eval_items.append((q, pos))
                add_doc(pos)
                add_doc(neg)
                continue
            if len(corpus) < n_corpus:
                add_doc(pos)
                add_doc(neg)
                continue
            break

    with open(out / "triples.tsv", "w", encoding="utf-8") as f:
        for q, pos, neg in train_triples:
            f.write(f"{q}\t{pos}\t{neg}\n")
    with open(out / "collection.tsv", "w", encoding="utf-8") as f:
        for text, pid in sorted(corpus.items(), key=lambda kv: kv[1]):
            f.write(f"{pid}\t{text}\n")
    with open(out / "queries.tsv", "w", encoding="utf-8") as fq, \
         open(out / "qrels.tsv", "w", encoding="utf-8") as fr:
        for qid, (q, pos) in enumerate(eval_items):
            fq.write(f"{qid}\t{q}\n")
            fr.write(f"{qid}\t0\t{corpus[pos]}\t1\n")

    print(f"[data] smoke готов: {out} | triples={len(train_triples)} "
          f"eval_q={len(eval_items)} corpus={len(corpus)}")
    return out


# ---------- full ----------

FULL_WARNING = """[!] mode=full — полный MS MARCO. Ресурсы:
    скачивание: collection ~1GB трафика (на диске ~3.2GB), queries ~18MB, срез triples ~2-3GB;
    обучение 50k шагов: часы на A100; кодирование 8.8M пассажей: ~1-2 ч;
    поиск по 6980 запросам: десятки минут; RAM до ~30GB; диск ~10GB в data/."""


def prepare_full(data_cfg: dict, force: bool = False) -> Path:
    out = dataset_dir(data_cfg, "full")
    if is_prepared(data_cfg, "full") and not force:
        print(f"[data] full уже готов: {out}")
        return out
    print(FULL_WARNING)
    p = data_cfg["full"]
    urls = data_cfg["urls"]
    raw = resolve_path(data_cfg["data_dir"]) / "raw"
    out.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)

    # 1) collection.tsv: стрим-распаковка сразу на диск (~3.2GB), архив не хранится
    coll = out / "collection.tsv"
    if not coll.exists():
        tmp = coll.with_name(coll.name + ".part")
        with open(tmp, "w", encoding="utf-8") as f:
            for line in tqdm(_stream_tar_member_lines(urls["collection"], "collection.tsv"),
                             desc="collection.tsv", unit=" строк", unit_scale=True,
                             total=8841823):
                f.write(line if line.endswith("\n") else line + "\n")
        tmp.rename(coll)

    # 2) qrels dev small
    qrels_raw = _download(urls["qrels_dev"], raw / "qrels.dev.small.tsv")
    shutil.copyfile(qrels_raw, out / "qrels.tsv")
    qrel_qids = {ln.split("\t")[0] for ln in qrels_raw.read_text(encoding="utf-8").splitlines()
                 if ln.strip()}

    # 3) queries: только dev-запросы, имеющие qrels; срез по num_eval_queries
    arch = _download(urls["queries"], raw / "queries.tar.gz")
    n_eval = p["num_eval_queries"]
    with tarfile.open(arch, "r:gz") as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("queries.dev.tsv"))
        lines = io.TextIOWrapper(tar.extractfile(member), encoding="utf-8")
        kept = [ln.rstrip("\n") for ln in lines if ln.split("\t")[0] in qrel_qids]
    kept.sort(key=lambda ln: int(ln.split("\t")[0]))  # детерминированный порядок
    if n_eval and n_eval > 0:
        kept = kept[:n_eval]
    (out / "queries.tsv").write_text("\n".join(kept) + "\n", encoding="utf-8")

    # 4) triples: первые num_train_triples строк (стрим с ранним обрывом)
    triples = out / "triples.tsv"
    if not triples.exists():
        n_train = p["num_train_triples"]
        tmp = triples.with_name(triples.name + ".part")
        written = 0
        with open(tmp, "w", encoding="utf-8") as f:
            for line in tqdm(_stream_tar_member_lines(urls["triples"], ".tsv"),
                             desc="triples.tsv", unit=" строк", unit_scale=True,
                             total=n_train if n_train and n_train > 0 else None):
                if n_train and 0 < n_train <= written:
                    break
                if _parse_triple(line) is None:
                    continue
                f.write(line if line.endswith("\n") else line + "\n")
                written += 1
        tmp.rename(triples)

    print(f"[data] full готов: {out}")
    return out


# ---------- загрузка ----------

def _read_tsv(path: Path, n_cols: int, desc: str = None):
    with open(path, encoding="utf-8") as f:
        lines = tqdm(f, desc=desc, unit=" строк", unit_scale=True) if desc else f
        for line in lines:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == n_cols:
                yield parts


def load_collection(ds_dir: Path):
    pids, texts = [], []
    for pid, text in _read_tsv(Path(ds_dir) / "collection.tsv", 2, desc="load:collection"):
        pids.append(pid)
        texts.append(text)
    return pids, texts


def load_queries(ds_dir: Path):
    qids, texts = [], []
    for qid, text in _read_tsv(Path(ds_dir) / "queries.tsv", 2):
        qids.append(qid)
        texts.append(text)
    return qids, texts


def load_qrels(ds_dir: Path) -> dict:
    qrels = {}
    for qid, _, pid, rel in _read_tsv(Path(ds_dir) / "qrels.tsv", 4):
        if int(rel) > 0:
            qrels.setdefault(qid, set()).add(pid)
    return qrels


def load_triples(ds_dir: Path, limit=None) -> list:
    triples = []
    for triple in _read_tsv(Path(ds_dir) / "triples.tsv", 3, desc="load:triples"):
        triples.append(tuple(triple))
        if limit and len(triples) >= limit:
            break
    return triples
