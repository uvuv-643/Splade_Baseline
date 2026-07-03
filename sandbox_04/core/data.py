import gzip
import json
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

from .hashing import sha256_file, sha256_text
from .paths import EVAL_DIR, TRAIN_DIR

URLS = {
    "collection": "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz",
    "queries": "https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz",
    "qrels_dev": "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.small.tsv",
    "triples": "https://msmarco.z22.web.core.windows.net/msmarcoranking/triples.train.small.tar.gz",
    "trec-dl-2019-queries": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz",
    "trec-dl-2020-queries": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz",
    "trec-dl-2019-qrels": "https://trec.nist.gov/data/deep/2019qrels-pass.txt",
    "trec-dl-2020-qrels": "https://trec.nist.gov/data/deep/2020qrels-pass.txt",
}

DATASETS = {
    "msmarco-dev": {"corpus": "msmarco-full", "queryset": "msmarco-dev", "rel_threshold": 1},
    "trec-dl-2019": {"corpus": "msmarco-full", "queryset": "trec-dl-2019", "rel_threshold": 2},
    "trec-dl-2020": {"corpus": "msmarco-full", "queryset": "trec-dl-2020", "rel_threshold": 2},
    "msmarco-dev-lite": {"corpus": "msmarco-lite", "queryset": "msmarco-dev", "rel_threshold": 1},
    "trec-dl-2019-lite": {"corpus": "msmarco-lite", "queryset": "trec-dl-2019", "rel_threshold": 2},
    "trec-dl-2020-lite": {"corpus": "msmarco-lite", "queryset": "trec-dl-2020", "rel_threshold": 2},
    "gate": {"corpus": "gate", "queryset": "gate-dev", "rel_threshold": 1},
}

# --- BEIR zero-shot наборы (строки таблицы исходной статьи SPLADE/BEIR) ---
# Каждый набор самодостаточен: свой корпус + свои запросы + свои qrels (в отличие
# от MS MARCO, где корпус общий). Поэтому и corpus, и queryset называем именем
# набора. rel_threshold=1: BEIR-qrels бинарны или градуированы, nDCG@10 (главная
# метрика BEIR) считает любые rel>0, а recall/MRR — rel>=1 (стандарт BEIR).
# Ключ слева — как в нашем реестре; PAPER_LABEL — строка таблицы статьи; archive —
# имя zip на зеркале BEIR; split — какой qrels-сплит берём для оценки.
BEIR_MIRROR = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{archive}.zip"
BEIR_SETS = {
    # ключ (=corpus=queryset)   archive              split   PAPER_LABEL
    "beir-arguana":       {"archive": "arguana",        "split": "test", "paper": "ArguAna"},
    "beir-climate-fever": {"archive": "climate-fever",  "split": "test", "paper": "Climate-FEVER"},
    "beir-dbpedia":       {"archive": "dbpedia-entity", "split": "test", "paper": "DBPedia"},
    "beir-fever":         {"archive": "fever",          "split": "test", "paper": "FEVER"},
    "beir-fiqa":          {"archive": "fiqa",           "split": "test", "paper": "FiQA-2018"},
    "beir-hotpotqa":      {"archive": "hotpotqa",       "split": "test", "paper": "HotpotQA"},
    "beir-nfcorpus":      {"archive": "nfcorpus",       "split": "test", "paper": "NFCorpus"},
    "beir-nq":            {"archive": "nq",             "split": "test", "paper": "NQ"},
    "beir-quora":         {"archive": "quora",          "split": "test", "paper": "Quora"},
    "beir-scidocs":       {"archive": "scidocs",        "split": "test", "paper": "SCIDOCS"},
    "beir-scifact":       {"archive": "scifact",        "split": "test", "paper": "SciFact"},
    "beir-trec-covid":    {"archive": "trec-covid",     "split": "test", "paper": "TREC-COVID"},
    "beir-touche2020":    {"archive": "webis-touche2020","split": "test", "paper": "Touché-2020"},
}
# Небольшие корпуса (< ~600k пассажей): кодируются за минуты, годятся для smoke.
BEIR_SMALL = ("beir-arguana", "beir-fiqa", "beir-nfcorpus", "beir-quora",
              "beir-scidocs", "beir-scifact", "beir-trec-covid", "beir-touche2020")
# Крупные корпуса (миллионы пассажей): проверяются на целостность, но в smoke не
# кодируются — их гоняют полноценным прогоном/до-eval обученной модели.
BEIR_LARGE = ("beir-climate-fever", "beir-dbpedia", "beir-fever",
              "beir-hotpotqa", "beir-nq")
for _name, _spec in BEIR_SETS.items():
    DATASETS[_name] = {"corpus": _name, "queryset": _name, "rel_threshold": 1}

CORPORA_DIR = EVAL_DIR / "corpora"
QUERYSETS_DIR = EVAL_DIR / "querysets"
RAW_DIR = EVAL_DIR / "raw"
MANIFEST_PATH = EVAL_DIR / "manifest.json"
TRAIN_POOL = TRAIN_DIR / "triples-2m.tsv"

COLLECTION_LINES = 8_841_823
LITE_CORPUS_SIZE = 1_000_000
GATE_CORPUS_SIZE = 1_000
GATE_QUERIES = 50
TRAIN_POOL_SIZE = 2_000_000
SAMPLING_SEED = 0


def corpus_path(name: str) -> Path:
    return CORPORA_DIR / name / "collection.tsv"


def queryset_paths(name: str):
    d = QUERYSETS_DIR / name
    return d / "queries.tsv", d / "qrels.tsv"


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
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


def _iter_text_lines(fileobj, chunk_size=1 << 20):
    # tar 'r|gz' отдаёт несеекаемый поток без .seekable() — TextIOWrapper на нём падает,
    # поэтому байты декодируются и режутся по '\n' вручную
    tail = b""
    while True:
        chunk = fileobj.read(chunk_size)
        if not chunk:
            break
        tail += chunk
        parts = tail.split(b"\n")
        tail = parts.pop()
        for part in parts:
            yield part.decode("utf-8", "replace")
    if tail:
        yield tail.decode("utf-8", "replace")


def _stream_tar_member_lines(url: str, suffix: str):
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        r.raw.decode_content = False
        with tarfile.open(fileobj=r.raw, mode="r|gz") as tar:
            for member in tar:
                if member.name.endswith(suffix):
                    yield from _iter_text_lines(tar.extractfile(member))
                    return
    raise RuntimeError(f"в {url} нет члена архива *{suffix}")


def _count_lines(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1 << 24), b""))


def _write_lines(dest: Path, lines):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with open(tmp, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line if line.endswith("\n") else line + "\n")
    tmp.rename(dest)


def prepare_collection(force=False):
    dest = corpus_path("msmarco-full")
    if dest.exists() and not force:
        print(f"[data] есть {dest}, пропуск")
        return
    print("[data] msmarco-full: стрим-распаковка collection.tar.gz (~3.2GB на диске)")
    lines = tqdm(_stream_tar_member_lines(URLS["collection"], "collection.tsv"),
                 desc="collection.tsv", unit=" строк", unit_scale=True,
                 total=COLLECTION_LINES)
    _write_lines(dest, lines)


def prepare_dev_queryset(force=False):
    qpath, rpath = queryset_paths("msmarco-dev")
    if qpath.exists() and rpath.exists() and not force:
        print(f"[data] есть {qpath.parent}, пропуск")
        return
    qrels_raw = _download(URLS["qrels_dev"], RAW_DIR / "qrels.dev.small.tsv")
    rows = []
    for line in qrels_raw.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 4:
            rows.append((parts[0], parts[2], int(parts[3])))
    qids_with_qrels = {qid for qid, _, _ in rows}
    _write_lines(rpath, (f"{qid}\t0\t{pid}\t{rel}" for qid, pid, rel in rows))

    arch = _download(URLS["queries"], RAW_DIR / "queries.tar.gz")
    kept = []
    with tarfile.open(arch, "r:gz") as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("queries.dev.tsv"))
        for line in _iter_text_lines(tar.extractfile(member)):
            qid = line.split("\t", 1)[0]
            if qid in qids_with_qrels:
                kept.append(line)
    kept.sort(key=lambda ln: int(ln.split("\t", 1)[0]))
    _write_lines(qpath, kept)
    print(f"[data] msmarco-dev: {len(kept)} запросов, {len(rows)} qrels")


def prepare_trec_queryset(year: int, force=False):
    name = f"trec-dl-{year}"
    qpath, rpath = queryset_paths(name)
    if qpath.exists() and rpath.exists() and not force:
        print(f"[data] есть {qpath.parent}, пропуск")
        return
    qrels_raw = _download(URLS[f"{name}-qrels"], RAW_DIR / f"{year}qrels-pass.txt")
    rows = []
    for line in qrels_raw.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 4:
            rows.append((parts[0], parts[2], int(parts[3])))
    judged_qids = {qid for qid, _, _ in rows}
    _write_lines(rpath, (f"{qid}\t0\t{pid}\t{rel}" for qid, pid, rel in rows))

    gz = _download(URLS[f"{name}-queries"], RAW_DIR / f"msmarco-test{year}-queries.tsv.gz")
    kept = []
    with gzip.open(gz, "rt", encoding="utf-8") as f:
        for line in f:
            qid = line.split("\t", 1)[0]
            if qid in judged_qids:
                kept.append(line.rstrip("\n"))
    kept.sort(key=lambda ln: int(ln.split("\t", 1)[0]))
    _write_lines(qpath, kept)
    print(f"[data] {name}: {len(kept)} запросов, {len(rows)} qrels")


# --- BEIR: скачивание zip, распаковка, приведение к канонической схеме TSV ---

def _beir_clean(text: str) -> str:
    """Убираем табы/переводы строк — иначе построчный TSV поедет по столбцам."""
    return " ".join((text or "").split())


def _beir_iter_jsonl(path: Path, desc: str):
    with open(path, encoding="utf-8") as f:
        for line in tqdm(f, desc=desc, unit=" строк", unit_scale=True):
            line = line.strip()
            if line:
                yield json.loads(line)


def _beir_read_qrels(qrels_path: Path):
    """BEIR qrels: 'query-id\\tcorpus-id\\tscore' с шапкой -> [(qid, pid, rel)]."""
    rows = []
    with open(qrels_path, encoding="utf-8") as f:
        header = f.readline()
        if header and "query" not in header.lower():
            f.seek(0)  # шапки не было — откатываемся к первой строке
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                try:
                    rows.append((parts[0], parts[1], int(float(parts[2]))))
                except ValueError:
                    continue
    return rows


def _beir_extract(zip_path: Path, raw_dir: Path, archive: str) -> Path:
    """Распаковывает zip; возвращает каталог с corpus.jsonl/queries.jsonl/qrels/."""
    target = raw_dir / archive
    if (target / "corpus.jsonl").exists():
        return target
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
    if (target / "corpus.jsonl").exists():
        return target
    for p in raw_dir.rglob("corpus.jsonl"):  # структура иная — ищем corpus.jsonl
        return p.parent
    raise RuntimeError(f"в {zip_path} не найден corpus.jsonl")


def prepare_beir_set(name: str, force=False):
    """Готовит один BEIR-набор: качает zip, распаковывает и пишет
    corpora/<name>/collection.tsv + querysets/<name>/{queries,qrels}.tsv.
    Корпус: title+text (конвенция BEIR). Запросы: только те, что есть в qrels
    выбранного сплита (в queries.jsonl лежат все сплиты)."""
    spec = BEIR_SETS[name]
    cpath = corpus_path(name)
    qpath, rpath = queryset_paths(name)
    if cpath.exists() and qpath.exists() and rpath.exists() and not force:
        print(f"[data] есть {name}, пропуск")
        return
    archive, split = spec["archive"], spec["split"]
    url = BEIR_MIRROR.format(archive=archive)
    zip_dest = _download(url, RAW_DIR / f"{archive}.zip")
    src = _beir_extract(zip_dest, RAW_DIR, archive)

    qrels_rows = _beir_read_qrels(src / "qrels" / f"{split}.tsv")
    if not qrels_rows:
        raise RuntimeError(f"пустой qrels: {src / 'qrels' / (split + '.tsv')}")
    keep_qids = {qid for qid, _, _ in qrels_rows}
    _write_lines(rpath, (f"{qid}\t0\t{pid}\t{rel}" for qid, pid, rel in qrels_rows))

    n_q = 0
    q_lines = []
    for obj in _beir_iter_jsonl(src / "queries.jsonl", f"{name}:queries"):
        qid = str(obj.get("_id", obj.get("id")))
        if qid in keep_qids:
            q_lines.append(f"{qid}\t{_beir_clean(obj.get('text', ''))}")
            n_q += 1
    _write_lines(qpath, q_lines)

    n_c = 0

    def corpus_lines():
        nonlocal n_c
        for obj in _beir_iter_jsonl(src / "corpus.jsonl", f"{name}:corpus"):
            pid = str(obj.get("_id", obj.get("id")))
            text = _beir_clean(f"{obj.get('title', '')} {obj.get('text', '')}")
            n_c += 1
            yield f"{pid}\t{text}"

    _write_lines(cpath, corpus_lines())
    print(f"[data] {name} ({spec['paper']}): corpus={n_c} queries={n_q} "
          f"qrels={len(qrels_rows)}")


def prepare_beir(only=None, force=False):
    """Готовит набор BEIR-датасетов (по умолчанию — все из BEIR_SETS)."""
    names = list(only) if only else list(BEIR_SETS)
    for name in names:
        print(f"[data] === {name} ===")
        prepare_beir_set(name, force=force)


def prepare_beir_small(force=False):
    prepare_beir(only=BEIR_SMALL, force=force)


def _all_judged_pids() -> set:
    judged = set()
    for qs in ("msmarco-dev", "trec-dl-2019", "trec-dl-2020"):
        _, rpath = queryset_paths(qs)
        for line in rpath.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) == 4:
                judged.add(parts[2])
    return judged


def _subsample_corpus(src: Path, dest: Path, target_size: int, judged: set):
    """Детерминированный подкорпус: все judged-пассажи + равномерная выборка
    остальных до target_size (rng PCG64, seed фиксирован — корпус заморожен)."""
    judged_idx = []
    total = 0
    with open(src, encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc=f"scan:{src.parent.name}",
                                      unit=" строк", unit_scale=True)):
            pid = line.split("\t", 1)[0]
            if pid in judged:
                judged_idx.append(i)
            total += 1
    judged_idx = np.asarray(judged_idx, dtype=np.int64)
    need = target_size - len(judged_idx)
    if need < 0:
        raise RuntimeError(f"judged-пассажей ({len(judged_idx)}) больше, чем "
                           f"target_size={target_size}")
    others = np.setdiff1d(np.arange(total, dtype=np.int64), judged_idx,
                          assume_unique=True)
    rng = np.random.default_rng(SAMPLING_SEED)
    sampled = rng.choice(others, size=need, replace=False)
    keep = np.zeros(total, dtype=bool)
    keep[judged_idx] = True
    keep[sampled] = True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with open(src, encoding="utf-8") as f, open(tmp, "w", encoding="utf-8") as out:
        for i, line in enumerate(f):
            if keep[i]:
                out.write(line if line.endswith("\n") else line + "\n")
    tmp.rename(dest)


def prepare_lite_corpus(force=False):
    dest = corpus_path("msmarco-lite")
    if dest.exists() and not force:
        print(f"[data] есть {dest}, пропуск")
        return
    judged = _all_judged_pids()
    print(f"[data] msmarco-lite: {LITE_CORPUS_SIZE} пассажей "
          f"({len(judged)} judged + выборка), seed={SAMPLING_SEED}")
    _subsample_corpus(corpus_path("msmarco-full"), dest, LITE_CORPUS_SIZE, judged)


def prepare_gate(force=False):
    qpath, rpath = queryset_paths("gate-dev")
    cpath = corpus_path("gate")
    if qpath.exists() and rpath.exists() and cpath.exists() and not force:
        print(f"[data] есть gate, пропуск")
        return
    dev_qpath, dev_rpath = queryset_paths("msmarco-dev")
    dev_queries = [ln for ln in dev_qpath.read_text(encoding="utf-8").splitlines() if ln]
    gate_queries = dev_queries[:GATE_QUERIES]
    gate_qids = {ln.split("\t", 1)[0] for ln in gate_queries}
    gate_qrels = []
    judged = set()
    for line in dev_rpath.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) == 4 and parts[0] in gate_qids:
            gate_qrels.append(line)
            judged.add(parts[2])
    _write_lines(qpath, gate_queries)
    _write_lines(rpath, gate_qrels)
    _subsample_corpus(corpus_path("msmarco-lite"), cpath, GATE_CORPUS_SIZE, judged)
    print(f"[data] gate: {len(gate_queries)} запросов, корпус {GATE_CORPUS_SIZE}")


def prepare_train_pool(force=False):
    if TRAIN_POOL.exists() and not force:
        print(f"[data] есть {TRAIN_POOL}, пропуск")
        return
    print(f"[data] train pool: первые {TRAIN_POOL_SIZE} триплетов "
          f"(query\\tpos\\tneg, сырые тексты)")
    TRAIN_POOL.parent.mkdir(parents=True, exist_ok=True)
    tmp = TRAIN_POOL.with_name(TRAIN_POOL.name + ".part")
    written = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for line in tqdm(_stream_tar_member_lines(URLS["triples"], ".tsv"),
                         desc="triples", unit=" строк", unit_scale=True,
                         total=TRAIN_POOL_SIZE):
            if written >= TRAIN_POOL_SIZE:
                break
            if len(line.split("\t")) == 3:
                f.write(line + "\n")
                written += 1
    tmp.rename(TRAIN_POOL)


def check_leakage() -> dict:
    """Утечка train→eval: пересечение нормализованных текстов запросов."""
    def norm(s):
        return " ".join(s.lower().split())
    train_queries = set()
    with open(TRAIN_POOL, encoding="utf-8") as f:
        for line in f:
            parts = line.split("\t")
            if len(parts) == 3:
                train_queries.add(norm(parts[0]))
    overlaps = {}
    for qs in ("msmarco-dev", "trec-dl-2019", "trec-dl-2020"):
        qpath, _ = queryset_paths(qs)
        eval_queries = {norm(ln.split("\t", 1)[1])
                        for ln in qpath.read_text(encoding="utf-8").splitlines()
                        if "\t" in ln}
        overlaps[qs] = len(eval_queries & train_queries)
    return overlaps


def _manifest_files() -> list:
    files = sorted(CORPORA_DIR.rglob("collection.tsv")) + sorted(QUERYSETS_DIR.rglob("*.tsv"))
    return [p for p in files if p.is_file()]


def write_manifest():
    files = {}
    for path in _manifest_files():
        rel = str(path.relative_to(EVAL_DIR))
        files[rel] = {"sha256": sha256_file(path), "lines": _count_lines(path)}
    MANIFEST_PATH.write_text(json.dumps({"files": files}, indent=2), encoding="utf-8")
    print(f"[data] manifest: {len(files)} файлов заморожено")


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def verify_manifest() -> list:
    if not MANIFEST_PATH.exists():
        return ["manifest.json отсутствует — запустите lab data prepare"]
    errors = []
    manifest = load_manifest()
    for rel, entry in manifest["files"].items():
        path = EVAL_DIR / rel
        if not path.exists():
            errors.append(f"{rel}: файл пропал")
            continue
        if _count_lines(path) != entry["lines"]:
            errors.append(f"{rel}: число строк не совпадает с manifest")
        elif sha256_file(path) != entry["sha256"]:
            errors.append(f"{rel}: sha256 не совпадает с manifest")
    return errors


def dataset_files(name: str) -> list:
    spec = DATASETS[name]
    qpath, rpath = queryset_paths(spec["queryset"])
    return [str(p.relative_to(EVAL_DIR))
            for p in (corpus_path(spec["corpus"]), qpath, rpath)]


def eval_data_hash(dataset_names) -> str:
    """Хэш замороженных eval-данных. sha256 берём из manifest; если файла там
    ещё нет (например BEIR подготовили после заморозки msmarco-манифеста) —
    считаем sha256 с диска на лету, чтобы прогон не падал. Обычные msmarco/trec
    наборы всегда в манифесте, так что fallback их не замедляет."""
    files = load_manifest().get("files", {})
    parts = []
    for ds in sorted(set(dataset_names)):
        for rel in dataset_files(ds):
            entry = files.get(rel)
            sha = entry["sha256"] if entry else sha256_file(EVAL_DIR / rel)
            parts.append(f"{rel}:{sha}")
    return sha256_text("\n".join(parts))[:12]


def datasets_prepared(names) -> list:
    missing = []
    for name in names:
        for rel in dataset_files(name):
            if not (EVAL_DIR / rel).exists():
                missing.append(rel)
    return sorted(set(missing))


def load_corpus(corpus_name: str):
    pids, texts = [], []
    with open(corpus_path(corpus_name), encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                pids.append(parts[0])
                texts.append(parts[1])
    return pids, texts


def load_queries(queryset: str, max_queries=None):
    qpath, _ = queryset_paths(queryset)
    qids, texts = [], []
    with open(qpath, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                qids.append(parts[0])
                texts.append(parts[1])
    if max_queries:
        qids, texts = qids[:max_queries], texts[:max_queries]
    return qids, texts


def load_qrels(queryset: str) -> dict:
    _, rpath = queryset_paths(queryset)
    qrels = {}
    with open(rpath, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 4:
                qrels.setdefault(parts[0], {})[parts[2]] = int(parts[3])
    return qrels


PREPARE_PARTS = {
    "collection": prepare_collection,
    "dev": prepare_dev_queryset,
    "trec": lambda force=False: (prepare_trec_queryset(2019, force),
                                 prepare_trec_queryset(2020, force)),
    "lite": prepare_lite_corpus,
    "gate": prepare_gate,
    "train-pool": prepare_train_pool,
    # BEIR: 'beir' — все 13 наборов таблицы статьи; 'beir-small' — только лёгкие
    # (годятся для smoke); 'beir-<key>' — один набор точечно (см. BEIR_SETS).
    "beir": prepare_beir,
    "beir-small": prepare_beir_small,
}
for _bname in BEIR_SETS:  # точечно: --part beir-scifact и т.п.
    PREPARE_PARTS[_bname] = (lambda n: (lambda force=False: prepare_beir_set(n, force)))(_bname)

# Части, которые собирает `lab data prepare --all`. BEIR сюда НЕ входит: он
# тяжёлый (десятки GB) и ставится осознанно через --part beir / beir-small.
CORE_PARTS = ("collection", "dev", "trec", "lite", "gate", "train-pool")


def prepare_all(force=False):
    for name in CORE_PARTS:
        print(f"[data] === {name} ===")
        PREPARE_PARTS[name](force=force)
    overlaps = check_leakage()
    print(f"[data] утечка train→eval (пересечение текстов запросов): {overlaps}")
    if any(overlaps.values()):
        print("[data] ВНИМАНИЕ: ненулевое пересечение — проверьте источники")
    write_manifest()


def status() -> dict:
    out = {"manifest": MANIFEST_PATH.exists(),
           "train_pool": TRAIN_POOL.exists(),
           "datasets": {}}
    for name in DATASETS:
        out["datasets"][name] = not datasets_prepared([name])
    return out
