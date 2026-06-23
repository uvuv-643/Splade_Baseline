"""Абстрактный слой источников данных для бенчмарка SPLADE (zero-shot eval).

Зачем модуль вообще нужен
-------------------------
Индексация полного MS MARCO (8.8M пассажей) при инференсе занимает ~1.5 ч —
это слишком долго, чтобы быстро проверять, что новая версия модели стала лучше.
Идея: обучение оставляем на MS MARCO (оно работает хорошо и не меняется), а для
*оценки* берём небольшие стандартные IR-наборы из коллекции BEIR. Например
SciFact — ~5K документов и ~300 запросов: индекс строится за секунды, а не часы.
Обученную на MS MARCO модель прогоняем по такому набору **zero-shot** (без
дообучения) — ровно так это делают в статьях SPLADE/BEIR.

Канонический формат (тот же, что у MS MARCO)
--------------------------------------------
Все наборы приводятся к той же схеме TSV, что и MS MARCO, поэтому читаются
теми же загрузчиками из ``splade_lab.data`` (load_collection/load_queries) без
их изменения:

    collection.tsv  pid \\t text
    queries.tsv     qid \\t text
    qrels.tsv       qid \\t 0 \\t pid \\t rel   (rel — градуированная релевантность)

``triples.tsv`` для этих наборов НЕ создаётся: они используются только для
оценки (обучающих троек у них нет). Поэтому проверка готовности здесь своя
(``is_prepared``), а не из ``data.py`` (где triples обязателен).

Источник данных: BEIR
---------------------
BEIR раздаёт каждый набор одним zip-архивом фиксированной структуры:
    <name>/corpus.jsonl    {"_id","title","text", ...}
    <name>/queries.jsonl   {"_id","text", ...}
    <name>/qrels/<split>.tsv   query-id \\t corpus-id \\t score   (с шапкой)
Текст документа собираем как ``title + " " + text`` (стандартная конвенция BEIR).
Запросы фильтруем по qrels выбранного сплита (в queries.jsonl лежат все сплиты).

Добавить новый набор = добавить запись в ``BEIR_DATASETS`` и сделать ещё один
ноутбук-копию (поменять только имя набора). Код менять не нужно.
"""
import json
import os
import zipfile
from pathlib import Path

import requests

from .config import resolve_path
from .progress import tqdm  # файловый прогресс вместо tqdm (пишет в лог, не в блокнот)

# Файлы канонической схемы, нужные для ОЦЕНКИ (без triples — обучения здесь нет).
BENCH_FILES = ("collection.tsv", "queries.tsv", "qrels.tsv")

# Зеркала BEIR (по одному zip на набор). Пробуются по порядку — первое доступное
# выигрывает. Если сеть закрыта фаерволом (в этом окружении исходящие соединения
# к внешним хостам отбиваются с [Errno 13] Permission denied), скачать нельзя —
# тогда работает ручной путь: положить zip в data/beir/raw/<name>.zip (см.
# prepare_beir и MANUAL_DOWNLOAD_HINT).
BEIR_MIRRORS = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip",
)
# Обратная совместимость: первый шаблон как одиночный URL.
BEIR_URL = BEIR_MIRRORS[0]


def _proxies():
    # Прокси можно задать переменными окружения (сэндбокс выпускает наружу только
    # через внутренний прокси): export BEIR_HTTPS_PROXY=http://host:port
    proxy = os.environ.get("BEIR_HTTPS_PROXY") or os.environ.get("HTTPS_PROXY")
    return {"https": proxy, "http": proxy} if proxy else None


MANUAL_DOWNLOAD_HINT = """[beir] Сеть закрыта — автоскачивание невозможно.
Ручной путь (один раз на набор {name!r}):
  1) на машине с интернетом скачайте zip BEIR-набора, напр.:
       {url}
  2) положите файл сюда (имя строго <name>.zip):
       {dest}
  3) перезапустите клетку — prepare_beir увидит готовый zip и распакует его.
Либо задайте прокси и повторите:
       export BEIR_HTTPS_PROXY=http://proxy.sandbox.yandex-team.ru:3128"""

# Реестр наборов, в формате которых уверены (все — стандартные BEIR zip'ы
# одной структуры). split — какой qrels-сплит брать для оценки. Числа в
# комментариях справочные (примерные размеры), нигде не используются как
# валидация — реальные размеры берутся из самих файлов.
BEIR_DATASETS = {
    # имя           сплит      ~корпус     ~запросов   главная метрика
    "scifact":   {"split": "test"},   # ~5.2K       ~300        nDCG@10  (ГЛАВНЫЙ)
    "nfcorpus":  {"split": "test"},   # ~3.6K       ~323        nDCG@10
    "scidocs":   {"split": "test"},   # ~25K        ~1000       nDCG@10
    "arguana":   {"split": "test"},   # ~8.7K       ~1406       nDCG@10
    "fiqa":      {"split": "test"},   # ~57K        ~648        nDCG@10
    "trec-covid":{"split": "test"},   # ~171K       ~50         nDCG@10
    "webis-touche2020": {"split": "test"},  # ~382K ~49         nDCG@10
}


def beir_config(name: str, data_dir: str = "data/beir",
                split: str = None, num_eval_queries: int = -1) -> dict:
    """Готовый конфиг данных для ноутбука: ``DATA = beir_config("scifact")``.

    Один словарь полностью описывает источник — чтобы новый ноутбук отличался
    от существующего ровно одной строкой (именем набора).
    """
    if name not in BEIR_DATASETS:
        raise KeyError(f"Неизвестный набор {name!r}. Известные: {sorted(BEIR_DATASETS)}")
    return {
        "name": name,
        "split": split or BEIR_DATASETS[name]["split"],
        "url": BEIR_URL.format(name=name),
        "data_dir": data_dir,            # относительно корня репозитория
        "num_eval_queries": num_eval_queries,  # -1 = все запросы сплита
    }


def dataset_dir(data_cfg: dict) -> Path:
    return resolve_path(data_cfg["data_dir"]) / data_cfg["name"]


def is_prepared(data_cfg: dict) -> bool:
    d = dataset_dir(data_cfg)
    return all((d / f).exists() for f in BENCH_FILES)


def dataset_stats(ds_dir: Path) -> dict:
    """Сколько строк в каждом файле — чтобы видеть масштаб набора."""
    ds_dir = Path(ds_dir)
    stats = {}
    for name in BENCH_FILES:
        path = ds_dir / name
        if not path.exists():
            stats[name] = 0
            continue
        with open(path, encoding="utf-8") as f:
            stats[name] = sum(1 for _ in f)
    return stats


# ---------- скачивание / распаковка ----------

def _download_one(url: str, dest: Path) -> Path:
    """Качает один URL в dest (через прокси из окружения, если задан)."""
    tmp = dest.with_name(dest.name + ".part")
    with requests.get(url, stream=True, timeout=300, proxies=_proxies()) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total or None, unit="B",
                                        unit_scale=True, desc=dest.name) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)
    return dest


def _ensure_zip(name: str, dest: Path) -> Path:
    """Гарантирует наличие zip набора на диске.

    Порядок: 1) если уже есть — берём как есть (ручной drop-in работает само
    собой); 2) иначе пробуем зеркала по очереди. Если все недоступны (закрытая
    сеть) — кидаем понятную ошибку с инструкцией по ручной загрузке.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[beir] есть {dest}, пропуск скачивания")
        return dest
    errors = []
    for tmpl in BEIR_MIRRORS:
        url = tmpl.format(name=name)
        try:
            print(f"[beir] качаю {url}")
            return _download_one(url, dest)
        except Exception as e:  # сеть закрыта / зеркало недоступно — пробуем дальше
            errors.append(f"  - {url}: {type(e).__name__}: {e}")
            print(f"[beir] не вышло с {url}: {type(e).__name__}")
    hint = MANUAL_DOWNLOAD_HINT.format(
        name=name, url=BEIR_MIRRORS[0].format(name=name), dest=dest)
    raise RuntimeError("Не удалось скачать ни с одного зеркала:\n"
                       + "\n".join(errors) + "\n\n" + hint)


def _extract(zip_path: Path, raw_dir: Path, name: str) -> Path:
    """Распаковывает архив; возвращает каталог с corpus.jsonl/queries.jsonl/qrels/."""
    target = raw_dir / name
    if (target / "corpus.jsonl").exists():
        print(f"[beir] уже распакован: {target}")
        return target
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
    # Архив BEIR кладёт всё в подпапку <name>/; если структура иная — ищем corpus.jsonl.
    if (target / "corpus.jsonl").exists():
        return target
    for p in raw_dir.rglob("corpus.jsonl"):
        return p.parent
    raise RuntimeError(f"В {zip_path} не найден corpus.jsonl")


def _clean(text: str) -> str:
    """Убираем табы/переводы строк — иначе сломается построчный TSV."""
    return " ".join((text or "").split())


def _iter_jsonl(path: Path, desc: str):
    with open(path, encoding="utf-8") as f:
        for line in tqdm(f, desc=desc, unit=" строк"):
            line = line.strip()
            if line:
                yield json.loads(line)


def _read_qrels(qrels_path: Path):
    """Читает BEIR qrels (query-id, corpus-id, score) с шапкой -> список троек."""
    rows = []
    with open(qrels_path, encoding="utf-8") as f:
        header = f.readline()  # пропускаем шапку 'query-id\tcorpus-id\tscore'
        if header and "query" not in header.lower():
            f.seek(0)  # шапки не было — откатываемся
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                qid, pid, score = parts[0], parts[1], parts[2]
                try:
                    rel = int(float(score))
                except ValueError:
                    continue
                rows.append((qid, pid, rel))
    return rows


# ---------- подготовка ----------

def prepare_beir(data_cfg: dict, force: bool = False) -> Path:
    """Скачивает BEIR-набор и приводит к канонической схеме TSV.

    Идемпотентно: если TSV уже на месте — ничего не качает (как prepare_full).
    """
    out = dataset_dir(data_cfg)
    if is_prepared(data_cfg) and not force:
        print(f"[beir] {data_cfg['name']} уже готов: {out}")
        return out

    name, split = data_cfg["name"], data_cfg["split"]
    raw = resolve_path(data_cfg["data_dir"]) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[beir] подготовка набора {name!r} (split={split}) -> {out}")

    zip_path = _ensure_zip(name, raw / f"{name}.zip")
    src = _extract(zip_path, raw, name)

    # 1) qrels выбранного сплита -> множество нужных запросов и qrels.tsv
    qrels_rows = _read_qrels(src / "qrels" / f"{split}.tsv")
    if not qrels_rows:
        raise RuntimeError(f"Пустой qrels: {src / 'qrels' / (split + '.tsv')}")
    qids_in_qrels = {qid for qid, _, _ in qrels_rows}

    # ограничение числа eval-запросов (детерминированно: по сортировке id)
    n_eval = data_cfg.get("num_eval_queries", -1)
    if n_eval and n_eval > 0:
        keep_qids = set(sorted(qids_in_qrels)[:n_eval])
    else:
        keep_qids = qids_in_qrels

    with open(out / "qrels.tsv", "w", encoding="utf-8") as f:
        for qid, pid, rel in qrels_rows:
            if qid in keep_qids:
                f.write(f"{qid}\t0\t{pid}\t{rel}\n")

    # 2) queries.jsonl -> queries.tsv (только запросы из qrels-сплита)
    n_q = 0
    with open(out / "queries.tsv", "w", encoding="utf-8") as f:
        for obj in _iter_jsonl(src / "queries.jsonl", "beir:queries"):
            qid = str(obj.get("_id", obj.get("id")))
            if qid in keep_qids:
                f.write(f"{qid}\t{_clean(obj.get('text', ''))}\n")
                n_q += 1

    # 3) corpus.jsonl -> collection.tsv (title + text)
    n_c = 0
    with open(out / "collection.tsv", "w", encoding="utf-8") as f:
        for obj in _iter_jsonl(src / "corpus.jsonl", "beir:corpus"):
            pid = str(obj.get("_id", obj.get("id")))
            text = _clean(f"{obj.get('title', '')} {obj.get('text', '')}")
            f.write(f"{pid}\t{text}\n")
            n_c += 1

    print(f"[beir] {name} готов: corpus={n_c} queries={n_q} "
          f"qrels_pairs={sum(1 for q, _, _ in qrels_rows if q in keep_qids)}")
    return out


# ==========================================================================
# Офлайн-источник: маленький быстрый набор из ЛОКАЛЬНОГО MS MARCO (без сети)
# ==========================================================================
# Зачем: в закрытом окружении исходящие соединения к BEIR/HF/GitHub отбиваются
# фаерволом ([Errno 13] Permission denied), поэтому zip BEIR не скачать. Но
# полный MS MARCO уже лежит на диске (data/msmarco/full). Из него можно собрать
# небольшой набор для быстрой оценки: ВСЕ dev-запросы с qrels + их релевантные
# пассажи (правда из judgements) + случайные дистракторы до небольшого лимита.
# Такой корпус индексируется за секунды (а не ~1.5 ч на 8.8M пассажей), оценка
# идёт по настоящим qrels, метрики версий сопоставимы между собой. Сеть не нужна.


def msmarco_subset_config(num_corpus_docs: int = 50000,
                          num_eval_queries: int = -1,
                          msmarco_full_dir: str = "data/msmarco/full",
                          data_dir: str = "data/beir",
                          name: str = "msmarco_dev_small",
                          seed: int = 0) -> dict:
    """Конфиг офлайн-набора (тот же интерфейс, что beir_config).

    num_corpus_docs — размер корпуса (позитивы qrels + дистракторы до лимита);
    меньше корпус = быстрее индекс. num_eval_queries=-1 — все dev-запросы.
    """
    return {
        "name": name,
        "split": "dev",
        "source": "msmarco_local",
        "msmarco_full_dir": msmarco_full_dir,
        "num_corpus_docs": num_corpus_docs,
        "num_eval_queries": num_eval_queries,
        "data_dir": data_dir,
        "seed": seed,
    }


def prepare_msmarco_subset(data_cfg: dict, force: bool = False) -> Path:
    """Собирает маленький набор из локального MS MARCO. Сеть не используется.

    Однопроходное чтение collection.tsv (8.8M строк): сохраняем все нужные
    позитивы и набираем дистракторы до num_corpus_docs. Детерминированно (seed).
    Результат — та же каноническая схема TSV (collection/queries/qrels).
    """
    out = dataset_dir(data_cfg)
    if is_prepared(data_cfg) and not force:
        print(f"[offline] {data_cfg['name']} уже готов: {out}")
        return out

    full = resolve_path(data_cfg["msmarco_full_dir"])
    coll_path = full / "collection.tsv"
    q_path = full / "queries.tsv"
    qrels_path = full / "qrels.tsv"
    for p in (coll_path, q_path, qrels_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Нет {p}. Офлайн-набор строится из локального MS MARCO full — "
                f"сначала подготовьте его основным блокнотом (mode=full).")

    out.mkdir(parents=True, exist_ok=True)
    n_corpus = int(data_cfg.get("num_corpus_docs", 50000))
    n_eval = int(data_cfg.get("num_eval_queries", -1))
    rng = __import__("random").Random(data_cfg.get("seed", 0))

    # 1) qrels: qid -> {pid: rel}. Берём срез запросов (детерминированно по id).
    qrels = {}
    with open(qrels_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 4:
                qid, _, pid, rel = parts
                if int(rel) > 0:
                    qrels.setdefault(qid, {})[pid] = int(rel)
    all_qids = sorted(qrels)
    if n_eval and n_eval > 0:
        all_qids = all_qids[:n_eval]
    keep_qids = set(all_qids)
    positive_pids = {pid for q in keep_qids for pid in qrels[q]}
    print(f"[offline] запросов с qrels={len(keep_qids)} позитивов={len(positive_pids)} "
          f"цель корпуса={n_corpus}")

    # 2) queries.tsv: только выбранные запросы
    n_q = 0
    with open(q_path, encoding="utf-8") as fin, \
         open(out / "queries.tsv", "w", encoding="utf-8") as fout:
        for line in fin:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2 and parts[0] in keep_qids:
                fout.write(f"{parts[0]}\t{parts[1]}\n")
                n_q += 1

    # 3) collection.tsv: один проход. Все позитивы + дистракторы до лимита.
    #    Вероятность отбора дистрактора задаём из доли target/total, со скромным
    #    оверсэмплом — так дистракторы распределены по ВСЕЙ коллекции, а не берутся
    #    только из её начала; стоп при достижении лимита (детерминированно по seed).
    total_docs = 8841823
    n_distract_target = max(0, n_corpus - len(positive_pids))
    p_keep = min(1.0, (n_distract_target / total_docs) * 1.3) if n_distract_target else 0.0
    kept_docs = {}        # pid -> text
    n_distract = 0        # счётчик добранных дистракторов (без O(n) пересчёта)
    with open(coll_path, encoding="utf-8") as fin:
        for line in tqdm(fin, desc="offline:scan-collection", unit=" строк",
                         unit_scale=True, total=total_docs):
            tab = line.find("\t")
            if tab < 0:
                continue
            pid = line[:tab]
            if pid in positive_pids:
                kept_docs[pid] = line[tab + 1:].rstrip("\n")
            elif n_distract < n_distract_target and rng.random() < p_keep:
                kept_docs[pid] = line[tab + 1:].rstrip("\n")
                n_distract += 1

    # гарантируем, что все позитивы на месте (вдруг прореживание их не затронуло)
    with open(out / "collection.tsv", "w", encoding="utf-8") as f:
        for pid, text in kept_docs.items():
            f.write(f"{pid}\t{text}\n")

    # 4) qrels.tsv: только по реально присутствующим документам
    n_pairs = 0
    with open(out / "qrels.tsv", "w", encoding="utf-8") as f:
        for qid in keep_qids:
            for pid, rel in qrels[qid].items():
                if pid in kept_docs:
                    f.write(f"{qid}\t0\t{pid}\t{rel}\n")
                    n_pairs += 1

    print(f"[offline] готов: {out} | corpus={len(kept_docs)} queries={n_q} "
          f"qrels_pairs={n_pairs}")
    return out


def prepare(data_cfg: dict, force: bool = False) -> Path:
    """Единая точка: BEIR (скачивание) или офлайн-набор из локального MS MARCO.

    Выбор по ключу 'source' в конфиге: 'msmarco_local' -> офлайн, иначе BEIR.
    """
    if data_cfg.get("source") == "msmarco_local":
        return prepare_msmarco_subset(data_cfg, force=force)
    return prepare_beir(data_cfg, force=force)
