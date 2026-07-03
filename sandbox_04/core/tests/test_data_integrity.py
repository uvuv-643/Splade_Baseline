import pytest

from core import data

needs_data = pytest.mark.skipif(
    not data.MANIFEST_PATH.exists(),
    reason="eval-данные не подготовлены (lab data prepare)")


@needs_data
def test_manifest_integrity():
    assert data.verify_manifest() == []


@needs_data
def test_datasets_files_present():
    for name in data.DATASETS:
        assert data.datasets_prepared([name]) == [], f"{name} не готов"


@needs_data
def test_eval_data_hash_stable():
    h1 = data.eval_data_hash(["msmarco-dev-lite", "trec-dl-2019-lite"])
    h2 = data.eval_data_hash(["trec-dl-2019-lite", "msmarco-dev-lite"])
    assert h1 == h2
    assert h1 != data.eval_data_hash(["msmarco-dev-lite"])


@needs_data
def test_gate_corpus_size():
    pids, texts = data.load_corpus("gate")
    assert len(pids) == data.GATE_CORPUS_SIZE
    assert len(set(pids)) == len(pids)


@needs_data
@pytest.mark.skipif(not data.TRAIN_POOL.exists(), reason="нет train-пула")
def test_leakage_train_eval():
    overlaps = data.check_leakage()
    qids, _ = data.load_queries("msmarco-dev")
    assert overlaps["msmarco-dev"] <= max(1, len(qids) // 100), \
        f"подозрительная утечка train→dev: {overlaps}"


@needs_data
@pytest.mark.parametrize("name", list(data.BEIR_SETS))
def test_beir_qrels_alignment(name):
    """Для каждого подготовленного BEIR-набора: qrels-запросы присутствуют в
    queries.tsv, а подавляющее большинство qrels-документов — в корпусе. Наборы,
    которые ещё не скачаны, пропускаем — их проверит verify после prepare.

    NB: у некоторых BEIR-наборов (напр. ArguAna) единичные qrels ссылаются на
    пассажи, которых нет в официальном corpus.jsonl — это известная особенность
    самого BEIR, а не ошибка скачивания. Такие «висячие» judged-документы просто
    никогда не будут найдены (чуть занижают recall) и ничего не ломают в eval.
    Поэтому требуем не строгое отсутствие, а долю < 1%."""
    if data.datasets_prepared([name]):
        pytest.skip(f"{name} не подготовлен")
    qids, _ = data.load_queries(name)
    qrels = data.load_qrels(name)
    assert qrels, f"{name}: пустые qrels"
    assert set(qrels).issubset(set(qids)), \
        f"{name}: есть qrels-запросы без строки в queries.tsv"
    pid_set = set(data.load_corpus(name)[0])
    judged_pids = {p for rel_map in qrels.values() for p in rel_map}
    missing = judged_pids - pid_set
    frac = len(missing) / max(1, len(judged_pids))
    assert frac < 0.01, (f"{name}: {len(missing)}/{len(judged_pids)} "
                         f"({frac:.1%}) judged-документов нет в корпусе — "
                         f"подозрительно много, проверьте скачивание")
