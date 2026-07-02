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
