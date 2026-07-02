import pytest
import yaml

from core import config as config_mod


def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    return path


BASE = {
    "name": "base",
    "model": {"hf_model": "m", "query_encoder": "mlm", "max_len_query": 32,
              "max_len_doc": 256, "encode_batch_docs": 8, "encode_batch_queries": 8},
    "data": {"train_pool": "p.tsv", "train_triples": 100, "sample_seed": "auto"},
    "train": {"seeds": [1, 2], "lr": 1e-5, "batch_size": 4, "max_steps": 10,
              "warmup_steps": 2, "flops_warmup_steps": 5, "lambda_q": 0.1,
              "lambda_d": 0.1, "log_every": 5},
    "eval": {"datasets": ["gate"], "topk": 10, "max_queries": None,
             "batch_size_search": 8, "save_index": False},
}


def test_extends_merge(tmp_path):
    _write(tmp_path, "base.yaml", BASE)
    _write(tmp_path, "child.yaml",
           {"extends": "base.yaml", "name": "child",
            "train": {"lr": 3e-5}})
    cfg = config_mod.load_config(tmp_path / "child.yaml")
    assert cfg["name"] == "child"
    assert cfg["train"]["lr"] == 3e-5
    assert cfg["train"]["batch_size"] == 4
    config_mod.validate(cfg)


def test_validate_missing_key(tmp_path):
    broken = {k: v for k, v in BASE.items() if k != "eval"}
    with pytest.raises(KeyError):
        config_mod.validate(broken)


def test_validate_unknown_dataset():
    cfg = yaml.safe_load(yaml.safe_dump(BASE))
    cfg["eval"]["datasets"] = ["no-such-dataset"]
    with pytest.raises(KeyError):
        config_mod.validate(cfg)


def test_expand_seeds():
    variants = config_mod.expand_seeds(yaml.safe_load(yaml.safe_dump(BASE)))
    assert [v["train"]["seed"] for v in variants] == [1, 2]
    assert all("seeds" not in v["train"] for v in variants)


def test_sweep_expansion():
    cfg = yaml.safe_load(yaml.safe_dump(BASE))
    sweep = config_mod.parse_sweep(["data.train_triples=10,20", "train.lr=1e-5"])
    variants = config_mod.expand_sweep(cfg, sweep)
    assert len(variants) == 2
    assert variants[0]["data"]["train_triples"] == 10
    assert variants[1]["data"]["train_triples"] == 20
    assert all(v["train"]["lr"] == 1e-5 for v in variants)
    assert variants[0]["name"] == "base@train_triples=10,lr=1e-05"


def test_set_path_unknown_key():
    cfg = yaml.safe_load(yaml.safe_dump(BASE))
    with pytest.raises(KeyError):
        config_mod.set_path(cfg, "train.nonexistent", 1)
