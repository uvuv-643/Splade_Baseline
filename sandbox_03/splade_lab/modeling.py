"""Загрузка уже обученных моделей SPLADE из outputs/ для zero-shot оценки.

Зачем
-----
Обучение (train.py) сохраняет каждую модель в
``outputs/<version>/<run_id>/model/`` (веса MLM + токенайзер) и кладёт рядом
``config.json`` (там query_encoder, max_len и т.д.). Для бенчмарка нам нужно
только *загрузить* такую модель и прогнать инференс — без обучения.

Этот модуль НЕ дублирует и НЕ меняет ``model.py``: он лишь использует готовый
класс ``Splade`` и читает сохранённые артефакты. Веса/токенайзер грузятся
строго из локального каталога (``from_pretrained`` по пути), поэтому обращения
к huggingface.co не происходит — это важно на машинах без доступа в сеть к HF
(в логах основного блокнота как раз видны таймауты до huggingface.co).

Поддерживается выбор любого прогона: последнего по времени, конкретного run_id
или произвольного пути с моделью. Это и есть «легко подгружать разные модели».
"""
import json
from pathlib import Path

from transformers import AutoTokenizer

from .config import OUTPUTS_DIR
from .model import Splade


def list_model_runs(version: str, outputs_dir=OUTPUTS_DIR) -> list:
    """Все прогоны версии с сохранённой моделью (model/config.json), по возрастанию.

    run_id = timestamp YYYYMMDD-HHMMSS, поэтому сортировка по имени = по времени.
    """
    base = Path(outputs_dir) / version
    return [d for d in sorted(base.glob("*"))
            if d.is_dir() and (d / "model" / "config.json").exists()]


def find_latest_model_run(version: str, outputs_dir=OUTPUTS_DIR) -> Path:
    """Самый свежий прогон версии с сохранённой моделью."""
    runs = list_model_runs(version, outputs_dir)
    if not runs:
        raise FileNotFoundError(
            f"Нет прогонов с сохранённой моделью в {Path(outputs_dir) / version}")
    return runs[-1]


def resolve_model_run(version: str = None, run_id: str = None, run_dir=None,
                      outputs_dir=OUTPUTS_DIR) -> Path:
    """Унифицированный выбор каталога прогона.

    Приоритет: явный run_dir > (version, run_id) > последний прогон version.
    Возвращает каталог прогона (внутри которого есть model/ и config.json).
    """
    if run_dir is not None:
        run_dir = Path(run_dir)
        if (run_dir / "model" / "config.json").exists():
            return run_dir
        raise FileNotFoundError(f"В {run_dir} нет model/config.json")
    if version is None:
        raise ValueError("Укажи либо run_dir, либо version (+ опц. run_id)")
    if run_id is not None:
        cand = Path(outputs_dir) / version / run_id
        if (cand / "model" / "config.json").exists():
            return cand
        raise FileNotFoundError(f"В {cand} нет model/config.json")
    return find_latest_model_run(version, outputs_dir)


def load_run_config(run_dir: Path) -> dict:
    """Конфиг исходного прогона (нужен query_encoder, max_len_*)."""
    cfg_path = Path(run_dir) / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Нет {cfg_path}")
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def load_model(run_dir: Path, device, query_encoder: str = None):
    """Грузит сохранённую модель + токенайзер из run_dir/model/ (локально).

    query_encoder по умолчанию берётся из config.json прогона (mlm/bow);
    можно переопределить аргументом (например, чтобы прогнать ту же тушку
    в режиме bow). Возвращает (model, tokenizer, cfg).
    """
    run_dir = Path(run_dir)
    model_dir = run_dir / "model"
    cfg = load_run_config(run_dir)
    qe = query_encoder or cfg["model"]["query_encoder"]

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = Splade(str(model_dir), qe).to(device)
    model.eval()
    return model, tokenizer, cfg
