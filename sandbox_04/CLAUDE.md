# Правила работы в sandbox_04

## Слои изменяемости

- `core/` — ЗАМОРОЖЕНО. Не менять без явного запроса человека. Любое изменение
  меняет `core_hash` и начинает новую «эпоху» сравнимости запусков.
- `data/eval/` + `manifest.json` — заморожено навсегда. Никогда не перезаписывать.
- `exp/` — менять свободно. Запуски исполняются из снапшотов, поэтому правки
  не ломают очередь.
- `configs/` — менять свободно.

## Контракт эксперимента (единственный интерфейс exp ↔ core)

```python
# exp/train.py
def train(cfg: dict, ctx) -> Encoder
def load(model_dir, cfg, device) -> Encoder
# Encoder.encode_queries/encode_docs(texts: list[str]) -> scipy.sparse.csr_matrix (n, vocab)
```

`ctx.log(step, loss=...)` — обязателен на каждом log_every-шаге (NaN = fail).
Веса сохранять в `ctx.model_dir`. Eval делает только core.

## Как ставить эксперименты

```bash
./lab queue configs/<cfg>.yaml       # gate (~2 мин) + очередь
./lab sweep configs/<cfg>.yaml key=v1,v2
./lab status / ./lab tail <run> / ./lab compare --exp <имя>
```

Состояние платформы читается напрямую из файлов: `queue/*.yaml`,
`runs/<id>/{status,metrics.json,stdout.log}`, `snapshots/index.json`.

## Запреты

- Не запускать обучение мимо `lab` (иначе нет снапшота/guards/артефактов).
- Не редактировать файлы внутри `runs/` и `snapshots/` руками.
- Не добавлять eval-датасеты без обновления manifest через `lab data prepare`
  и явного согласования с человеком.
- Комментарии в коде — только про неочевидную математику/протоколы.
