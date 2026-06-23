# SPLADE: воспроизводимые эксперименты на MS MARCO

Обучение и сравнение моделей разреженного поиска по SPLADE v2 (arXiv:2109.10086).
Интерфейс — один ноутбук `splade_experiments.ipynb`: клетки по порядку, всё работает.
Конфиги экспериментов — питоновские словари прямо в ноутбуке; `splade_lab/` — только
общий код (данные, модель, обучение, метрики, артефакты).

Две версии: `v1_splade_max` (общий MLM-энкодер) и `v2_splade_doc` (только документы,
запрос — мешок токенов). Общий код, разные словари, общий датасет (скачивается один раз).

## Установка (на сервере)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip freeze > requirements.lock   # зафиксировать точное окружение прогонов
jupyter lab                      # и открыть splade_experiments.ipynb
```

Перенос с локальной машины:
`rsync -av --exclude data --exclude outputs --exclude .venv ./ user@server:~/diploma/`

## Запуск

Открыть `splade_experiments.ipynb`, запускать клетки по очереди:

1. Setup — устройство (cuda/mps/cpu).
2. Конфиг данных — `MODE = "smoke"` (по умолчанию) или `"full"`.
3. Конфиги экспериментов — все гиперпараметры печатаются перед запуском.
4. Данные — скачиваются один раз в `data/msmarco/<mode>/`, переиспользуются всеми версиями.
5. Прогоны — обучение + eval каждой версии.
6. Сравнение — сводная таблица + `outputs/comparison.csv`.

Smoke: ~10–30MB трафика, минуты. Full: ~1GB трафика, ~6GB диска, часы на A100 —
только осознанно, ноутбук печатает предупреждение.

## Артефакты прогона

`outputs/<version>/<run_id>/`:
- `config.json` — конфиг, как был на момент прогона;
- `metrics.json` — MRR@10, Recall@k, sparsity (avg nnz), loss;
- `meta.json` — время, host, устройство, seed, git commit, версии пакетов, sha256 конфига;
- `model/` — веса HF + токенизатор.

## Новая версия эксперимента

В клетке 3 добавить запись в `EXPERIMENTS` (переопределения поверх `BASE`):

```python
"v3_high_reg": {"train": {"lambda_d": 1e-3}},
```

и перезапустить клетки 3 → 5 → 6. Код не трогать.

## Структура

- `splade_experiments.ipynb` — единая точка запуска, конфиги-словари;
- `splade_lab/config.py` — пути, merge_config, валидация конфига;
- `splade_lab/data.py` — скачивание/подготовка/чтение MS MARCO (tqdm везде);
- `splade_lab/model.py` — SPLADE-энкодер (max-pooling; запрос mlm|bow), FLOPS-loss;
- `splade_lab/train.py` — обучение (InfoNCE + hard negative + FLOPS-рег.), run_experiment;
- `splade_lab/evaluate.py` — sparse-кодирование (CSR), поиск (GPU/CPU), MRR/Recall;
- `splade_lab/artifacts.py` — каталоги прогонов, config/metrics/meta;
- `splade_lab/compare.py` — сводная таблица.

## Данные

- Smoke: детерминированный сабсет из первых строк `triples.train.small`
  (тексты query/pos/neg уже в файле; полная коллекция не нужна). Eval-запросы
  не пересекаются с train, корпус = positives + дистракторы.
- Full: коллекция 8.8M пассажей, dev small (6980 запросов, qrels), срез triples
  на `num_train_triples` строк (по умолчанию 3.2M ≈ 50k шагов x 64).
- Источник: официальные файлы Microsoft (urls в клетке 2; если хост недоступен —
  зеркало `msmarco.blob.core.windows.net`).

## Воспроизводимость

- Seed фиксирован в конфиге (python/numpy/torch, cudnn.deterministic).
- Конфиг + метаданные + версии пакетов сохраняются в каждом прогоне.
- Smoke-данные детерминированы (первые строки файла, без случайности).
- Битовая воспроизводимость на GPU не гарантируется (atomics в cuda-ядрах),
  метрики воспроизводятся с точностью до незначительного шума.

## Ориентиры (SPLADE v2, full MS MARCO dev)

- SPLADE-max: MRR@10 ~ 0.340, R@1000 ~ 0.965 (150k шагов, batch 124).
- SPLADE-doc: MRR@10 ~ 0.322, R@1000 ~ 0.946.
- Здесь 50k шагов / batch 64 на 1xA100 — ожидаемо немного ниже статьи.
