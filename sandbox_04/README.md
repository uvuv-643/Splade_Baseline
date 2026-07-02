# sandbox_04 — платформа экспериментов SPLADE

Реализация `ARCHITECTURE.md` (корень репозитория): замороженное ядро `core/`
(eval, метрики, статистика, раннер, очередь), свободно меняемый код эксперимента
`exp/` (SPLADE-max), снапшоты кода, gate перед очередью, worker-демон вместо
tmux+papermill, веб-UI и CLI `lab` с одинаковыми возможностями.

Два запуска сравнимы ⇔ совпадают `core_hash` и `eval_data_hash` —
`lab compare` и UI проверяют это сами и предупреждают.

## Установка (на сервере с GPU)

```bash
cd sandbox_04
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./lab test                      # тесты ядра: метрики, статистика, снапшоты, guards
```

Все команды ниже — из `sandbox_04` с активированным venv. `./lab` можно
заменить на `python lab`.

## Данные (один раз)

```bash
./lab data prepare --all        # ~1 час, ~12GB диска, ~4GB трафика
./lab data status               # что готово
./lab data verify               # sha256 + количества против manifest
./lab test                      # теперь пройдут и тесты целостности данных
```

Что готовится (всё замораживается в `data/eval/manifest.json`):

| что | размер | зачем |
|---|---|---|
| `msmarco-full` | 8.8M пассажей | финальные прогоны (дорого: энкод ~1ч/GPU) |
| `msmarco-lite` | 1M пассажей: все judged-документы dev+TREC + детерминированная выборка | ночные эксперименты (энкод ~10 мин) |
| `gate` | 1k доков / 50 запросов | микро-прогон перед очередью |
| `msmarco-dev` | 6980 запросов + qrels | MRR@10, nDCG@10 (co-primary) |
| `trec-dl-2019/2020` | 43/54 запроса, градуированные qrels | nDCG@10 (main), порог бинарных метрик rel≥2 |
| train pool | первые 2M триплетов `triples.train.small` | из него сэмплируются train-данные |

Абсолютные числа на lite-корпусе завышены относительно полного (меньше
дистракторов), но **сравнения между системами валидны** — у всех один и тот же
замороженный корпус. Для цифр «как в статье» — до-eval на полном корпусе
(`lab eval <run> --datasets msmarco-dev`), это не требует переобучения.

## Быстрая проверка, что всё живо (~10 мин)

```bash
./lab run configs/presets/smoke.yaml
./lab status
```

## Ночной эксперимент: данные MS MARCO и качество

**Вопрос**: при фиксированном компьюте (15k шагов × batch 32 = 480k показов
триплетов) как качество зависит от числа N *уникальных* триплетов?
N=30k — модель видит каждый триплет ~16 раз; N=1.92M — только четверть пула,
каждый один раз. Это изолирует эффект разнообразия данных от эффекта компьюта.

Дизайн: N ∈ {30k, 120k, 480k, 1.92M} (геометрическая сетка ×4, центр — 1 эпоха)
× сиды {1,2,3} = **12 запусков**. Сид меняет и подвыборку триплетов, и порядок,
и инициализацию головы — то есть дисперсия по сидам включает дисперсию выбора
данных, что и есть правильная модель шума для выводов о данных.

```bash
./lab sweep configs/data_scaling.yaml data.train_triples=30000,120000,480000,1920000
# gate прогонит первый вариант (~2 мин), затем 12 джобов встанут в очередь

./lab worker start --gpus 0,1   # демон; переживает закрытие ssh, tmux НЕ нужен
./lab worker status
./lab tail <run-id> -f          # живой лог любого запуска (Ctrl-C безопасен)
```

Бюджет: ~70 мин/запуск на A100 (train ~50 мин + энкод lite ~10 мин + поиск).
12 запусков ≈ 7 ч на 2 GPU, ≈ 14 ч на 1 GPU. Если GPU один — безопаснее 2 сида:
`./lab sweep configs/data_scaling.yaml data.train_triples=30000,120000,480000,1920000 --no-gate`
после правки `seeds: [1, 2]` в конфиге (gate уже пройден этим снапшотом).

Утром:

```bash
./lab status
./lab compare --exp data-scaling --out data-scaling-night1
# консоль: таблицы; reports/data-scaling-night1/: report.md, report.json, png-графики
```

Что будет в отчёте:

1. По каждому датасету: mean ± std по сидам для MRR@10, nDCG@10, R@10/100/1000.
2. Все пары N: Δ метрики, 95% BCa-CI (парный bootstrap по запросам, 10k),
   p рандомизационного теста Фишера (первичный) и парного t-теста (сверка),
   поправка Холма на все пары × метрики. Значимость — по Холм-скорректированному p.
3. Скейлинг: фит m(N) = a − b·N^(−γ) (a — оценка потолка качества на этих данных)
   и тест тренда Кендалла (τ, p). **Первичный вывод о монотонности — Кендалл**;
   γ на 4 точках слабо идентифицируем, фит — описательная кривая.
4. Разреженность (avg nnz doc/query) по N — в `metrics.json` каждого запуска:
   влияет ли объём данных на эффективный размер индекса при тех же λ.

## Worker: жизнь без tmux

`lab worker start` порождает процесс в новой сессии (`start_new_session`),
stdout — в `queue/worker.log`, pid — в `queue/worker.pid`. Закрытие терминала
и обрыв ssh на него не влияют.

```bash
./lab worker start --gpus 0,1   # или без --gpus: возьмёт все видимые CUDA
./lab worker status             # жив ли, очередь, что бежит
./lab worker stop               # останавливает демона И бегущие запуски (status=failed)
tail -f queue/worker.log
```

Каждый свободный GPU атомарно забирает следующий джоб из `queue/` и запускает
`core.runner` отдельным процессом с `CUDA_VISIBLE_DEVICES=<gpu>`. Падение
запуска не останавливает очередь. После рестарта демон возвращает осиротевшие
джобы в очередь и помечает мёртвые `running`-запуски как `failed`.

## UI

```bash
./lab ui --port 8000            # затем ssh -L 8000:localhost:8000 user@server
```

- **Runs** — все запуски, live-обновление каждые 5с, фильтр.
- **Run** — конфиг, кривая loss, метрики, хвост stdout, kill, дифф с любым запуском.
- **Queue** — переупорядочить/удалить, поставить новый (файл или yaml в textarea,
  gate в фоне), убить бегущий.
- **Compare** — выбрать ≥2 → полный статпротокол + скейлинг.
- **Snapshots** — список, дифф кода между любыми двумя.

UI ничего не хранит — источник правды только файлы (`runs/`, `queue/`,
`snapshots/`), поэтому CLI и UI взаимозаменяемы в любой момент.

## CLI шпаргалка

```bash
./lab run cfg.yaml                    # сейчас, в форграунде, из авто-снапшота
./lab queue cfg.yaml [--snap X] [--no-gate]
./lab sweep cfg.yaml k=v1,v2 ... [--gate-all]   # gate по умолч. только 1-й вариант
./lab worker start|stop|status [--gpus 0,1]
./lab status                          # очередь + таблица запусков
./lab tail <run> [-f] / ./lab kill <run>
./lab snap save <имя> -m "..." / ./lab snap list
./lab diff <a> <b>                    # run|snapshot: конфиг + unified diff кода
./lab compare <runs...> | --exp <имя> [--out dir]
./lab eval <run> --datasets msmarco-dev,trec-dl-2019   # до-eval без переобучения
./lab model name <run> <имя> -m "..."  # реестр весов (runs/models.json)
./lab data prepare|status|verify
./lab test
./lab ui [--port 8000]
```

## Папка запуска (самодостаточна)

```
runs/<id>/
  config.yaml        # полностью резолвленный конфиг (с seed)
  snapshot.json      # хэш+имя снапшота кода — код восстановим всегда
  status             # queued | running | done | failed
  meta.json          # core_hash, eval_data_hash, git sha, GPU, версии, длительность
  train_log.jsonl    # step, loss, компоненты, lr — построчно
  metrics.json       # агрегаты по датасетам + nnz + тайминги
  per_query.parquet  # dataset × qid × metric × value — основа всей статистики
  index/             # опц. (eval.save_index) — док-матрица CSR для до-eval
  model/             # HF-веса + токенизатор
  stdout.log, pid
```

## Как добавить свой эксперимент

Правишь только `exp/` и конфиг. Контракт (§6 архитектуры):

```python
# exp/train.py
def train(cfg: dict, ctx) -> Encoder      # полный цикл обучения, веса -> ctx.model_dir
def load(model_dir, cfg, device) -> Encoder   # для lab eval / переиспользования

# Encoder:
#   encode_queries(texts: list[str]) -> scipy.sparse.csr_matrix (n, vocab)
#   encode_docs(texts: list[str])   -> scipy.sparse.csr_matrix (n, vocab)
```

`ctx`: `.device`, `.model_dir`, `.cache_dir`, `.root`,
`.log(step, loss=..., ...)` (NaN/inf в log = немедленный fail — guard).
Eval после train делает только core — код эксперимента на замер не влияет.
Очередь исполняется из снапшота: после `lab queue` можно сразу править `exp/`
дальше, ночные запуски не затронет.

Новый лосс — функция в `exp/loss.py` (margin_mse для дистилляции уже лежит),
новая модель — класс в `exp/model.py`, wiring — в `exp/train.py`. Конфиг:
`extends: base.yaml` + переопределения. Дорогие артефакты (teacher-скоры) —
через `core.cache.get_or_compute` в `data/cache/`, переиспользуются всеми
запусками.

## Статистический протокол (замороженный, core/stats.py)

Вход — только `per_query.parquet`, всё считается за секунды на CPU:

1. per-query метрики усредняются по сидам системы (выравнивание по qid);
2. значимость: primary — парный рандомизационный тест Фишера
   (точный перебор 2^n при n≤20, иначе Monte-Carlo 10k), secondary — парный t-тест;
3. поправка Холма на все пары × метрики таблицы;
4. 95% CI разности — парный BCa-bootstrap по запросам, 10k итераций
   (сверен с `scipy.stats.bootstrap(method="BCa")` в тестах);
5. роли: msmarco-dev — MRR@10 + nDCG@10 co-primary; trec-dl — nDCG@10 main
   (бинарные метрики TREC считаются по порогу rel≥2 — стандарт TREC-DL).

Метрики сверены бит-в-бит (1e-9) с `ir_measures` на 100 случайных ранжированиях
(`core/tests/test_metrics.py`).

## Отклонения от ARCHITECTURE.md

- Fisher/t-test реализованы напрямую по векторам per-query (а не «через ranx»):
  ranx требует сырые ранжирования, а протокол §8 требует считать только из
  `per_query.parquet`. Взамен — точный перебор при n≤20 и тесты на эталонах.
- `lab sweep` гейтит по умолчанию только первый вариант (один снапшот кода);
  `--gate-all` возвращает поведение «gate на каждый джоб».

## Troubleshooting

- **gate ПРОВАЛ** — читай хвост в сообщении или `runs/_gate/<...>/stdout.log`.
  Очередь не тронута, ночь спасена.
- **CUDA OOM** — уменьшай `train.batch_size` (лосс InfoNCE зависит от батча —
  фиксируй одно значение на весь эксперимент) или `model.encode_batch_docs`.
- **worker умер** (проверь `./lab worker status`) — просто `./lab worker start`:
  осиротевшие джобы вернутся в очередь, мёртвые running помечаются failed.
- **`сравнение невалидно: разные eval_data_hash`** — запуски мерились на разных
  данных; пере-eval старый run: `./lab eval <run> --datasets <ds>`.
- **скачивание MS MARCO падает** — зеркала: замени в `core/data.py` хост
  `msmarco.z22.web.core.windows.net` на `msmarco.blob.core.windows.net`.
- **повторить упавший джоб** — он самодостаточен: `./lab queue <конфиг>` заново
  (или UI → Queue → textarea с конфигом из `runs/<id>/config.yaml`).
