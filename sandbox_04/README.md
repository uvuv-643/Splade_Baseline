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
./lab data prepare --all        # ~1 час, ~12GB диска, ~4GB трафика (без BEIR)
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

### BEIR zero-shot наборы (13 строк таблицы статьи)

Отдельно от `--all` (тяжёлые, ставятся осознанно) — 13 стандартных BEIR-наборов,
ровно те, что в таблице исходной статьи. Каждый самодостаточен (свой корпус +
запросы + qrels), главная метрика — **nDCG@10**.

```bash
./lab data prepare --part beir          # все 13 наборов (~15-20GB диска)
./lab data prepare --part beir-small    # только лёгкие (для smoke, ~2GB)
./lab data prepare --part beir-scifact  # один набор точечно
./lab data verify                       # проверит sha256 всех скачанных, в т.ч. BEIR
```

| наш ключ | строка статьи | archive (BEIR) | корпус |
|---|---|---|---|
| `beir-arguana` | ArguAna | arguana | ~8.7K (small) |
| `beir-climate-fever` | Climate-FEVER | climate-fever | 5.4M (large) |
| `beir-dbpedia` | DBPedia | dbpedia-entity | 4.6M (large) |
| `beir-fever` | FEVER | fever | 5.4M (large) |
| `beir-fiqa` | FiQA-2018 | fiqa | ~57K (small) |
| `beir-hotpotqa` | HotpotQA | hotpotqa | 5.2M (large) |
| `beir-nfcorpus` | NFCorpus | nfcorpus | ~3.6K (small) |
| `beir-nq` | NQ | nq | 2.7M (large) |
| `beir-quora` | Quora | quora | ~523K (small) |
| `beir-scidocs` | SCIDOCS | scidocs | ~25K (small) |
| `beir-scifact` | SciFact | scifact | ~5.2K (small) |
| `beir-trec-covid` | TREC-COVID | trec-covid | ~171K (small) |
| `beir-touche2020` | Touché-2020 | webis-touche2020 | ~382K (small) |

`small`/`large` — эвристика по размеру корпуса: `beir-small` (8 лёгких) кодируются
за минуты и входят в smoke; крупные проверяются целостностью и гоняются полным
прогоном / до-eval (`lab eval <run> --datasets beir-fever,...`) на обученной модели.

### Smoke на BEIR (проверка, что данные скачались)

```bash
./lab data prepare --part beir-small,train-pool   # если ещё не готовили
./lab run configs/presets/smoke_beir.yaml         # мини-train + zero-shot eval на 8 лёгких BEIR
./lab status                                       # nDCG@10 по каждому набору
```

Абсолютные числа на lite-корпусе завышены относительно полного (меньше
дистракторов), но **сравнения между системами валидны** — у всех один и тот же
замороженный корпус. Для цифр «как в статье» — до-eval на полном корпусе
(`lab eval <run> --datasets msmarco-dev`), это не требует переобучения.

## Быстрая проверка, что всё живо (~10 мин)

```bash
./lab run configs/presets/smoke.yaml
./lab status
```

## Всё через очередь на 1 GPU (целевой сценарий)

Идея: **ничего не считается в терминале**. Вы только ставите задачи в очередь и
запускаете один worker-демон на вашем единственном GPU — он по порядку сам
обучает модели, строит индексы, считает eval и сохраняет метрики. Терминал можно
закрыть, ssh оборвать — worker живёт (`start_new_session`).

Два вида джобов в очереди:

- **train** — полный цикл `train → построение индекса → eval` из снапшота кода
  (это уже одна атомарная джоба; метрики и `per_query.parquet` пишутся в конце);
- **eval** — до-eval уже обученной модели на новых/полных датасетах: строит
  индекс на их корпусе и дописывает метрики в тот же `runs/<id>/`. Может
  **зависеть** от train-джоба (`depends_on`) — тогда worker не возьмёт eval,
  пока train не завершится со статусом `done`. Так получается цепочка
  train→index→eval, исполняемая сама.

Worker на 1 GPU берёт задачи строго по одной: пока идёт train, зависимый eval
«ждёт», независимые джобы очереди — нет (worker их пропускает вперёд не будет —
берёт первый *готовый*).

### 6 команд по порядку (пример: обучить и посчитать полный eval)

Разберём ровно 6 команд, которые ставятся в очередь и постепенно исполняются на
одном GPU. Здесь: обучаем модель, а тяжёлый eval на **полном** корпусе
`msmarco-dev` идёт отдельной джобой сразу после обучения (цепочка).

```bash
# 0. один раз: поднять worker-демон на вашем единственном GPU (номер 0)
./lab worker start --gpus 0

# 1. поставить обучение + следом (авто-зависимость) eval на полном msmarco-dev.
#    train сам сделает быстрый lite-eval (base.yaml), а эта eval-джоба добавит
#    цифры «как в статье» на полном корпусе. gate прогонится перед постановкой.
./lab queue configs/base.yaml --eval-datasets msmarco-dev,trec-dl-2019,trec-dl-2020 --save-index

# 2. ещё один вариант модели (например, другой lambda) — тоже train+eval цепочкой
./lab queue configs/base.yaml --snap <снапшот_из_шага_1> --no-gate \
      --eval-datasets msmarco-dev --save-index

# 3. посмотреть, что стоит в очереди и что уже считается
./lab status

# 4. живой лог текущего запуска (Ctrl-C не трогает сам запуск)
./lab tail <run-id> -f

# 5. когда всё посчиталось — статистика и отчёт по завершённым запускам
./lab compare --exp base --out night1
```

Что произойдёт: после шага 1 в очереди появятся `train`-джоб и зависимый
`eval`-джоб; worker возьмёт train, обучит, сохранит модель + lite-метрики, затем
возьмёт eval (он «разморозится», как только train станет `done`), построит
полный индекс msmarco-full и допишет метрики в тот же `runs/<id>/metrics.json`.
Всё это — без единой блокирующей команды в терминале.

**Вариант «6 отдельных обучений»**: если 6 команд — это просто 6 разных моделей
(train+eval уже внутри одной train-джобы, полный корпус не нужен), то шесть раз
`./lab queue <config>` и один `./lab worker start --gpus 0` — worker прогонит их
по очереди:

```bash
./lab worker start --gpus 0
./lab queue configs/base.yaml                        # 1-я модель (с gate)
./lab queue configs/exp2.yaml --snap <snap> --no-gate  # 2-я
./lab queue configs/exp3.yaml --snap <snap> --no-gate  # 3-я
# ... и т.д. до шести; затем:
./lab status
./lab compare --exp base --out night1
```

**До-eval уже обученной модели** (без переобучения) тоже уходит в очередь, а не
блокирует терминал:

```bash
./lab eval <run-id> --datasets msmarco-dev --save-index   # ставит eval-джоб
./lab eval <run-id> --datasets beir-fever --now           # ...или считать сейчас (блокирует)
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

## Memory watchdog (memwatch)

Построение CSR-индекса при недостаточной разреженности съедает всю RAM и
раньше валило ядро/OOM-killer. Теперь у каждого эксперимента два уровня защиты:

- **in-process** (`core/memwatch.py`, поток внутри `core.runner`): каждые 2с
  мерит эффективную память (host по MemAvailable **и** cgroup-лимит контейнера —
  берётся более ограниченный), RSS дерева процесса, пишет `runs/<id>/memory.json`
  (текущее, для UI) и `memory.jsonl` (таймлайн). При ≥80% — warning в stdout и UI;
  при ≥90% — пишет `oom_kill.json` (полный отчёт: проценты, RSS, avg_nnz,
  прогноз размера индекса, последние сэмплы) и **аккуратно** гасит процесс:
  SIGTERM (runner ловит его, дописывает traceback/meta/status) → grace 15с →
  SIGKILL группы. Отчёт пишется ДО сигнала — переживает даже SIGKILL.
- **worker-страховка**: если процесс застрял в нативном коде и свой watchdog
  не сработал, worker при ≥90% три полла подряд убивает джоб с максимальным
  RSS тем же способом.

Во время энкода корпуса в stdout и UI каждые 32k доков: прогресс, **средний
nnz вектора** и прогноз размера полного индекса — недостаточную разреженность
видно за минуты, а не при падении.

Пороги через окружение: `LAB_MEM_WARN_PCT=80`, `LAB_MEM_KILL_PCT=90`,
`LAB_MEM_GRACE_S=15` (worker и runner читают при старте).

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
./lab queue cfg.yaml [--snap X] [--no-gate] [--eval-datasets ds1,ds2] [--save-index]
./lab sweep cfg.yaml k=v1,v2 ... [--gate-all] [--eval-datasets ds1,ds2] [--save-index]
#   --eval-datasets: после train автоматически ставится зависимый eval-джоб
#                    (цепочка train→index→eval исполняется сама на 1 GPU)
./lab worker start|stop|status [--gpus 0]   # на 1 GPU: --gpus 0
./lab status                          # очередь (train/eval, кто ждёт) + запуски
./lab tail <run> [-f] / ./lab kill <run>
./lab snap save <имя> -m "..." / ./lab snap list
./lab diff <a> <b>                    # run|snapshot: конфиг + unified diff кода
./lab compare <runs...> | --exp <имя> [--out dir]
./lab eval <run> --datasets msmarco-dev,trec-dl-2019 [--save-index]  # до-eval В ОЧЕРЕДЬ
./lab eval <run> --datasets msmarco-dev --now          # ...или считать сейчас (блокирует)
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
  metrics.json       # агрегаты по датасетам + nnz + тайминги (train + до-eval)
  per_query.parquet  # dataset × qid × metric × value — основа всей статистики
  eval_log.jsonl     # опц. — прогресс/итог eval-джобов (до-eval из очереди)
  index/             # опц. (--save-index) — док-матрица CSR для до-eval
  model/             # HF-веса + токенизатор
  memory.json        # текущее состояние памяти (пишет memwatch, читает UI)
  memory.jsonl       # таймлайн памяти (график в UI, post-mortem)
  oom_kill.json      # опц. — детальный отчёт, если запуск убит по памяти
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
- **запуск убит по памяти (⛔ OOM в UI)** — читай `runs/<id>/oom_kill.json` и
  хвост `stdout.log`: там avg_nnz и прогноз размера индекса на момент смерти.
  Обычно виновата недостаточная разреженность — поднимай flops-регуляризацию
  (λ) или уменьшай корпус (msmarco-lite). Пороги: `LAB_MEM_WARN_PCT/KILL_PCT`.
- **worker умер** (проверь `./lab worker status`) — просто `./lab worker start`:
  осиротевшие джобы вернутся в очередь, мёртвые running помечаются failed.
  eval-джобы идемпотентны, поэтому при рестарте прерванный eval тоже вернётся
  в очередь и досчитается.
- **`сравнение невалидно: разные eval_data_hash`** — запуски мерились на разных
  данных; пере-eval старый run в очередь: `./lab eval <run> --datasets <ds>`,
  worker досчитает на GPU (не блокирует терминал).
- **eval-джоб упал** — сам train-запуск остаётся `done` (eval не трогает его
  статус); лог в `runs/<id>/stdout.log` и `runs/<id>/eval_log.jsonl`, сам джоб
  уезжает в `queue/failed/`. Повторить: `./lab eval <run> --datasets <ds>`.
- **eval не стартует, висит «ждёт train»** — это нормально: зависимый eval ждёт,
  пока его train-джоб не станет `done`. `./lab status` покажет, кто кого ждёт.
- **скачивание MS MARCO падает** — зеркала: замени в `core/data.py` хост
  `msmarco.z22.web.core.windows.net` на `msmarco.blob.core.windows.net`.
- **повторить упавший джоб** — он самодостаточен: `./lab queue <конфиг>` заново
  (или UI → Queue → textarea с конфигом из `runs/<id>/config.yaml`).
