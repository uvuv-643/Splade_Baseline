"""Файловый прогресс-логгер — замена tqdm без вывода в блокнот.

Зачем: tqdm рисует виджеты прямо в ноутбук, спамит и не показывает целиком.
Здесь — свой прогресс, который пишет в ОТДЕЛЬНЫЙ файл с timestamp в имени
(outputs/logs/progress_YYYYMMDD-HHMMSS.log), а в блокнот ничего не льёт.

Поведение:
- строка пишется при изменении доли на одну десятую процента (0.1%, 0.2%, ...);
- в каждой строке: процент, обработано/всего элементов, прошедшие секунды,
  эвристика «сколько осталось» (ETA) и скорость — как в tqdm;
- если общий объём неизвестен (стримы), пишем не чаще раза в секунду.

API совместим с tqdm в объёме, который использует проект:
    for x in tqdm(iterable, desc=..., unit=..., total=...): ...
    with tqdm(total=..., desc=...) as bar: bar.update(n)
    pbar = tqdm(range(n), desc=...); pbar.set_postfix(loss=...)
поэтому в модулях достаточно заменить импорт tqdm на этот.
"""
import threading
import time
from datetime import datetime
from pathlib import Path

from .config import ROOT

# Один файл логов на процесс (на сессию ядра). Имя — с timestamp.
_LOG_DIR = ROOT / "outputs" / "logs"
_log_path = None
_log_lock = threading.Lock()


def _get_log_path() -> Path:
    """Лениво создаёт (один раз) файл логов с timestamp в имени."""
    global _log_path
    if _log_path is None:
        with _log_lock:
            if _log_path is None:
                _LOG_DIR.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                _log_path = _LOG_DIR / f"progress_{ts}.log"
                # Единственная строка в блокнот — где смотреть прогресс.
                print(f"[progress] лог прогресса: {_log_path}")
    return _log_path


def log_path() -> Path:
    """Путь к текущему файлу логов прогресса (создаёт при первом вызове)."""
    return _get_log_path()


def _write(line: str):
    path = _get_log_path()
    with _log_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()


def _fmt_secs(s: float) -> str:
    """Секунды + человекочитаемое ЧЧ:ММ:СС для удобства."""
    s = max(0.0, s)
    total = int(s)
    h, rem = divmod(total, 3600)
    m, sec = divmod(rem, 60)
    hms = f"{h:d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"
    return f"{s:.1f}s ({hms})"


def _fmt_count(n: float, scale: bool) -> str:
    """Число элементов; при unit_scale — с суффиксами K/M/G (как tqdm)."""
    if not scale:
        return str(int(n))
    n = float(n)
    for suffix in ("", "K", "M", "G", "T"):
        if abs(n) < 1000.0:
            return f"{int(n)}" if suffix == "" else f"{n:.1f}{suffix}"
        n /= 1000.0
    return f"{n:.1f}P"


class FileTqdm:
    """Минималистичная tqdm-совместимая обёртка, пишущая в файл логов."""

    def __init__(self, iterable=None, total=None, desc=None, unit="it",
                 unit_scale=False, mininterval=1.0, **_ignored):
        self.iterable = iterable
        if total is None and iterable is not None:
            try:
                total = len(iterable)
            except (TypeError, AttributeError):
                total = None
        self.total = total
        self.desc = desc or "progress"
        self.unit = (unit or "it").strip()
        self.unit_scale = unit_scale
        self.mininterval = mininterval  # для неизвестного total
        self.n = 0
        self.postfix = ""
        self.start_t = time.time()
        self.last_print_t = self.start_t
        self.last_permille = -1
        self.closed = False
        # Стартовая отметка — видно, на каком этапе и когда начали.
        total_str = _fmt_count(self.total, self.unit_scale) if self.total else "?"
        _write(f"[{datetime.now():%H:%M:%S}] {self.desc}: старт | "
               f"всего={total_str} {self.unit}")

    # --- итерирование: for x in tqdm(iterable) ---
    def __iter__(self):
        for obj in self.iterable:
            yield obj
            self.update(1)
        self.close()

    # --- контекстный менеджер: with tqdm(...) as bar ---
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False

    # --- ручное обновление ---
    def update(self, n=1):
        self.n += n
        self._maybe_emit()

    def set_postfix(self, **kw):
        self.postfix = " | ".join(f"{k}={v}" for k, v in kw.items())

    def set_description(self, desc):
        self.desc = desc or self.desc

    # --- внутреннее ---
    def _maybe_emit(self):
        if self.total:
            # Печатаем при изменении доли на десятую процента (0.1%).
            permille = int(self.n / self.total * 1000)
            if permille != self.last_permille:
                self.last_permille = permille
                self._emit()
        else:
            now = time.time()
            if now - self.last_print_t >= self.mininterval:
                self._emit()

    def _emit(self, final=False):
        now = time.time()
        elapsed = now - self.start_t
        self.last_print_t = now
        rate = self.n / elapsed if elapsed > 0 else 0.0
        rate_str = f"{_fmt_count(rate, self.unit_scale)} {self.unit}/s"
        n_str = _fmt_count(self.n, self.unit_scale)
        stamp = f"[{datetime.now():%H:%M:%S}]"

        if self.total:
            pct = self.n / self.total * 100.0
            total_str = _fmt_count(self.total, self.unit_scale)
            remaining = (self.total - self.n) / rate if rate > 0 else 0.0
            line = (f"{stamp} {self.desc}: {pct:5.1f}% | "
                    f"{n_str}/{total_str} {self.unit} | "
                    f"прошло={_fmt_secs(elapsed)} | "
                    f"осталось~{_fmt_secs(remaining)} | {rate_str}")
        else:
            line = (f"{stamp} {self.desc}: {n_str} {self.unit} | "
                    f"прошло={_fmt_secs(elapsed)} | {rate_str}")

        if self.postfix:
            line += f" | {self.postfix}"
        if final:
            line += " | готово"
        _write(line)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self._emit(final=True)


# Имя tqdm — чтобы импорт был drop-in заменой во всех модулях.
tqdm = FileTqdm
