import json
import threading
from pathlib import Path

import yaml
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from core import config as config_mod
from core import data, gate, queue as queue_mod, runs, snapshots, stats, worker
from core.paths import CONFIGS_DIR, FAILED_JOBS_DIR

app = FastAPI(title="splade lab")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Фоновые статусы: gate/enqueue и подготовка данных — чтобы UI показывал прогресс
# долгих операций, не блокируя запрос.
GATING = {}
DATA_TASKS = {}


# --------------------------------------------------------------------------- #
#  вспомогательные рендеры
# --------------------------------------------------------------------------- #

def _loss_svg(run_dir, width=800, height=240) -> str:
    log_path = Path(run_dir) / "train_log.jsonl"
    if not log_path.exists():
        return ""
    points = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
            points.append((rec["step"], rec["loss"]))
        except (json.JSONDecodeError, KeyError):
            continue
    if len(points) < 2:
        return ""
    stride = max(1, len(points) // 500)
    points = points[::stride]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    pad = 30
    sx = lambda x: pad + (x - x0) / max(1, x1 - x0) * (width - 2 * pad)
    sy = lambda y: height - pad - (y - y0) / max(1e-12, y1 - y0) * (height - 2 * pad)
    poly = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
    return (
        f'<svg viewBox="0 0 {width} {height}" style="width:100%;max-width:{width}px;'
        f'background:#fafafa;border:1px solid #ddd">'
        f'<polyline points="{poly}" fill="none" stroke="#0074d9" stroke-width="1.5"/>'
        f'<text x="{pad}" y="14" font-size="11">loss: {y1:.3f} … {y0:.3f}</text>'
        f'<text x="{width - pad - 80}" y="{height - 8}" font-size="11">шаг {x1}</text>'
        f"</svg>")


def _tail(path: Path, n=80) -> str:
    if not path.exists():
        return f"(нет {path.name})"
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])


def _stdout_tail(run_dir, n=80) -> str:
    return _tail(Path(run_dir) / "stdout.log", n)


def _configs() -> list:
    return sorted(str(p.relative_to(CONFIGS_DIR)) for p in CONFIGS_DIR.rglob("*.yaml"))


def _active_jobs() -> list:
    """Джобы, реально выполняющиеся сейчас (файл в claimed/). Возвращаем в том же
    виде, что и очередь, плюс флаг live (жив ли процесс)."""
    out = []
    for p, job in queue_mod.list_claimed():
        kind = job.get("kind", "train")
        rid = job.get("run_id")
        run_dir = runs.run_dir(rid)
        if kind == "eval":
            live = runs.eval_pid(run_dir) is not None
        else:
            live = runs.get_status(run_dir) == "running"
        out.append({
            "file": p.name, "run_id": rid, "kind": kind,
            "detail": ",".join(job.get("datasets", [])) if kind == "eval"
                      else job.get("snapshot", {}).get("hash", ""),
            "datasets": job.get("datasets", []),
            "live": live,
        })
    return out


# --------------------------------------------------------------------------- #
#  Dashboard
# --------------------------------------------------------------------------- #

@app.get("/", response_class=HTMLResponse)
def index():
    return RedirectResponse("/dashboard")


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse(request, "dashboard.html", {})


@app.get("/dashboard/panel", response_class=HTMLResponse)
def dashboard_panel(request: Request):
    st = worker.worker_state()
    all_runs = runs.list_runs()
    counts = {}
    for r in all_runs:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    active = _active_jobs()
    recent = all_runs[:8]
    return templates.TemplateResponse(request, "dashboard_panel.html", {
        "state": st, "counts": counts, "active": active,
        "n_runs": len(all_runs), "recent": recent,
        "worker_log": worker.worker_log_tail(40),
    })


# --------------------------------------------------------------------------- #
#  Runs
# --------------------------------------------------------------------------- #

@app.get("/runs", response_class=HTMLResponse)
def runs_page(request: Request, filter: str = ""):
    return templates.TemplateResponse(request, "runs.html", {"filter": filter})


@app.get("/runs/table", response_class=HTMLResponse)
def runs_table(request: Request, filter: str = ""):
    items = runs.list_runs()
    if filter:
        items = [r for r in items if filter in r["id"]]
    return templates.TemplateResponse(request, "runs_table.html",
                                      {"runs": items, "filter": filter})


@app.get("/runs/{run_id}", response_class=HTMLResponse)
def run_page(request: Request, run_id: str):
    d = runs.resolve_run(run_id)
    info = runs.run_info(d)
    cfg_text = (d / "config.yaml").read_text(encoding="utf-8")
    metrics_js = {}
    if (d / "metrics.json").exists():
        metrics_js = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    others = [r["id"] for r in runs.list_runs() if r["id"] != d.name]
    all_datasets = sorted(data.DATASETS)
    return templates.TemplateResponse(request, "run.html", {
        "info": info, "config": cfg_text, "metrics": metrics_js,
        "loss_svg": _loss_svg(d), "stdout": _stdout_tail(d), "others": others,
        "eval_log": _tail(d / "eval_log.jsonl", 30) if (d / "eval_log.jsonl").exists() else "",
        "all_datasets": all_datasets,
    })


@app.get("/runs/{run_id}/stdout", response_class=HTMLResponse)
def run_stdout(run_id: str):
    d = runs.resolve_run(run_id)
    return HTMLResponse(f"<pre id='stdout'>{_stdout_tail(d)}</pre>")


@app.post("/runs/{run_id}/kill")
def run_kill(run_id: str):
    runs.kill_run(run_id)
    return RedirectResponse(f"/runs/{run_id}", status_code=303)


@app.post("/runs/{run_id}/kill_eval")
def run_kill_eval(run_id: str):
    runs.kill_eval(run_id)
    return RedirectResponse(f"/runs/{run_id}", status_code=303)


@app.post("/runs/{run_id}/delete")
def run_delete(run_id: str):
    try:
        runs.delete_run(run_id)
    except (RuntimeError, FileNotFoundError):
        return RedirectResponse(f"/runs/{run_id}", status_code=303)
    return RedirectResponse("/runs", status_code=303)


@app.post("/runs/{run_id}/requeue")
def run_requeue(run_id: str):
    """«Оживить» упавший запуск: снова поставить его train-джоб в очередь
    (тот же каталог/снапшот/конфиг). Заберёт worker при следующем поллинге."""
    try:
        runs.requeue_run(run_id)
    except (RuntimeError, FileNotFoundError):
        pass
    return RedirectResponse(f"/runs/{run_id}", status_code=303)


@app.post("/runs/{run_id}/eval")
def run_eval(run_id: str, datasets: list = Form(...),
             save_index: bool = Form(False), now: bool = Form(False)):
    d = runs.resolve_run(run_id)
    ds = [x for x in datasets if x]
    if not ds:
        return RedirectResponse(f"/runs/{run_id}", status_code=303)
    if now:
        from core import runner

        def _bg():
            try:
                runner.execute_eval(d, ds, save_index=save_index)
            except SystemExit:
                pass
        threading.Thread(target=_bg, daemon=True).start()
    else:
        queue_mod.enqueue_eval(d.name, ds, save_index=save_index)
    return RedirectResponse(f"/runs/{run_id}", status_code=303)


@app.post("/runs/{run_id}/name_model")
def run_name_model(run_id: str, model_name: str = Form(...), message: str = Form("")):
    try:
        runs.name_model(run_id, model_name, message)
    except FileNotFoundError:
        pass
    return RedirectResponse(f"/runs/{run_id}", status_code=303)


# --------------------------------------------------------------------------- #
#  Queue
# --------------------------------------------------------------------------- #

@app.get("/queue", response_class=HTMLResponse)
def queue_page(request: Request):
    jobs = []
    for p, job in queue_mod.list_jobs():
        kind = job.get("kind", "train")
        jobs.append({
            "file": p.name, "run_id": job["run_id"], "kind": kind,
            "snapshot": job.get("snapshot", {}).get("hash", "")
                        if kind == "train"
                        else ",".join(job.get("datasets", [])),
            "dep_state": queue_mod.dep_state(job),
            "created": job.get("created", ""),
        })
    active = _active_jobs()
    configs = _configs()
    failed = []
    if FAILED_JOBS_DIR.exists():
        for p in sorted(FAILED_JOBS_DIR.glob("*.log")):
            failed.append({"name": p.stem, "log": p.read_text(encoding="utf-8")[-2000:]})
    snaps = snapshots.list_snapshots()
    return templates.TemplateResponse(request, "queue.html", {
        "jobs": jobs, "active": active, "configs": configs,
        "gating": dict(GATING), "failed": failed, "snapshots": snaps,
        "worker": worker.worker_state(),
    })


def _gate_and_enqueue(cfg, snap_hash, snap_name, do_gate, sweep=None,
                      eval_datasets=None, save_index=False):
    key = cfg["name"]
    try:
        variants = config_mod.expand_sweep(cfg, sweep) if sweep else [cfg]
        for v in variants:
            config_mod.validate(v)
        if do_gate:
            GATING[key] = "gate: бежит"
            ok, gate_dir = gate.run_gate(variants[0], snap_hash, snap_name)
            if not ok:
                GATING[key] = f"gate ПРОВАЛ: {gate_dir}/stdout.log"
                return
        n = 0
        for v in variants:
            for cfg_seed in config_mod.expand_seeds(v):
                run_id = runs.new_run_id(cfg_seed["name"], cfg_seed["train"]["seed"])
                run_dir = runs.reserve_run(run_id, cfg_seed, snap_hash, snap_name)
                queue_mod.enqueue_train(cfg_seed, snap_hash, snap_name, run_dir.name)
                if eval_datasets:
                    queue_mod.enqueue_eval(run_dir.name, eval_datasets,
                                           depends_on=run_dir.name,
                                           save_index=save_index)
                n += 1
        GATING[key] = f"поставлено {n} джобов ✓"
    except Exception as e:
        GATING[key] = f"ошибка: {e}"


@app.post("/queue/new")
def queue_new(config_file: str = Form(""), config_yaml: str = Form(""),
              snapshot: str = Form(""), do_gate: bool = Form(False),
              sweep: str = Form(""), eval_datasets: str = Form(""),
              save_index: bool = Form(False)):
    if config_yaml.strip():
        cfg = yaml.safe_load(config_yaml)
        if "extends" in cfg:
            cfg = config_mod.deep_merge(
                config_mod.load_config(cfg.pop("extends")), cfg)
    else:
        cfg = config_mod.load_config(config_file)
    config_mod.validate(cfg)
    if snapshot:
        snap_hash = snapshots.resolve(snapshot)
        snap_name = snapshots.load_index()[snap_hash]["name"]
    else:
        snap_hash = snapshots.save(name=cfg["name"])
        snap_name = snapshots.load_index()[snap_hash]["name"]
    sweep_dict = config_mod.parse_sweep(sweep.split()) if sweep.strip() else None
    eval_ds = [d for d in eval_datasets.split(",") if d.strip()] or None
    threading.Thread(
        target=_gate_and_enqueue,
        args=(cfg, snap_hash, snap_name, do_gate, sweep_dict, eval_ds, save_index),
        daemon=True).start()
    GATING[cfg["name"]] = "поставлен: gate/enqueue в фоне"
    return RedirectResponse("/queue", status_code=303)


@app.post("/queue/{job_id}/delete")
def queue_delete(job_id: str):
    queue_mod.remove(job_id)
    return RedirectResponse("/queue", status_code=303)


@app.post("/queue/{job_id}/up")
def queue_up(job_id: str):
    queue_mod.move(job_id, -1)
    return RedirectResponse("/queue", status_code=303)


@app.post("/queue/{job_id}/down")
def queue_down(job_id: str):
    queue_mod.move(job_id, +1)
    return RedirectResponse("/queue", status_code=303)


# --------------------------------------------------------------------------- #
#  Worker
# --------------------------------------------------------------------------- #

@app.get("/worker", response_class=HTMLResponse)
def worker_page(request: Request):
    return templates.TemplateResponse(request, "worker.html", {})


@app.get("/worker/panel", response_class=HTMLResponse)
def worker_panel(request: Request):
    return templates.TemplateResponse(request, "worker_panel.html", {
        "state": worker.worker_state(),
        "log": worker.worker_log_tail(200),
    })


@app.post("/worker/start")
def worker_start(gpus: str = Form("")):
    worker.start_daemon(gpus or None)
    return RedirectResponse("/worker", status_code=303)


@app.post("/worker/stop")
def worker_stop():
    worker.stop_daemon()
    return RedirectResponse("/worker", status_code=303)


@app.get("/worker/badge", response_class=HTMLResponse)
def worker_badge():
    st = worker.worker_state()
    dot = "on" if st["running"] else "off"
    n_active = len(st["active"])
    txt = f"worker pid={st['pid']}" if st["running"] else "worker остановлен"
    return HTMLResponse(
        f'<span class="dot {dot}"></span>{txt} · очередь {st["n_jobs"]} · '
        f'активно {n_active}')


# --------------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------------- #

@app.get("/data", response_class=HTMLResponse)
def data_page(request: Request):
    st = data.status()
    return templates.TemplateResponse(request, "data.html", {
        "status": st, "parts": list(data.PREPARE_PARTS),
        "tasks": dict(DATA_TASKS),
    })


def _prepare_bg(parts, force):
    key = ",".join(parts) if parts else "all"
    try:
        DATA_TASKS[key] = "готовится…"
        if parts:
            for part in parts:
                data.PREPARE_PARTS[part](force=force)
            data.write_manifest()
        else:
            data.prepare_all(force=force)
        DATA_TASKS[key] = "готово ✓"
    except Exception as e:
        DATA_TASKS[key] = f"ошибка: {e}"


@app.post("/data/prepare")
def data_prepare(parts: str = Form(""), force: bool = Form(False)):
    part_list = [p for p in parts.split(",") if p.strip()]
    threading.Thread(target=_prepare_bg, args=(part_list, force), daemon=True).start()
    return RedirectResponse("/data", status_code=303)


@app.post("/data/verify")
def data_verify():
    errors = data.verify_manifest()
    DATA_TASKS["verify"] = "целостность ок ✓" if not errors else " / ".join(errors[:5])
    return RedirectResponse("/data", status_code=303)


# --------------------------------------------------------------------------- #
#  Models
# --------------------------------------------------------------------------- #

@app.get("/models", response_class=HTMLResponse)
def models_page(request: Request):
    registry = runs.list_models()
    return templates.TemplateResponse(request, "models.html", {"models": registry})


# --------------------------------------------------------------------------- #
#  Compare
# --------------------------------------------------------------------------- #

@app.get("/compare", response_class=HTMLResponse)
def compare_page(request: Request):
    run_ids = request.query_params.getlist("run")
    done = [r for r in runs.list_runs() if r["status"] == "done"]
    report = None
    error = None
    if len(run_ids) >= 2:
        systems = {}
        for rid in run_ids:
            d = runs.resolve_run(rid)
            cfg = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))
            systems.setdefault(cfg["name"], []).append(d)
        try:
            report = stats.compare_systems(systems)
        except Exception as e:
            error = str(e)
    return templates.TemplateResponse(request, "compare.html", {
        "done": done, "selected": set(run_ids), "report": report, "error": error,
        "primary": {ds: set(stats.primary_metrics(ds))
                    for ds in (report or {}).get("datasets", {})},
    })


# --------------------------------------------------------------------------- #
#  Snapshots / diff
# --------------------------------------------------------------------------- #

@app.get("/snapshots", response_class=HTMLResponse)
def snapshots_page(request: Request, a: str = "", b: str = ""):
    snaps = snapshots.list_snapshots()
    diff = None
    if a and b:
        diff = snapshots.diff_snapshots(snapshots.resolve(a), snapshots.resolve(b))
    return templates.TemplateResponse(request, "snapshots.html",
                                      {"snapshots": snaps, "diff": diff, "a": a, "b": b})


@app.get("/diff", response_class=HTMLResponse)
def diff_page(request: Request, a: str, b: str):
    da, db = runs.resolve_run(a), runs.resolve_run(b)
    import difflib
    cfg_a = (da / "config.yaml").read_text(encoding="utf-8")
    cfg_b = (db / "config.yaml").read_text(encoding="utf-8")
    cfg_diff = "".join(difflib.unified_diff(
        cfg_a.splitlines(keepends=True), cfg_b.splitlines(keepends=True),
        fromfile=da.name, tofile=db.name)) or "(конфиги идентичны)"
    snap_a = json.loads((da / "snapshot.json").read_text(encoding="utf-8"))["hash"]
    snap_b = json.loads((db / "snapshot.json").read_text(encoding="utf-8"))["hash"]
    code_diff = (f"(один снапшот {snap_a})" if snap_a == snap_b
                 else snapshots.diff_snapshots(snap_a, snap_b))
    return templates.TemplateResponse(request, "diff.html", {
        "a": da.name, "b": db.name, "cfg_diff": cfg_diff, "code_diff": code_diff,
    })
