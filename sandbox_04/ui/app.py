import json
import threading
from pathlib import Path

import yaml
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from core import config as config_mod
from core import data, gate, queue as queue_mod, runs, snapshots, stats
from core.paths import CONFIGS_DIR, FAILED_JOBS_DIR

app = FastAPI(title="splade lab")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

GATING = {}


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


def _stdout_tail(run_dir, n=80) -> str:
    log = Path(run_dir) / "stdout.log"
    if not log.exists():
        return "(нет stdout.log)"
    return "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])


@app.get("/", response_class=HTMLResponse)
def index():
    return RedirectResponse("/runs")


@app.get("/runs", response_class=HTMLResponse)
def runs_page(request: Request, filter: str = ""):
    return templates.TemplateResponse(request, "runs.html",
                                      {"filter": filter})


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
    return templates.TemplateResponse(request, "run.html", {
        "info": info, "config": cfg_text, "metrics": metrics_js,
        "loss_svg": _loss_svg(d), "stdout": _stdout_tail(d), "others": others,
    })


@app.get("/runs/{run_id}/stdout", response_class=HTMLResponse)
def run_stdout(run_id: str):
    d = runs.resolve_run(run_id)
    return HTMLResponse(f"<pre id='stdout'>{_stdout_tail(d)}</pre>")


@app.post("/runs/{run_id}/kill")
def run_kill(run_id: str):
    runs.kill_run(run_id)
    return RedirectResponse(f"/runs/{run_id}", status_code=303)


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
    running = [r for r in runs.list_runs() if r["status"] == "running"]
    configs = sorted(str(p.relative_to(CONFIGS_DIR))
                     for p in CONFIGS_DIR.rglob("*.yaml"))
    failed = []
    if FAILED_JOBS_DIR.exists():
        for p in sorted(FAILED_JOBS_DIR.glob("*.log")):
            failed.append({"name": p.stem, "log": p.read_text(encoding="utf-8")[-2000:]})
    snaps = snapshots.list_snapshots()
    return templates.TemplateResponse(request, "queue.html", {
        "jobs": jobs, "running": running, "configs": configs,
        "gating": dict(GATING), "failed": failed, "snapshots": snaps,
    })


def _gate_and_enqueue(cfg, snap_hash, snap_name, do_gate):
    key = cfg["name"]
    try:
        if do_gate:
            GATING[key] = "gate: бежит"
            ok, gate_dir = gate.run_gate(cfg, snap_hash, snap_name)
            if not ok:
                GATING[key] = f"gate ПРОВАЛ: {gate_dir}/stdout.log"
                return
        for cfg_seed in config_mod.expand_seeds(cfg):
            run_id = runs.new_run_id(cfg_seed["name"], cfg_seed["train"]["seed"])
            run_dir = runs.reserve_run(run_id, cfg_seed, snap_hash, snap_name)
            queue_mod.enqueue_train(cfg_seed, snap_hash, snap_name, run_dir.name)
        GATING.pop(key, None)
    except Exception as e:
        GATING[key] = f"ошибка: {e}"


@app.post("/queue/new")
def queue_new(config_file: str = Form(""), config_yaml: str = Form(""),
              snapshot: str = Form(""), do_gate: bool = Form(False)):
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
    threading.Thread(target=_gate_and_enqueue,
                     args=(cfg, snap_hash, snap_name, do_gate), daemon=True).start()
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
