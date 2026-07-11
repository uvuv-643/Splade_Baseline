import argparse
import difflib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from . import config as config_mod
from . import data, gate, queue as queue_mod, runs, snapshots, worker
from .paths import REPORTS_DIR, ROOT


def _load_validated(cfg_path: str) -> dict:
    cfg = config_mod.load_config(cfg_path)
    config_mod.validate(cfg)
    return cfg


def _check_data_ready(cfg: dict):
    missing = data.datasets_prepared(cfg["eval"]["datasets"])
    if missing:
        sys.exit(f"eval-данные не готовы: {missing}\nзапустите: ./lab data prepare --all")
    pool = ROOT / cfg["data"]["train_pool"]
    if not pool.exists():
        sys.exit(f"нет train-пула {pool}\nзапустите: ./lab data prepare --all")


def _snapshot_for(cfg: dict, snap_ref=None) -> tuple:
    if snap_ref:
        h = snapshots.resolve(snap_ref)
        return h, snapshots.load_index()[h]["name"]
    h = snapshots.save(name=cfg["name"])
    return h, snapshots.load_index()[h]["name"]


def _gate_or_die(cfg: dict, snap_hash: str, snap_name: str):
    print(f"[gate] микро-прогон снапшота {snap_hash} (~1-2 мин)...")
    t0 = time.time()
    ok, gate_dir = gate.run_gate(cfg, snap_hash, snap_name)
    dt = round(time.time() - t0)
    if not ok:
        tail = (gate_dir / "stdout.log")
        log_tail = "\n".join(tail.read_text(encoding="utf-8").splitlines()[-25:]) \
            if tail.exists() else "(нет лога)"
        sys.exit(f"[gate] ПРОВАЛ за {dt}s, лог {gate_dir}/stdout.log:\n{log_tail}")
    print(f"[gate] ок за {dt}s ({gate_dir.name})")


def _validate_datasets(spec: str) -> list:
    datasets = [d for d in spec.split(",") if d]
    unknown = [x for x in datasets if x not in data.DATASETS]
    if unknown:
        sys.exit(f"неизвестные датасеты: {unknown}")
    missing = data.datasets_prepared(datasets)
    if missing:
        sys.exit(f"eval-данные не готовы: {missing}\n"
                 f"запустите: ./lab data prepare ...")
    return datasets


def _enqueue_all(variants: list, snap_hash: str, snap_name: str,
                 eval_datasets=None, save_index=False) -> list:
    """Ставит train-джобы (по варианту × сиду). Каталог запуска резервируется
    сразу (runs.reserve_run) — запуск виден в `lab status`/UI как 'queued', и на
    него можно навесить зависимый eval-джоб (цепочка train→eval). Возвращает
    список фактических run_id."""
    run_ids = []
    for cfg in variants:
        for cfg_seed in config_mod.expand_seeds(cfg):
            run_id = runs.new_run_id(cfg_seed["name"], cfg_seed["train"]["seed"])
            run_dir = runs.reserve_run(run_id, cfg_seed, snap_hash, snap_name)
            queue_mod.enqueue_train(cfg_seed, snap_hash, snap_name, run_dir.name)
            print(f"[queue] + train {run_dir.name}")
            run_ids.append(run_dir.name)
            if eval_datasets:
                queue_mod.enqueue_eval(run_dir.name, eval_datasets,
                                       depends_on=run_dir.name,
                                       save_index=save_index)
                print(f"[queue]   ↳ eval {run_dir.name} "
                      f"(после train): {eval_datasets}")
    return run_ids


def cmd_run(args):
    cfg = _load_validated(args.config)
    _check_data_ready(cfg)
    snap_hash, snap_name = _snapshot_for(cfg, args.snap)
    from . import runner
    for cfg_seed in config_mod.expand_seeds(cfg):
        run_id = runs.new_run_id(cfg_seed["name"], cfg_seed["train"]["seed"])
        run_dir = runs.reserve_run(run_id, cfg_seed, snap_hash, snap_name)
        print(f"[run] {run_dir.name} (форграунд)")
        code = runner.run_foreground(run_dir)
        if code != 0:
            sys.exit(f"[run] {run_dir.name} упал (exit={code})")


def cmd_queue(args):
    cfg = _load_validated(args.config)
    _check_data_ready(cfg)
    eval_datasets = _validate_datasets(args.eval_datasets) if args.eval_datasets else None
    snap_hash, snap_name = _snapshot_for(cfg, args.snap)
    if not args.no_gate:
        _gate_or_die(cfg, snap_hash, snap_name)
    _enqueue_all([cfg], snap_hash, snap_name,
                 eval_datasets=eval_datasets, save_index=args.save_index)


def cmd_sweep(args):
    cfg = _load_validated(args.config)
    _check_data_ready(cfg)
    eval_datasets = _validate_datasets(args.eval_datasets) if args.eval_datasets else None
    sweep = config_mod.parse_sweep(args.overrides)
    variants = config_mod.expand_sweep(cfg, sweep)
    for v in variants:
        config_mod.validate(v)
    snap_hash, snap_name = _snapshot_for(cfg, args.snap)
    if not args.no_gate:
        to_gate = variants if args.gate_all else variants[:1]
        for v in to_gate:
            _gate_or_die(v, snap_hash, snap_name)
    _enqueue_all(variants, snap_hash, snap_name,
                 eval_datasets=eval_datasets, save_index=args.save_index)
    n_jobs = sum(len(config_mod.expand_seeds(v)) for v in variants)
    tail = f" (+{n_jobs} eval-джобов)" if eval_datasets else ""
    print(f"[sweep] {len(variants)} вариантов × сиды = {n_jobs} джобов в очереди{tail}")


def cmd_worker(args):
    if args.mem_gb:
        os.environ["LAB_MEM_TOTAL_GB"] = str(args.mem_gb)
    if args.action == "start":
        worker.start_daemon(args.gpus)
    elif args.action == "stop":
        worker.stop_daemon()
    elif args.action == "status":
        worker.daemon_status()
    elif args.action == "run":
        worker.worker_loop(worker.detect_gpus(args.gpus))


def cmd_status(args):
    jobs = queue_mod.list_jobs()
    if jobs:
        print(f"=== очередь ({len(jobs)}) ===")
        for i, (_, job) in enumerate(jobs, 1):
            kind = job.get("kind", "train")
            state = queue_mod.dep_state(job)
            note = {"waiting": " (ждёт train)", "dep_failed": " (train упал!)"}.get(state, "")
            if kind == "eval":
                detail = f"eval {','.join(job.get('datasets', []))}"
            else:
                detail = f"train snap={job['snapshot']['hash']}"
            print(f"{i:3d}. [{kind:5}] {job['run_id']}  {detail}{note}")
    all_runs = runs.list_runs()
    if not all_runs:
        print("запусков нет")
        return
    print(f"=== запуски ({len(all_runs)}) ===")
    fmt = "{:<44} {:<8} {:<13} {:>8}  {}"
    print(fmt.format("run", "status", "snapshot", "мин", "метрики"))
    for r in all_runs:
        dur = f"{r['duration_s'] / 60:.0f}" if r["duration_s"] else ""
        m = " ".join(f"{k}={v:.4f}" for k, v in sorted(r["metrics"].items())[:2])
        print(fmt.format(r["id"][:44], r["status"] or "?", r["snapshot"][:13], dur, m))


def cmd_tail(args):
    d = runs.resolve_run(args.run)
    log = d / "stdout.log"
    if not log.exists():
        sys.exit(f"нет {log}")
    lines = log.read_text(encoding="utf-8").splitlines()
    print("\n".join(lines[-args.n:]))
    if args.follow:
        with open(log, encoding="utf-8") as f:
            f.seek(0, 2)
            try:
                while True:
                    line = f.readline()
                    if line:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                    else:
                        time.sleep(1)
            except KeyboardInterrupt:
                pass


def cmd_kill(args):
    if runs.kill_run(args.run):
        print(f"{args.run}: остановлен, status=failed")
    else:
        print(f"{args.run}: не бежит (нечего останавливать)")


def cmd_snap(args):
    if args.action == "save":
        h = snapshots.save(name=args.name, message=args.message or "")
        print(f"снапшот {h} ({args.name})")
    elif args.action == "list":
        for e in snapshots.list_snapshots():
            gate_mark = {True: "gate✓", False: "gate✗", None: ""}[e.get("gate_ok")]
            print(f"{e['hash']}  {e['name']:<28} {e['created'][:19]}  {gate_mark}  {e['description']}")


def _run_or_snapshot(ref: str):
    try:
        d = runs.resolve_run(ref)
        snap = json.loads((d / "snapshot.json").read_text(encoding="utf-8"))
        cfg_text = (d / "config.yaml").read_text(encoding="utf-8")
        return {"kind": "run", "label": d.name, "snap": snap["hash"], "config": cfg_text}
    except FileNotFoundError:
        h = snapshots.resolve(ref)
        return {"kind": "snapshot", "label": h, "snap": h, "config": None}


def cmd_diff(args):
    a, b = _run_or_snapshot(args.a), _run_or_snapshot(args.b)
    if a["config"] and b["config"]:
        cfg_diff = list(difflib.unified_diff(
            a["config"].splitlines(keepends=True), b["config"].splitlines(keepends=True),
            fromfile=f"{a['label']}/config.yaml", tofile=f"{b['label']}/config.yaml"))
        print("=== конфиг ===")
        print("".join(cfg_diff) if cfg_diff else "(конфиги идентичны)")
    print("=== код (снапшоты) ===")
    if a["snap"] == b["snap"]:
        print(f"(один снапшот {a['snap']})")
    else:
        print(snapshots.diff_snapshots(a["snap"], b["snap"]))


def _group_systems(run_dirs: list) -> dict:
    systems = {}
    for d in run_dirs:
        cfg = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))
        systems.setdefault(cfg["name"], []).append(d)
    return systems


def cmd_compare(args):
    from . import stats
    if args.exp:
        candidates = [r for r in runs.list_runs()
                      if r["status"] == "done" and
                      (r["name"] == args.exp or r["name"].startswith(args.exp + "@"))]
        run_dirs = [r["dir"] for r in candidates]
        if not run_dirs:
            sys.exit(f"нет завершённых запусков эксперимента {args.exp!r}")
    else:
        run_dirs = [runs.resolve_run(ref) for ref in args.runs]
    systems = _group_systems(run_dirs)
    dsets = args.datasets.split(",") if args.datasets else None
    report = stats.compare_systems(systems, datasets=dsets)
    md = stats.render_markdown(report)
    print(md)
    if args.out:
        out = Path(args.out) if Path(args.out).is_absolute() else REPORTS_DIR / args.out
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.md").write_text(md, encoding="utf-8")
        (out / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        plots = stats.save_scaling_plots(report, out)
        print(f"\nотчёт: {out}/report.md" +
              (f", графики: {len(plots)} png" if plots else ""))


def cmd_eval(args):
    """До-eval готового запуска на новых датасетах (построение индекса + метрики).
    По умолчанию ставится в очередь (не блокирует терминал) — worker посчитает
    на GPU по очереди. С --now считает прямо сейчас в этом процессе."""
    d = runs.resolve_run(args.run)
    datasets = _validate_datasets(args.datasets)
    if args.now:
        from . import runner
        runner.execute_eval(d, datasets, save_index=args.save_index)
    else:
        queue_mod.enqueue_eval(d.name, datasets, save_index=args.save_index)
        print(f"[queue] + eval {d.name}: {datasets}\n"
              f"        запустите worker: ./lab worker start --gpus 0")


def cmd_model(args):
    if args.action == "name":
        runs.name_model(args.run, args.model_name, args.message or "")
        print(f"модель {args.model_name} -> {args.run}")
    elif args.action == "list":
        for name, e in runs.list_models().items():
            print(f"{name:<28} {e['run_id']}  {e['description']}")


def cmd_data(args):
    if args.action == "prepare":
        if args.part and not args.all:
            for part in args.part.split(","):
                if part not in data.PREPARE_PARTS:
                    sys.exit(f"неизвестная часть {part!r} "
                             f"(есть: {list(data.PREPARE_PARTS)})")
                data.PREPARE_PARTS[part](force=args.force)
            data.write_manifest()
        else:
            data.prepare_all(force=args.force)
    elif args.action == "status":
        st = data.status()
        print(f"manifest: {'есть' if st['manifest'] else 'НЕТ'}")
        print(f"train pool: {'есть' if st['train_pool'] else 'НЕТ'}")
        for name, ok in st["datasets"].items():
            print(f"{name:<20} {'готов' if ok else 'НЕТ'}")
    elif args.action == "verify":
        errors = data.verify_manifest()
        if errors:
            sys.exit("\n".join(errors))
        print("eval-данные целы (sha256 + количества совпадают с manifest)")


def cmd_test(args):
    code = subprocess.run([sys.executable, "-m", "pytest", "core/tests", "-q"],
                          cwd=ROOT).returncode
    sys.exit(code)


def cmd_ui(args):
    import uvicorn
    sys.path.insert(0, str(ROOT))
    uvicorn.run("ui.app:app", host=args.host, port=args.port, log_level="warning")


def build_parser():
    p = argparse.ArgumentParser(prog="lab", description="платформа экспериментов SPLADE")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("run", help="запуск сейчас, в форграунде")
    sp.add_argument("config")
    sp.add_argument("--snap")
    sp.set_defaults(fn=cmd_run)

    sp = sub.add_parser("queue", help="gate + в очередь")
    sp.add_argument("config")
    sp.add_argument("--snap")
    sp.add_argument("--no-gate", action="store_true")
    sp.add_argument("--eval-datasets",
                    help="после train поставить eval-джоб на этих датасетах "
                         "(цепочка train→eval), напр. msmarco-dev,trec-dl-2019")
    sp.add_argument("--save-index", action="store_true",
                    help="сохранять построенный индекс в runs/<id>/index/")
    sp.set_defaults(fn=cmd_queue)

    sp = sub.add_parser("sweep", help="сетка: lab sweep cfg.yaml key=v1,v2 ...")
    sp.add_argument("config")
    sp.add_argument("overrides", nargs="+")
    sp.add_argument("--snap")
    sp.add_argument("--no-gate", action="store_true")
    sp.add_argument("--gate-all", action="store_true")
    sp.add_argument("--eval-datasets",
                    help="после каждого train поставить eval-джоб на этих датасетах")
    sp.add_argument("--save-index", action="store_true")
    sp.set_defaults(fn=cmd_sweep)

    sp = sub.add_parser("worker", help="демон очереди")
    sp.add_argument("action", choices=["start", "stop", "status", "run"])
    sp.add_argument("--gpus")
    sp.add_argument("--mem-gb", type=float,
                    help="сколько GB оперативки выделено вам на общей машине; "
                         "лимитом станет суммарный RSS ваших процессов")
    sp.set_defaults(fn=cmd_worker)

    sp = sub.add_parser("status", help="очередь + все запуски")
    sp.set_defaults(fn=cmd_status)

    sp = sub.add_parser("tail", help="хвост stdout.log запуска")
    sp.add_argument("run")
    sp.add_argument("-n", type=int, default=50)
    sp.add_argument("-f", "--follow", action="store_true")
    sp.set_defaults(fn=cmd_tail)

    sp = sub.add_parser("kill", help="остановить бегущий запуск")
    sp.add_argument("run")
    sp.set_defaults(fn=cmd_kill)

    sp = sub.add_parser("snap", help="снапшоты кода exp/")
    snap_sub = sp.add_subparsers(dest="action", required=True)
    s = snap_sub.add_parser("save")
    s.add_argument("name")
    s.add_argument("-m", "--message")
    s.set_defaults(fn=cmd_snap)
    s = snap_sub.add_parser("list")
    s.set_defaults(fn=cmd_snap)

    sp = sub.add_parser("diff", help="дифф двух запусков/снапшотов")
    sp.add_argument("a")
    sp.add_argument("b")
    sp.set_defaults(fn=cmd_diff)

    sp = sub.add_parser("compare", help="статистика по протоколу §8")
    sp.add_argument("runs", nargs="*")
    sp.add_argument("--exp", help="все done-запуски эксперимента по имени")
    sp.add_argument("--datasets")
    sp.add_argument("--out", help="каталог отчёта (report.md/json + png)")
    sp.set_defaults(fn=cmd_compare)

    sp = sub.add_parser("eval", help="до-eval готового run: индекс + метрики (в очередь)")
    sp.add_argument("run")
    sp.add_argument("--datasets", required=True)
    sp.add_argument("--now", action="store_true",
                    help="считать сейчас в этом процессе (блокирует), "
                         "а не ставить в очередь")
    sp.add_argument("--save-index", action="store_true")
    sp.set_defaults(fn=cmd_eval)

    sp = sub.add_parser("model", help="реестр именованных моделей")
    model_sub = sp.add_subparsers(dest="action", required=True)
    s = model_sub.add_parser("name")
    s.add_argument("run")
    s.add_argument("model_name")
    s.add_argument("-m", "--message")
    s.set_defaults(fn=cmd_model)
    s = model_sub.add_parser("list")
    s.set_defaults(fn=cmd_model)

    sp = sub.add_parser("data", help="подготовка/проверка данных")
    sp.add_argument("action", choices=["prepare", "status", "verify"])
    sp.add_argument("--part", help="collection,dev,trec,lite,gate,train-pool")
    sp.add_argument("--all", action="store_true")
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(fn=cmd_data)

    sp = sub.add_parser("test", help="тесты ядра + контрактные")
    sp.set_defaults(fn=cmd_test)

    sp = sub.add_parser("ui", help="веб-UI")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    sp.set_defaults(fn=cmd_ui)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.fn(args)
    return 0
