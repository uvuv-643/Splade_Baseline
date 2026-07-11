import json
import signal

import pytest

from core import memwatch


def test_level_thresholds(monkeypatch):
    monkeypatch.setattr(memwatch, "WARN_PCT", 80.0)
    monkeypatch.setattr(memwatch, "KILL_PCT", 90.0)
    assert memwatch.level(50) == "ok"
    assert memwatch.level(79.9) == "ok"
    assert memwatch.level(80) == "warning"
    assert memwatch.level(89.9) == "warning"
    assert memwatch.level(90) == "critical"
    assert memwatch.level(99) == "critical"


def test_read_int_handles_max_and_garbage(tmp_path):
    (tmp_path / "num").write_text("123\n")
    (tmp_path / "max").write_text("max\n")
    (tmp_path / "junk").write_text("не число")
    assert memwatch._read_int(tmp_path / "num") == 123
    assert memwatch._read_int(tmp_path / "max") is None
    assert memwatch._read_int(tmp_path / "junk") is None
    assert memwatch._read_int(tmp_path / "нет_файла") is None


def test_cgroup_memory_v2(tmp_path, monkeypatch):
    monkeypatch.setattr(memwatch, "CGROUP_V2", tmp_path)
    monkeypatch.setattr(memwatch, "CGROUP_V1", tmp_path / "нет")
    (tmp_path / "memory.max").write_text(str(4 * 2**30))
    (tmp_path / "memory.current").write_text(str(3 * 2**30))
    cg = memwatch.cgroup_memory()
    assert cg == {"limit": 4 * 2**30, "used": 3 * 2**30, "percent": 75.0}


def test_cgroup_memory_unlimited_v1_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(memwatch, "CGROUP_V2", tmp_path / "нет")
    monkeypatch.setattr(memwatch, "CGROUP_V1", tmp_path)
    (tmp_path / "memory.limit_in_bytes").write_text(str(2**63 - 4096))
    (tmp_path / "memory.usage_in_bytes").write_text(str(2**30))
    assert memwatch.cgroup_memory() is None


def test_system_snapshot_prefers_constrained_cgroup(tmp_path, monkeypatch):
    monkeypatch.delenv("LAB_MEM_TOTAL_GB", raising=False)
    monkeypatch.setattr(memwatch, "CGROUP_V2", tmp_path)
    monkeypatch.setattr(memwatch, "CGROUP_V1", tmp_path / "нет")
    (tmp_path / "memory.max").write_text(str(10 * 2**30))
    (tmp_path / "memory.current").write_text(str(9 * 2**30))
    snap = memwatch.system_snapshot()
    if snap["host"]["percent"] < 90:
        assert snap["source"] == "cgroup"
        assert snap["percent"] == 90.0
        assert snap["limit"] == 10 * 2**30


def test_user_memory_sums_own_rss(monkeypatch):
    monkeypatch.setenv("LAB_MEM_TOTAL_GB", "64")
    um = memwatch.user_memory()
    assert um["limit"] == 64 * 2**30
    assert um["used"] > 0
    assert um["percent"] == round(um["used"] / um["limit"] * 100, 1)


def test_user_memory_invalid_limit(monkeypatch):
    monkeypatch.delenv("LAB_MEM_TOTAL_GB", raising=False)
    assert memwatch.user_memory() is None
    monkeypatch.setenv("LAB_MEM_TOTAL_GB", "мусор")
    assert memwatch.user_memory() is None
    monkeypatch.setenv("LAB_MEM_TOTAL_GB", "0")
    assert memwatch.user_memory() is None


def test_system_snapshot_uses_user_quota(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_MEM_TOTAL_GB", "1000000")
    monkeypatch.setattr(memwatch, "CGROUP_V2", tmp_path / "нет")
    monkeypatch.setattr(memwatch, "CGROUP_V1", tmp_path / "нет")
    snap = memwatch.system_snapshot()
    assert snap["source"] == "user"
    assert snap["limit"] == 1000000 * 2**30
    assert snap["used"] == snap["user"]["used"]


def test_note_encode_math():
    enc = memwatch.note_encode("doc:msmarco", done=1000, total=10000,
                               nnz_total=250000)
    assert enc["avg_nnz"] == 250.0
    assert enc["projected_bytes"] == 250 * 10000 * memwatch.BYTES_PER_NNZ
    assert memwatch.encode_state()["kind"] == "doc:msmarco"


def _fake_snap(percent):
    return {"t": "2026-01-01T00:00:00+00:00", "percent": percent,
            "used": percent * 2**28, "limit": 100 * 2**28, "source": "host",
            "host": {}, "cgroup": None,
            "swap": {"total": 0, "used": 0, "percent": 0.0}}


def test_write_oom_report_does_not_overwrite(tmp_path):
    first = memwatch.write_oom_report(tmp_path, "memwatch", "train",
                                      _fake_snap(95), reason="первый")
    memwatch.write_oom_report(tmp_path, "worker", "train", _fake_snap(96))
    report = json.loads(first.read_text())
    assert report["reason"] == "первый"
    assert report["killed_by"] == "memwatch"
    assert report["thresholds"]["kill_pct"] == memwatch.KILL_PCT


def test_memwatch_sample_writes_state(tmp_path, monkeypatch):
    monkeypatch.setattr(memwatch, "system_snapshot", lambda: _fake_snap(50.0))
    watch = memwatch.MemWatch(tmp_path, phase="train")
    watch._sample()
    state = json.loads((tmp_path / "memory.json").read_text())
    assert state["level"] == "ok"
    assert state["phase"] == "train"
    assert state["peak"]["percent"] == 50.0
    assert (tmp_path / "memory.jsonl").exists()


def test_memwatch_warning_prints_once(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(memwatch, "system_snapshot", lambda: _fake_snap(85.0))
    watch = memwatch.MemWatch(tmp_path, phase="train")
    watch._sample()
    watch._sample()
    out = capsys.readouterr().out
    assert out.count("ВНИМАНИЕ") == 1
    assert json.loads((tmp_path / "memory.json").read_text())["level"] == "warning"


def test_memwatch_kill_writes_report_then_sigterm(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(memwatch, "system_snapshot", lambda: _fake_snap(95.0))
    monkeypatch.setattr(memwatch, "GRACE_S", 0.0)
    monkeypatch.setattr(memwatch.os, "kill",
                        lambda pid, sig: calls.append(("kill", sig)))
    monkeypatch.setattr(memwatch.os, "killpg",
                        lambda pgid, sig: calls.append(("killpg", sig)))
    monkeypatch.setattr(memwatch.os, "getpgid", lambda pid: 1)
    watch = memwatch.MemWatch(tmp_path, phase="eval")
    memwatch.note_encode("doc:msmarco", 1000, 10000, 500000)
    watch._sample()
    report = json.loads((tmp_path / "oom_kill.json").read_text())
    assert report["killed_by"] == "memwatch"
    assert report["phase"] == "eval"
    assert report["encode"]["avg_nnz"] == 500.0
    assert report["history"]
    assert calls == [("kill", signal.SIGTERM), ("killpg", signal.SIGKILL)]
    assert watch._stop.is_set()


def test_start_removes_stale_oom_report(tmp_path, monkeypatch):
    monkeypatch.setattr(memwatch, "system_snapshot", lambda: _fake_snap(10.0))
    (tmp_path / "oom_kill.json").write_text("{}")
    watch = memwatch.MemWatch(tmp_path, phase="train").start()
    watch.stop()
    assert not (tmp_path / "oom_kill.json").exists()


def test_terminate_tree_gone_pid():
    assert memwatch.terminate_tree(2 ** 22 + 12345, grace=0.1) is False
