from core import snapshots


def _make_exp(tmp_path, content):
    exp = tmp_path / "exp"
    exp.mkdir(exist_ok=True)
    (exp / "train.py").write_text(content, encoding="utf-8")
    return exp


def test_same_content_same_hash(tmp_path):
    exp = _make_exp(tmp_path, "def train(cfg, ctx): pass\n")
    snaps = tmp_path / "snapshots"
    h1 = snapshots.save(name="a", exp_dir=exp, snapshots_dir=snaps)
    h2 = snapshots.save(name="a", exp_dir=exp, snapshots_dir=snaps)
    assert h1 == h2
    assert len(list(snaps.glob("*/exp"))) == 1


def test_changed_content_new_hash(tmp_path):
    exp = _make_exp(tmp_path, "v1\n")
    snaps = tmp_path / "snapshots"
    h1 = snapshots.save(name="a", exp_dir=exp, snapshots_dir=snaps)
    _make_exp(tmp_path, "v2\n")
    h2 = snapshots.save(name="b", exp_dir=exp, snapshots_dir=snaps)
    assert h1 != h2
    idx = snapshots.load_index(snaps)
    assert idx[h2]["parent"] == h1


def test_pycache_ignored(tmp_path):
    exp = _make_exp(tmp_path, "v1\n")
    snaps = tmp_path / "snapshots"
    h1 = snapshots.save(name="a", exp_dir=exp, snapshots_dir=snaps)
    cache = exp / "__pycache__"
    cache.mkdir()
    (cache / "x.pyc").write_text("junk")
    h2 = snapshots.save(name="a", exp_dir=exp, snapshots_dir=snaps)
    assert h1 == h2


def test_resolve_and_diff(tmp_path):
    exp = _make_exp(tmp_path, "line1\n")
    snaps = tmp_path / "snapshots"
    h1 = snapshots.save(name="first", exp_dir=exp, snapshots_dir=snaps)
    _make_exp(tmp_path, "line2\n")
    h2 = snapshots.save(name="second", exp_dir=exp, snapshots_dir=snaps)
    assert snapshots.resolve("first", snapshots_dir=snaps) == h1
    diff = snapshots.diff_snapshots(h1, h2, snapshots_dir=snaps)
    assert "-line1" in diff and "+line2" in diff
