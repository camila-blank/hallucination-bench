"""Plans 0028 / 0029: the v8 main family reaches plan 0028's 3x bank through ``--bank x3``
(``HBENCH_X3_BANK`` / ``paths.X3_READOUTS``) instead of the ``HBENCH_WSBENCH_CHAT_ITEMS`` override,
which keeps working."""

import argparse
import json
import os

import pytest

from hallucination_bench import cli, paths
from hallucination_bench.evals import bank_for
from hallucination_bench.evals.wsbench_chat import bank


@pytest.fixture
def x3(tmp_path, monkeypatch):
    """A toy x3 bank (two WSB items, tagged like plan 0028's) and a restored bank module."""
    src = json.loads((bank.BANK_DIR / "items.json").read_text())
    items = [{**it, "subset": "wsb149"} for it in src["items"][:2]]
    p = tmp_path / "x3" / "items.json"
    p.parent.mkdir()
    p.write_text(json.dumps({"meta": {"plan": "0028"}, "items": items}))
    monkeypatch.setattr(bank, "X3_ITEMS_PATH", p)
    saved = (bank.ITEMS_PATH, bank.BANK_SOURCE, bank.BANK_COMMIT)
    yield p, items
    bank.ITEMS_PATH, bank.BANK_SOURCE, bank.BANK_COMMIT = saved
    bank.load_bank.cache_clear()


def test_use_bank_x3_points_the_chat_bank_at_plan_0028(x3):
    p, items = x3
    n_wsb = len(bank.load_items())
    bank.use_bank("x3")
    assert bank.ITEMS_PATH == p
    assert bank_for("wsbench_chat").load_items() == items
    assert bank.BANK_SOURCE == f"plan0028-x3:{p}" and bank.BANK_COMMIT.startswith("sha256:")
    bank.use_bank("wsb")
    assert len(bank.load_items()) == n_wsb
    assert bank.BANK_SOURCE.startswith("camilablank/wsbench-pub@")


def test_default_x3_location_follows_plan_0028():
    assert bank.BANKS == ("wsb", "x3")
    if "HBENCH_X3_BANK" not in os.environ:
        assert bank.X3_ITEMS_PATH == bank.DATA / "wsbench_chat_x3" / "items.json"


def test_use_bank_x3_missing_or_conflicting(x3, tmp_path, monkeypatch):
    p, _ = x3
    monkeypatch.setattr(bank, "_OVERRIDE", str(tmp_path / "other.json"))
    with pytest.raises(SystemExit, match="conflicts with HBENCH_WSBENCH_CHAT_ITEMS"):
        bank.use_bank("x3")
    monkeypatch.setattr(bank, "_OVERRIDE", str(p))  # the override naming the same file is fine
    bank.use_bank("x3")
    assert bank.ITEMS_PATH == p
    monkeypatch.setattr(bank, "_OVERRIDE", None)
    monkeypatch.setattr(bank, "X3_ITEMS_PATH", tmp_path / "absent" / "items.json")
    with pytest.raises(SystemExit, match="not found"):
        bank.use_bank("x3")
    with pytest.raises(ValueError):
        bank.use_bank("nope")


def _run_args(*argv: str) -> argparse.Namespace:
    return cli.build_parser().parse_args(["run", "--out", "o", *argv])


def test_cli_bank_x3_defaults_the_readouts_root(x3):
    args = _run_args("--bank", "x3")
    cli._apply_bank(args, readouts=True)
    assert args.readouts_root == str(paths.X3_READOUTS)
    assert bank.ITEMS_PATH == x3[0]
    args = _run_args("--bank", "x3", "--readouts-root", "/elsewhere")
    cli._apply_bank(args, readouts=True)
    assert args.readouts_root == "/elsewhere"
    # without --bank nothing changes (the env override path stays as it was)
    before = bank.ITEMS_PATH
    args = _run_args()
    cli._apply_bank(args, readouts=True)
    assert args.readouts_root is None and bank.ITEMS_PATH == before
    rp = cli.build_parser().parse_args(["report", "--bank", "x3", "--view", "v2"])
    assert rp.bank == "x3"


def test_cli_bank_is_main_only(x3):
    for ev in ("story", "agentic_misalignment"):
        with pytest.raises(SystemExit, match="wsbench_chat"):
            cli._apply_bank(_run_args("--bank", "x3", "--eval", ev), readouts=True)
