"""Plan 0029 review fix 1: the main bank an arm dir was judged on is recorded in ``arm.json``
(``bank`` / ``bank_source`` / ``bank_sha256``, x3 only; missing = wsb). Judging into a dir on
another bank exits 3; rebuild / report replay the recorded bank when ``--bank`` is omitted and
refuse an explicit other one; ``hbench final`` requires the x3 bank for the main family."""

import json

import pytest
from conftest import TOY_ITEMS
from test_final import _main, _manifest

from hallucination_bench import cli, final
from hallucination_bench.evals.wsbench_chat import bank
from hallucination_bench.evals.wsbench_chat import run as wrun
from hallucination_bench.llm import JudgeConfigError


@pytest.fixture
def x3(tmp_path, monkeypatch):
    """A toy x3 bank: the two toy WSB items tagged like plan 0028's, plus one non-WSB item
    (a copy of the first with a new id) that the WSB bank does not know."""
    src = json.loads((bank.BANK_DIR / "items.json").read_text())
    by_id = {it["id"]: it for it in src["items"]}
    items = [{**by_id[i], "subset": "wsb149"} for i in TOY_ITEMS]
    items.append({**by_id[TOY_ITEMS[0]], "id": "x3-extra-0000", "subset": "x3"})
    p = tmp_path / "x3bank" / "items.json"
    p.parent.mkdir()
    p.write_text(json.dumps({"meta": {"plan": "0028"}, "items": items}))
    monkeypatch.setattr(bank, "X3_ITEMS_PATH", p)
    return p


@pytest.fixture
def x3_readouts(tmp_path, prose_readouts):
    """The toy prose readouts plus the same cells under the extra x3 item's id."""
    rows = [json.loads(x) for x in prose_readouts.read_text().splitlines() if x.strip()]
    extra = [{**r, "id": "x3-extra-0000"} for r in rows if r["id"] == TOY_ITEMS[0]]
    p = tmp_path / "x3_readouts.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows + extra))
    return p


def _run(arms, out, fake, models=None):
    return wrun.run_arms(
        arms,
        out,
        models or wrun.Models(),
        layers=None,
        limit=None,
        rpm=1e6,
        concurrency=4,
        allow_missing=True,
        llm=fake,
    )


def _ids(d):
    return {json.loads(x)["id"] for x in (d / "cells.jsonl").read_text().splitlines()}


def test_wsb_run_records_no_bank(tmp_path, prose_readouts, fake):
    _run([("toy", prose_readouts)], tmp_path, fake)
    d = wrun.arm_dir(tmp_path, wrun.Models(), "toy")
    rec = json.loads((d / "arm.json").read_text())
    assert not {"bank", "bank_source", "bank_sha256"} & set(rec)
    assert wrun.recorded_bank(d) == {"bank": "wsb", "bank_sha256": None}


def test_x3_run_records_the_bank_and_report_replays_it(tmp_path, x3, x3_readouts, fake):
    bank.use_bank("x3")
    _run([("toy", x3_readouts)], tmp_path, fake)
    d = wrun.arm_dir(tmp_path, wrun.Models(), "toy")
    rec = json.loads((d / "arm.json").read_text())
    assert rec["bank"] == "x3" and rec["bank_source"] == f"plan0028-x3:{x3}"
    assert len(rec["bank_sha256"]) == 64
    assert "x3-extra-0000" in _ids(d)
    before = (d / "cells.jsonl").read_bytes()

    # a later process without --bank: the module is back on wsb
    bank.reset_bank()
    bank.SELECTED = None
    assert bank.current_bank() == "wsb"
    # rebuild replays x3: the non-WSB item is still there, cells byte-identical
    s = wrun.rebuild_arm("toy", x3_readouts, tmp_path, wrun.Models())
    assert s["config"]["bank_source"] == f"plan0028-x3:{x3}"
    assert (d / "cells.jsonl").read_bytes() == before
    # the CLI report path (v1 view) too
    bank.reset_bank()
    argv = ["report", "--out", str(tmp_path), "--audit-n", "0"]
    assert cli.main(argv) == 0
    assert (d / "cells.jsonl").read_bytes() == before
    assert bank.current_bank() == "x3"


def test_x3_root_refuses_a_wsb_run_and_an_explicit_wsb_report(tmp_path, x3, x3_readouts, fake):
    bank.use_bank("x3")
    _run([("toy", x3_readouts)], tmp_path, fake)
    bank.reset_bank()
    bank.SELECTED = None
    # judging without --bank (= wsb) into the x3 dir: exit 3
    with pytest.raises(JudgeConfigError, match="--bank x3"):
        _run([("toy", x3_readouts)], tmp_path, fake)
    argv = ["run", "--arm", f"toy={x3_readouts}", "--out", str(tmp_path)]
    assert cli.main(argv) == 3
    # an explicit --bank wsb on rebuild / report: exit 3
    bank.use_bank("wsb")
    with pytest.raises(JudgeConfigError, match="omit --bank"):
        wrun.rebuild_arm("toy", x3_readouts, tmp_path, wrun.Models())
    assert cli.main(["report", "--out", str(tmp_path), "--bank", "wsb"]) == 3


def test_wsb_root_refuses_an_x3_run_and_x3_report(tmp_path, x3, prose_readouts, fake):
    _run([("toy", prose_readouts)], tmp_path, fake)
    bank.use_bank("x3")
    with pytest.raises(JudgeConfigError, match="judged on --bank wsb"):
        _run([("toy", prose_readouts)], tmp_path, fake)
    with pytest.raises(JudgeConfigError, match="judged on --bank wsb"):
        wrun.rebuild_arm("toy", prose_readouts, tmp_path, wrun.Models())


def test_a_replay_back_to_wsb_after_an_x3_dir(tmp_path, x3, x3_readouts, prose_readouts, fake):
    """One report over two roots' worth of dirs: each rebuild uses its own recorded bank."""
    _run([("wsbarm", prose_readouts)], tmp_path, fake)
    bank.use_bank("x3")
    _run([("x3arm", x3_readouts)], tmp_path, fake)
    bank.reset_bank()
    bank.SELECTED = None
    wrun.rebuild_arm("x3arm", x3_readouts, tmp_path, wrun.Models())
    assert bank.current_bank() == "x3"
    wrun.rebuild_arm("wsbarm", prose_readouts, tmp_path, wrun.Models())
    assert bank.current_bank() == "wsb"
    assert "x3-extra-0000" not in _ids(wrun.arm_dir(tmp_path, wrun.Models(), "wsbarm"))


def test_a_changed_x3_items_file_is_refused(tmp_path, x3, x3_readouts, fake):
    bank.use_bank("x3")
    _run([("toy", x3_readouts)], tmp_path, fake)
    data = json.loads(x3.read_text())
    data["meta"]["rebuilt"] = True
    x3.write_text(json.dumps(data))
    bank.use_bank("x3")  # re-point (new sha)
    with pytest.raises(JudgeConfigError, match="hashes differently"):
        _run([("toy", x3_readouts)], tmp_path, fake)
    with pytest.raises(JudgeConfigError, match="hashes differently"):
        wrun.rebuild_arm("toy", x3_readouts, tmp_path, wrun.Models())


def test_legacy_arm_json_means_wsb(tmp_path, prose_readouts, fake):
    _run([("toy", prose_readouts)], tmp_path, fake)
    d = wrun.arm_dir(tmp_path, wrun.Models(), "toy")
    rec = json.loads((d / "arm.json").read_text())
    (d / "arm.json").write_text(json.dumps({k: v for k, v in rec.items() if "bank" not in k}))
    wrun.check_bank(d, wrun.Models())  # no flag: fine
    wrun.replay_bank(d, wrun.Models())
    assert bank.current_bank() == "wsb"


def test_story_is_never_checked(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    (d / "arm.json").write_text(json.dumps({"arm": "d", "bank": "x3"}))
    m = wrun.Models(eval="story")
    wrun.check_bank(d, m)
    wrun.replay_bank(d, m)


# ---------------------------------------------------------------- hbench final


def test_final_requires_the_x3_bank_for_main(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(final, "EXPECTED_LAYERS", (20, 24))  # test_final's toy grid

    def wsb_main(roots):
        a = roots / "main" / "jlens" / "arm.json"
        rec = json.loads(a.read_text())
        a.write_text(json.dumps({k: v for k, v in rec.items() if not k.startswith("bank")}))

    assert _main(tmp_path / "a", _manifest(tmp_path / "a", prep=wsb_main)) == 2
    assert "records bank 'wsb'" in capsys.readouterr().err

    def override_main(roots):
        a = roots / "main" / "jlens" / "arm.json"
        rec = json.loads(a.read_text())
        a.write_text(json.dumps({**rec, "bank_source": "override:/elsewhere/items.json"}))

    assert _main(tmp_path / "b", _manifest(tmp_path / "b", prep=override_main)) == 2
    assert "--bank x3" in capsys.readouterr().err

    def story_without_bank(roots):  # only main is checked
        a = roots / "story" / "jlens" / "arm.json"
        assert "bank" not in json.loads(a.read_text())

    assert _main(tmp_path / "c", _manifest(tmp_path / "c", prep=story_without_bank)) == 0
