"""Plan 0002 rev 18: junk = non_claim + off_topic, merged after parsing (the claimmap-v2.7 prompt
and every cached :M answer stay valid; rebuilding makes no API calls)."""

import json

import pytest
from test_claimmap_v2 import MapFake, _run, e, prompt_quote, unit_lines

from hallucination_bench import calibrate_v2 as cal
from hallucination_bench import consistency_v2 as cons
from hallucination_bench import report_v2
from hallucination_bench import taxonomy_v2 as tx
from hallucination_bench.evals.wsbench_chat import run as wrun
from hallucination_bench.judges import claimmap_v2 as cm
from hallucination_bench.units import split_cell


def test_judge_enum_and_prompt_are_unchanged():
    assert cm.PROMPT_VERSION == "claimmap-v2.7"
    enum = cm.SCHEMA["schema"]["properties"]["units"]["items"]["properties"]["leaf"]["enum"]
    assert enum == list(tx.JUDGE_LEAF_NAMES) and "junk" not in enum
    assert "non_claim" in enum and "off_topic" in enum
    assert "`junk`:" not in cm.SYSTEM and "## The tree: 12 leaves in 8 groups" in cm.SYSTEM
    assert tx.TAXONOMY_VERSION == "v2-r17" and tx.SCORING_VERSION.startswith("v2-r1")


def test_scoring_leaves_and_verdicts():
    assert "junk" in tx.LEAVES and "non_claim" not in tx.LEAVES and "off_topic" not in tx.LEAVES
    assert tx.VERDICTS["junk"] == ("junk",) and "non_claim" not in tx.VERDICTS
    assert tx.group_of("junk") == tx.JUNK and tx.NON_CLAIM not in tx.GROUPS
    assert tx.verdict_of("off_topic") == tx.verdict_of("non_claim") == tx.V_JUNK  # robust
    assert tx.JUNK_REASONS == (
        "off_topic",
        "generic",
        "filler",
        "formatting",
        "junk_text",
        "lens_meta",
    )
    assert not tx.is_claim("junk")


def test_both_old_leaves_map_to_junk_with_the_right_reason():
    units = split_cell(["Hmm, well. Mars is red. Lorem ipsum. Fine."])[0]
    raw = {
        "units": [
            e("R0.U1", "non_claim", reason="filler"),
            e("R0.U2", "off_topic"),
            e("R0.U3", "non_claim"),  # no reason given
            e("R0.U4", "off_topic", reason="generic"),  # a reason on off_topic is overridden
        ],
        "rationale": "",
    }
    segs, _ = cm.parse(raw, units)
    assert [(s.leaf, s.raw_leaf, s.reason) for s in segs] == [
        ("junk", "junk", "filler"),
        ("junk", "junk", "off_topic"),
        ("junk", "junk", None),
        ("junk", "junk", "off_topic"),
    ]


def test_old_gold_maps_to_junk_with_reason_tags():
    c = cal.skeleton(
        {
            "case_id": "g",
            "part": "synthetic",
            "rules": [],
            "prompt": "q",
            "response": "a⟦READ⟧ b",
            "readouts": ["Hmm, well. Mars is red. Lorem ipsum."],
        }
    )
    for g, lf in zip(c["gold"]["units"], ["filler", "off_topic", "non_claim"], strict=True):
        g["leaf"] = lf
    m = cal.remap_case(c)
    got = [(g["leaf"], g.get("tags")) for g in m["gold"]["units"]]
    assert got == [
        ("junk", {"reason": "filler"}),
        ("junk", {"reason": "off_topic"}),
        ("junk", None),
    ]
    assert cal.validate_case(c) == []
    raw = (cal.CAL_DIR / "dev_v2.jsonl").read_text()
    assert '"leaf": "non_claim"' not in raw and '"leaf": "off_topic"' not in raw


def _judge(user):
    q = prompt_quote(user)
    out = []
    for uid, _est, text in unit_lines(user):
        if "Paris" in text:
            out.append(e(uid, "off_topic"))
        elif "friendly" in text and "reply" in text:
            out.append(e(uid, "non_claim", reason="generic"))
        else:
            out.append(e(uid, "correct", source="prefix", supported_by=q))
    return {"units": out, "rationale": "r"}


def test_rebuild_after_the_merge_makes_no_calls(tmp_path, prose_readouts):
    out = tmp_path / "out"
    fake = MapFake({"map": _judge})
    _run([("toy", prose_readouts)], out, fake, stages=["map"])
    assert fake.n("map") == 15
    # a second run (and the rebuild inside it) is served entirely from the cached answers
    fake2 = MapFake({"map": _judge})
    _run([("toy", prose_readouts)], out, fake2, stages=["map"])
    assert fake2.calls == []
    # the cache-only rebuild path never gets an LLM at all
    wrun.rebuild_arm("toy", prose_readouts, out, wrun.Models())
    d = wrun.arm_dir(out, wrun.Models(), "toy")
    s2 = json.loads((d / "summary_v2.json").read_text())
    b = s2["overall"]
    assert "junk" in b["char_share_leaf"] and "off_topic" not in b["char_share_leaf"]
    jr = {k: v["value"] for k, v in b["char_share_junk_reason"].items()}
    assert jr["off_topic"] > 0 and jr["generic"] > 0
    assert sum(jr.values()) == pytest.approx(b["char_share_verdict"]["junk"]["value"])
    segs = [
        s
        for line in (d / "cells.jsonl").read_text().splitlines()
        for s in (json.loads(line).get("map") or {}).get("segments", [])
    ]
    leaves = {s["leaf"] for s in segs}
    assert "junk" in leaves and not leaves & {"non_claim", "off_topic"}
    assert {s["reason"] for s in segs if "Paris" in s["text"]} == {"off_topic"}
    root = report_v2.report(out, "claude-opus-5-5", rebuild=False, audit_n=1, verdict_n=1)
    md = (root / "report_v2.md").read_text()
    assert "Junk share by reason" in md and "| off_topic |" not in md.split("Junk share")[0][-50:]


def test_consistency_merges_old_cells_and_rescores_without_calls(tmp_path):
    cells = [
        {
            "arm": "a",
            "key": "k",
            "units": [{"id": "R0.U1", "text": "x"}, {"id": "R0.U2", "text": "y"}],
            "r0": {"R0.U1": "non_claim", "R0.U2": "off_topic"},
            "r1": {"R0.U1": "off_topic", "R0.U2": "non_claim"},
        }
    ]
    res = cons.score(cells)
    assert res["overall"]["raw_agreement"] == 1.0  # both units are junk in both runs
    d = tmp_path / "consistency" / "claude-opus-5-5"
    d.mkdir(parents=True)
    stem = f"consistency_{cm.PROMPT_VERSION}"
    (d / f"{stem}_cells.jsonl").write_text(json.dumps(cells[0]) + "\n")
    (d / f"{stem}.json").write_text(json.dumps({"arms": "a:44", "spend_this_run": {"usd": 5}}))
    out = cons.rescore(tmp_path, "claude-opus-5-5")
    assert out["spend_this_run"]["usd"] == 0 and out["spend_original"]["usd"] == 5
    assert out["arms"] == "a:44" and out["overall"]["self_agreement"]["junk"]["dice"] == 1.0
    row = json.loads((d / f"{stem}_cells.jsonl").read_text().splitlines()[0])
    assert set(row["r0"].values()) == {"junk"}
