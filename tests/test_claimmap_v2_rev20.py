"""Plan 0002 rev 20 (scoring only): substitution_category is the third hallucination component;
substitution_sibling stays a probe candidate."""

import json

import pytest
from test_claimmap_v2 import MapFake, _rec, _run, e, prompt_quote, unit_lines

from hallucination_bench import consistency_v2 as cons
from hallucination_bench import report_v2
from hallucination_bench import taxonomy_v2 as tx
from hallucination_bench.evals.wsbench_chat import run as wrun
from hallucination_bench.evals.wsbench_chat import score_v2
from hallucination_bench.judges import claimmap_v2 as cm
from hallucination_bench.units import split_cell


def test_category_is_a_hallucination_sibling_a_probe():
    assert tx.VERDICTS["hallucination"] == ("detail", "false_claim", "substitution_category")
    assert tx.verdict_of("substitution_category") == tx.V_HALLUCINATION
    assert tx.verdict_of("substitution_sibling") == tx.V_PROBE
    assert "substitution_category" not in tx.PROBE_CANDIDATE_LEAVES
    assert cm.PROMPT_VERSION == "claimmap-v2.7"  # scoring-only change


def test_parse_keeps_the_category_pair_but_no_probe_target():
    units = split_cell(["Hendrix played at Abbey Road. Clapton played guitar."])[0]
    conv = cm.conversation_text("Who played Woodstock?", "Jimi Hendrix⟦READ⟧ at Woodstock.")
    raw = {
        "units": [
            e(
                "R0.U1",
                "substitution_category",
                contradicted_by="Woodstock",
                probe_target="x",
                pair={"said": "Abbey Road", "actual": "Woodstock"},
            ),
            e(
                "R0.U2",
                "substitution_sibling",
                contradicted_by="Jimi Hendrix",
                probe_target="Clapton",
                pair={"said": "Clapton", "actual": "Hendrix"},
            ),
        ],
        "rationale": "",
    }
    segs, c = cm.parse(raw, units, conv)
    assert segs[0].pair == {"said": "Abbey Road", "actual": "Woodstock"}
    assert segs[0].probe_target is None and segs[1].probe_target == "Clapton"
    assert c["n_probe_candidates"] == 1


def test_three_components_in_rates_shares_and_the_v5c_comparison():
    recs = [
        _rec("a", 44, [[("substitution_category", 5), ("correct", 5)]], cls="hallucinated"),
        _rec("b", 44, [[("detail", 2), ("correct", 8)]], cls="consistent"),
        _rec("c", 44, [[("substitution_sibling", 10)]], cls="consistent"),
    ]
    acc = score_v2.AccumulatorV2()
    for r in recs:
        acc.add(r)
    b = score_v2.block(acc.cells, n_boot=50)
    assert b["rates"]["verdict_hallucination"]["all"]["value"] == pytest.approx(2 / 3)
    assert b["rates"]["substitution_category"]["all"]["value"] == pytest.approx(1 / 3)
    assert b["rates"]["verdict_probe_candidate"]["all"]["value"] == pytest.approx(1 / 3)
    cv = b["char_share_verdict"]
    assert cv["hallucination"]["value"] == pytest.approx((0.5 + 0.2) / 3)
    assert b["char_share_leaf"]["substitution_category"]["value"] == pytest.approx(0.5 / 3)
    assert b["v5c_comparison"]["v2_substitution_category_rate"]["value"] == pytest.approx(1 / 3)
    nz = cons.noise(
        [
            {
                "arm": "a",
                "key": "k",
                "units": [{"id": "R0.U1", "text": "xx"}],
                "r0": {"R0.U1": "substitution_category"},
                "r1": {"R0.U1": "detail"},
            }
        ]
    )
    assert nz["hallucination"]["rate"]["diff"] == 0
    assert nz["substitution_category"]["rate"]["pm"] == pytest.approx(0.5)


def _judge(user):
    q = prompt_quote(user)
    out = []
    for uid, _est, text in unit_lines(user):
        if "Paris" in text:
            out.append(
                e(
                    uid,
                    "substitution_category",
                    contradicted_by=q,
                    pair={"said": "Paris", "actual": "x"},
                    probe_target="Paris",
                )
            )
        else:
            out.append(e(uid, "correct", source="prefix", supported_by=q))
    return {"units": out, "rationale": "r"}


def test_rebuild_report_and_probe_candidates_without_calls(tmp_path, prose_readouts):
    out = tmp_path / "out"
    _run([("toy", prose_readouts)], out, MapFake({"map": _judge}), stages=["map"])
    d = wrun.arm_dir(out, wrun.Models(), "toy")
    n = len((d / "cache.jsonl").read_text().splitlines())
    root = report_v2.report(out, "claude-opus-5-5", audit_n=1, verdict_n=1)  # rebuilds, cache only
    assert len((d / "cache.jsonl").read_text().splitlines()) == n
    rows = (root / "probe_candidates.jsonl").read_text().splitlines()
    assert not any(json.loads(x)["leaf"] == "substitution_category" for x in rows)
    md = (root / "report_v2.md").read_text()
    head = next(line for line in md.splitlines() if line.startswith("| arm | n readouts | hall"))
    assert "↳ false_claim | ↳ detail | ↳ substitution_category" in head
    assert "↳ v2 substitution_category" in md
    assert "judge noise ± (h / fc / d / sc / p / c / u)" in md
    s2 = json.loads((d / "summary_v2.json").read_text())
    assert s2["overall"]["rates"]["verdict_hallucination"]["all"]["value"] > 0
