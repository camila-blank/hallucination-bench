"""Plan 0002 rev 17: contradicted_by / supported_by quotes (fail closed) and the reported judge
noise (consistency-v2 r0 vs r1, the "judge noise ±" column)."""

import json

import pytest
from test_claimmap_v2 import MapFake, _run, e, prompt_quote, sp, unit_lines

from hallucination_bench import consistency_v2 as cons
from hallucination_bench import report_v2
from hallucination_bench import taxonomy_v2 as tx
from hallucination_bench.evals.wsbench_chat import run as wrun
from hallucination_bench.judges import claimmap_v2 as cm
from hallucination_bench.units import split_cell

CONV = cm.conversation_text(
    "Who painted the Mona Lisa?", "It was painted by Leonardo⟦READ⟧ da Vinci around 1503."
)


def test_prompt_states_the_quote_rules():
    for s in (cm.SYSTEM, cm.SYSTEM_V5C):
        assert (
            "If you cannot quote the text that rules it out, it is not contradicted: label it "
            "`unverifiable`." in s
        )
        assert tx.SUPPORT_RULE in s and "- `contradicted_by` " in s and "- `supported_by` " in s
    props = cm.SCHEMA["schema"]["properties"]["units"]["items"]["properties"]
    assert "contradicted_by" in props and "supported_by" in props
    assert cm.PROMPT_VERSION == "claimmap-v2.7"


def test_conversation_text_drops_the_marker():
    assert "⟦READ⟧" not in CONV and "Leonardo da Vinci" in CONV


def test_contradiction_quotes_are_checked_and_fail_closed():
    units = split_cell(["Picasso painted it. It was in 1990 exactly. A painting. Sure."])[0]
    raw = {
        "units": [
            e(
                "R0.U1",
                "substitution_sibling",
                pair={"said": "Picasso", "actual": "Leonardo"},
                probe_target="Picasso",
                contradicted_by="“...painted by Leonardo da Vinci…”",  # quotes/ellipses trimmed
            ),
            e("R0.U2", "detail", contradicted_by="painted in 1990"),  # not in the conversation
            e("R0.U3", "false_claim"),  # no quote at all
            e("R0.U4", "non_claim", contradicted_by="Mona Lisa"),  # dropped: not incorrect
        ],
        "rationale": "",
    }
    segs, c = cm.parse(raw, units, CONV)
    got = [(s.leaf, s.raw_leaf, s.judge_leaf, s.unquoted_contradiction) for s in segs]
    assert got == [
        ("substitution_sibling", "substitution_sibling", None, False),
        ("unverifiable", "unverifiable", "detail", True),
        ("unverifiable", "unverifiable", "false_claim", True),
        ("non_claim", "non_claim", None, False),
    ]
    assert segs[0].contradicted_by == "painted by Leonardo da Vinci"  # spans the marker
    assert segs[1].contradicted_by == "painted in 1990"  # kept for the audit
    assert segs[3].contradicted_by is None
    assert c["n_contradiction_quotes_checked"] == 3 and c["n_unquoted_contradictions"] == 2
    assert c["n_probe_candidates"] == 1  # the substitution only


def test_support_quotes_are_checked_except_entailed():
    units = split_cell(["Leonardo painted it. Around 1503 or so. It is a painting. Yes."])[0]
    raw = {
        "units": [
            e("R0.U1", "correct", source="prefix", supported_by="painted by Leonardo"),
            e("R0.U2", "correct", source="after", supported_by="around 1600"),
            e("R0.U3", "correct", source="entailed"),  # no quote needed
            e("R0.U4", "correct"),  # no source, no quote -> fail closed
        ],
        "rationale": "",
    }
    segs, c = cm.parse(raw, units, CONV)
    assert [(s.leaf, s.unquoted_support) for s in segs] == [
        ("correct", False),
        ("unverifiable", True),
        ("correct", False),
        ("unverifiable", True),
    ]
    assert c["n_support_quotes_checked"] == 3 and c["n_unquoted_support"] == 2


def test_split_pieces_carry_their_own_quotes():
    units = split_cell(["Picasso painted the Mona Lisa"])[0]
    raw = {
        "units": [
            e(
                "R0.U1",
                "non_claim",
                splits=[
                    sp(
                        "Picasso",
                        "substitution_sibling",
                        contradicted_by="Leonardo",
                        pair={"said": "Picasso", "actual": "Leonardo"},
                    )
                ],
                rest_leaf="correct",
                supported_by="the Mona Lisa",
                source="prefix",
            )
        ],
        "rationale": "",
    }
    segs, c = cm.parse(raw, units, CONV)
    assert [(s.leaf, s.contradicted_by, s.supported_by) for s in segs] == [
        ("substitution_sibling", "Leonardo", None),
        ("correct", None, "the Mona Lisa"),
    ]
    assert c["n_unquoted_contradictions"] == 0 and c["n_unquoted_support"] == 0


def test_pipeline_fails_closed_on_unquoted_incorrect(tmp_path, prose_readouts):
    def mapper(user):
        q = prompt_quote(user)
        out = []
        for uid, _est, text in unit_lines(user):
            if "Paris" in text:
                out.append(e(uid, "false_claim", contradicted_by="NOT IN THE CONVERSATION"))
            else:
                out.append(e(uid, "correct", source="prefix", supported_by=q))
        return {"units": out, "rationale": "r"}

    out = tmp_path / "out"
    _run([("toy", prose_readouts)], out, MapFake({"map": mapper}), stages=["map"])
    d = wrun.arm_dir(out, wrun.Models(), "toy")
    s2 = json.loads((d / "summary_v2.json").read_text())
    h = s2["health"]
    assert h["n_unquoted_contradictions"] == 1 and h["n_unquoted_support"] == 0
    assert s2["overall"]["rates"]["verdict_hallucination"]["all"]["value"] == 0
    rec = json.loads((d / "cells.jsonl").read_text().splitlines()[0])
    paris = next(x for x in rec["map"]["segments"] if "Paris" in x["text"])
    assert paris["leaf"] == "unverifiable" and paris["judge_leaf"] == "false_claim"
    assert paris["contradicted_by"] == "NOT IN THE CONVERSATION"


# ------------------------------------------------------------------ noise


def _cell(arm, r0, r1):
    units = [{"id": f"R0.U{i + 1}", "text": "x" * 10} for i in range(len(r0))]
    return {
        "arm": arm,
        "key": f"{arm}-k",
        "units": units,
        "r0": {u["id"]: lf for u, lf in zip(units, r0, strict=True)},
        "r1": {u["id"]: lf for u, lf in zip(units, r1, strict=True)},
    }


def test_noise_split_kappa_and_families():
    cells = [
        _cell("s3d-rl600", ["detail", "correct"], ["false_claim", "correct"]),
        _cell("s3d-rl600", ["false_claim", "non_claim"], ["false_claim", "non_claim"]),
        _cell("nla-rl-L42", ["unverifiable", "correct"], ["detail", "correct"]),
    ]
    res = cons.score(cells)
    assert set(res["by_family"]) == {"s3d", "nla"}
    sk = res["detail_vs_false_claim"]["overall"]
    assert sk["n_units"] == 2 and sk["raw_agreement"] == 0.5  # nla's pair is not both-hal
    nz = res["noise"]["overall"]
    assert nz["n_readouts"] == 3
    h = nz["hallucination"]["rate"]
    assert h["r0"] == pytest.approx(2 / 3) and h["r1"] == pytest.approx(1.0)
    assert h["diff"] == pytest.approx(-1 / 3) and h["pm"] == pytest.approx(1 / 6)
    d = nz["detail"]["share"]
    assert d["r0"] == pytest.approx(0.5 / 3) and d["r1"] == pytest.approx(0.5 / 3)
    assert res["noise"]["by_arm"]["nla-rl-L42"]["unresolved"]["rate"]["r0"] == 1.0
    md = cons.render_markdown(res)
    assert "family s3d" in md and "detail vs false_claim" in md and "rate r0" in md


def test_report_has_noise_column_section_and_quoted_examples(tmp_path, prose_readouts):
    def mapper(user):
        q = prompt_quote(user)
        out = []
        for uid, _est, text in unit_lines(user):
            if "Paris" in text:
                out.append(e(uid, "detail", contradicted_by=q))
            else:
                out.append(e(uid, "correct", source="prefix", supported_by=q))
        return {"units": out, "rationale": "r"}

    out = tmp_path / "out"
    _run([("toy", prose_readouts)], out, MapFake({"map": mapper}), stages=["map"])
    root = report_v2.report(out, "claude-opus-5-5", audit_n=1, verdict_n=1)
    md = (root / "report_v2.md").read_text()
    assert report_v2.NOISE_HEAD in md and "**Not run for this prompt version**" in md
    assert "Hallucination examples (with the quote that contradicts them)" in md
    ex = md.split("Hallucination examples")[1]
    assert "| detail |" in ex and "contradicted_by" in ex
    # with a consistency result for this prompt version the column and section fill in
    cdir = out / "consistency" / "claude-opus-5-5"
    cdir.mkdir(parents=True)
    res = cons.score([_cell("toy", ["detail", "correct"], ["false_claim", "correct"])])
    res.update({"arms": "toy:44", "n_per_arm": 1, "seed": 0})
    (cdir / f"consistency_{cm.PROMPT_VERSION}.json").write_text(json.dumps(res))
    root = report_v2.report(out, "claude-opus-5-5", rebuild=False, audit_n=1, verdict_n=1)
    md = (root / "report_v2.md").read_text()
    assert "h 0.0 / fc 50.0 / d 50.0" in md  # readout-rate noise of the toy sample
    assert "Claim-map judge consistency" in md and "Not run for this prompt" not in md
    audit = (root / "audit_samples_v2.md").read_text()
    assert "- contradicted_by:" in audit
