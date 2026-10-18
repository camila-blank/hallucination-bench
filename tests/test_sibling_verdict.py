"""Plan 0004: claim units, consensus, verdict judge parsing, categories, rule labels, set-B
alignment and the viewer data builder."""

import json
import sys
from pathlib import Path

from hallucination_bench.evals.sibling_probe import br_bank as BB
from hallucination_bench.evals.sibling_probe import claims as CL
from hallucination_bench.evals.sibling_probe import verdict as V

ROOT = Path(__file__).resolve().parents[1]


def seg(readout, start, end, text, leaf="false_claim", verdict="hallucination", target=None, tc=()):
    return {
        "readout": readout,
        "start": start,
        "end": end,
        "text": text,
        "leaf": leaf,
        "verdict": verdict,
        "probe_target": target,
        "target_concepts": list(tc),
    }


def cell_a(**kw):
    rec = {
        "key": "it1__L020__p5",
        "id": "it1",
        "layer": 20,
        "pos": 5,
        "category": "country",
        "gold": "south_korea",
        "named": [],
        "response_named": [],
        "kind": "prose",
        "status": "judged",
        "token_hits": {},
        "mentions": [],
        "segments": [],
    }
    rec.update(kw)
    return rec


# ------------------------------------------------------------------ claim units


def test_units_set_a_dedup_and_exclusions():
    segs = [
        seg(0, 0, 20, "The user is from Japan", tc=["japan"]),
        seg(
            1,
            0,
            25,
            "Japan, China, or Korea",
            leaf="substitution_sibling",
            verdict="probe_candidate",
            tc=["japan"],
        ),
        seg(1, 26, 40, "user is Korean", leaf="correct_stated", verdict="correct"),
    ]
    ments = [
        {"concept": "japan", "segment": 0, "verdict": "hallucination", "readout": 0},
        {"concept": "japan", "segment": 1, "verdict": "probe_candidate", "readout": 1},
        {"concept": "china", "segment": 1, "verdict": "probe_candidate", "readout": 1},
        {
            "concept": "south_korea",
            "segment": 1,
            "verdict": "probe_candidate",
            "readout": 1,
        },  # gold
        {"concept": "vietnam", "segment": 2, "verdict": "correct", "readout": 1},  # correct: no
        {"concept": "taiwan", "segment": 1, "verdict": "probe_candidate", "readout": 1},
    ]
    rec = cell_a(segments=segs, mentions=ments, response_named=["taiwan"])
    us = CL.units_set_a("s3d", [rec])
    by = {u.concept: u for u in us}
    assert set(by) == {"japan", "china"}  # gold, correct-only and response-named excluded
    assert len(by["japan"].segments) == 2 and by["japan"].readouts == [0, 1]
    assert by["japan"].uid == "A|s3d|it1|20|5|japan"


def test_units_set_a_jlens_token_rule():
    segs = [seg(0, 0, 10, "Japan maybe")]
    ments = [{"concept": "japan", "segment": 0, "verdict": "hallucination", "readout": 0}]
    rec = cell_a(kind="tokens", segments=segs, mentions=ments, token_hits={})
    assert CL.units_set_a("jlens", [rec]) == []
    rec["token_hits"] = {"japan": [" Japan"]}
    assert len(CL.units_set_a("jlens", [rec])) == 1


def test_units_set_b_norm_and_entity_substitution():
    segs = [
        seg(0, 0, 30, "plays the sport of cricket", leaf="substitution_sibling", target="cricket"),
        seg(
            0, 31, 60, "Usain Bolt is a sprinter", leaf="substitution_sibling", target="Usain Bolt"
        ),
        seg(0, 61, 70, "swimming", leaf="substitution_category", target="swim"),
    ]
    rec = cell_a(
        category="sport", gold="boxing", segments=segs, key="br-ali__L020__p7", id="br-ali"
    )
    table = {
        "cricket": ("member", "cricket"),
        "Usain Bolt": ("entity_substitution", "usain_bolt"),
        "swim": None,
    }
    us = CL.units_set_b("mlnla-rl600", [rec], lambda r, s: table[s["probe_target"]])
    kinds = {u.concept: u.kind for u in us}
    assert kinds == {"cricket": "sibling", "usain_bolt": "entity_substitution"}
    assert all(u.set == "B" for u in us)


# ------------------------------------------------------------------ consensus


def U(method, layer, item="i", pos=1, concept="c", set_="A", kind="sibling"):
    return CL.Unit(method, set_, item, layer, pos, concept, "cat", "g", kind=kind)


def test_consensus_triples_and_projection():
    us = [U("s3d", 28), U("jlens", 32), U("nla-rl-L42", 42), U("s3d", 52), U("jlens", 60)]
    f = CL.consensus_flags(us)
    assert f[us[0].uid]["consensus"] and f[us[1].uid]["consensus"]
    assert f[us[2].uid]["consensus"]  # NLA-L42 unit at the triple
    assert not f[us[3].uid]["consensus"]  # 52 outside [24, 36] and no partner within 4
    assert not f[us[4].uid]["consensus"]
    assert CL.consensus_triples(us) == {("A", "i", 1, "c"): [(28, 32)]}


def test_consensus_requires_nla_and_same_pos():
    us = [U("s3d", 28), U("jlens", 28), U("nla-rl-L42", 42, pos=2)]
    f = CL.consensus_flags(us)
    assert not any(v["consensus"] for v in f.values())


def test_consensus_mlnla_variant_is_separate():
    us = [U("s3d", 40), U("jlens", 44), U("mlnla-rl600", 40), U("mlnla-rl600", 56)]
    f = CL.consensus_flags(us)
    assert not f[us[0].uid]["consensus"]  # no NLA-L42: primary label off
    assert f[us[0].uid]["consensus_mlnla"] and f[us[1].uid]["consensus_mlnla"]
    assert f[us[2].uid]["consensus_mlnla"]  # its layer is part of the triple (40, 44, 40)
    assert not f[us[3].uid]["consensus_mlnla"]  # 56: 12 from s3d
    us2 = [U("s3d", 40), U("jlens", 44), U("mlnla-rl600", 36)]  # 36 vs 44: 8 apart
    assert not any(v["consensus_mlnla"] for v in CL.consensus_flags(us2).values())


def test_consensus_ignores_entity_substitution():
    us = [U("s3d", 28, kind="entity_substitution"), U("jlens", 28), U("nla-rl-L42", 42)]
    assert not any(v["consensus"] for v in CL.consensus_flags(us).values())


# ------------------------------------------------------------------ categories / rule


def test_category_precedence():
    kw = {"kind": "sibling", "form_label": "assertion"}
    assert V.category(probe_status="ok", detected=True, consensus=True, **kw) == "validated"
    assert V.category(probe_status="ok", detected=False, consensus=True, **kw) == "consensus"
    assert V.category(probe_status="na", detected=False, consensus=True, **kw) == "consensus"
    assert V.category(probe_status="na", detected=False, consensus=False, **kw) == "probe_na"
    assert V.category(probe_status="ok", detected=False, consensus=False, **kw) == "hallucination"
    assert (
        V.category(
            kind="sibling",
            probe_status="ok",
            detected=False,
            consensus=False,
            form_label="option_form",
        )
        == "option"
    )
    assert (
        V.category(
            kind="sibling", probe_status="ok", detected=False, consensus=False, form_label="other"
        )
        == "unclear"
    )
    assert (
        V.category(
            kind="entity_substitution",
            probe_status="ok",
            detected=True,
            consensus=True,
            form_label="assertion",
        )
        == "entity_substitution"
    )
    assert V.is_hallucination("entity_substitution", "assertion")
    assert not V.is_hallucination("entity_substitution", "option_form")


def test_depth_and_rule_labels():
    assert [V.depth(L) for L in (20, 36, 40, 42, 44, 60)] == [
        "early",
        "early",
        "mid",
        "mid",
        "late",
        "late",
    ]
    assert (
        V.rule_label(36) == "hallucination"
        and V.rule_label(40) == "unclear"
        and V.rule_label(44) == "option"
    )
    assert V.rule_label(40, 40) == "option" and V.rule_label(40, 44) == "hallucination"
    assert V.rule_label(42, 44) == "hallucination" and V.rule_label(42, 40) == "option"


def test_unit_form_derivation():
    assert V.unit_form(["alternatives", "assertion"]) == "assertion"
    assert V.unit_form(["alternatives", "question", "continuation"]) == "option_form"
    assert V.unit_form(["alternatives", "other"]) == "other"
    assert V.unit_form([]) == "other"


# ------------------------------------------------------------------ judge prompt / parser


def test_parse_fails_closed_on_non_verbatim_quote():
    samples = ["The sport he practices is Cricket. Boxing or football?"]
    raw = {
        "segments": [
            {"id": "S1", "form": "assertion", "quote": "The sport he practices is Cricket"},
            {"id": "S2", "form": "alternatives", "quote": "Boxing or soccer"},
        ],
        "why": "x",
    }
    p = V.parse(raw, 2, samples)
    assert p["seg_forms"] == ["assertion", "other"]
    assert p["n_unquoted"] == 1 and p["form_label"] == "assertion"
    p = V.parse(None, 2, samples)
    assert p["seg_forms"] == ["other", "other"] and p["form_label"] == "other"
    assert not V.validate({"segments": [{"id": "S1", "form": "assertion", "quote": "x"}]}, 2)


def test_parse_quote_with_marks_and_whitespace():
    samples = ["a  b\nc d"]
    raw = {"segments": [{"id": "s1", "form": "question", "quote": "⟦S1: a b c⟧"}], "why": ""}
    assert V.parse(raw, 1, samples)["seg_forms"] == ["question"]


def test_render_layer_blind_omits_layer():
    segs = [{"readout": 0, "start": 4, "end": 11, "text": "Cricket"}]
    kw = dict(
        prompt="P",
        response="R",
        concept="Cricket",
        samples=["Was Cricket!"],
        segments=segs,
        probe={"status": "ok", "z": -0.5, "detected": False},
    )
    _s1, u1 = V.render(layer=24, **kw)
    s2, u2 = V.render(layer=None, **kw)
    assert "LENS LAYER: 24" in u1 and "early" in u1
    assert "LENS LAYER" not in u2 and "early" not in u2 and "early" not in s2
    assert "⟦S1: Cricket⟧" in u1  # (start..end slices the sample)
    assert "Segments to label: S1." in u2


def test_mark_readouts_multiple():
    out = V.mark_readouts(
        ["aaa bbb ccc", "ddd"],
        [
            {"readout": 0, "start": 0, "end": 3},
            {"readout": 0, "start": 8, "end": 11},
            {"readout": 1, "start": 0, "end": 3},
        ],
    )
    assert out == ["⟦S1: aaa⟧ bbb ⟦S2: ccc⟧", "⟦S3: ddd⟧"]


# ------------------------------------------------------------------ set B alignment


def test_set_b_alignment():
    bank = [{"id": "x", "pos": 7, "tokens": ["The"] * 7 + [" of"]}]
    good = {"s3d": [{"id": "x", "layer": 20, "pos": 7, "token": " of"}]}
    assert BB.alignment_problems(bank, good) == []
    bad = {
        "s3d": [{"id": "x", "layer": 20, "pos": 6, "token": " of"}],
        "jl": [{"id": "x", "layer": 20, "pos": 7, "token": " the"}],
    }
    probs = BB.alignment_problems(bank, bad)
    assert len(probs) == 2


def test_template_of():
    t, e = BB.template_of(
        {"prompt": "The athlete Muhammad Ali plays the sport of", "eval_render": "plain"}
    )
    assert t == "The athlete NAME plays the sport of" and e == "Muhammad Ali"
    t, e = BB.template_of(
        {"prompt": "What is your favorite tree? Answer in one word.", "eval_render": "chat_prefill"}
    )
    assert t.endswith(BB.ANSWER_STEM) and e is None


# ------------------------------------------------------------------ viewer data builder


def test_viewer_builder_emits_every_item_and_badge(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts" / "sibling_probe" / "viewer"))
    import build_viewer as BV

    items = [
        {
            "set": "A",
            "id": "it1",
            "prompt_tokens": ["a", "b"],
            "positions": [1],
            "gold": "g",
            "response": "r",
            "prompt": "a b",
            "category": "c",
        },
        {
            "set": "B",
            "id": "it2",
            "prompt_tokens": ["x"],
            "positions": [0],
            "gold": "g2",
            "response": "r2",
            "prompt": "x",
            "category": "c2",
        },
    ]
    readouts = {("A", "it1"): {"s3d": {"1": {"20": {"samples": ["s"]}}}}, ("B", "it2"): {}}
    units = [
        {
            "uid": "A|s3d|it1|20|1|japan",
            "set": "A",
            "item": "it1",
            "method": "s3d",
            "layer": 20,
            "pos": 1,
            "concept": "japan",
            "category": "hallucination",
            "segments": [],
            "probe": {},
            "judge": None,
            "rule": "hallucination",
            "form_label": "assertion",
            "kind": "sibling",
        },
    ]
    idx = BV.write_viewer(items, readouts, units, tmp_path, human_uids={"A|s3d|it1|20|1|japan"})
    assert {e["key"] for e in idx["items"]} == {"A__it1", "B__it2"}
    d = json.loads((tmp_path / "items" / "A__it1.json").read_text())
    assert d["badges"][0]["category"] == "hallucination" and d["badges"][0]["human"]
    assert (tmp_path / "items" / "B__it2.json").exists()
