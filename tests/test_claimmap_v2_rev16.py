"""Plan 0002 rev 16: the lean 12-leaf tree, leaf tags, merges of old gold, false_claim / detail
as separate headline metrics, and the judge-consistency check."""

import asyncio
import json

import pytest
from test_claimmap_v2 import MapFake, _rec, _run, default_map, e, sp

from hallucination_bench import calibrate_v2 as cal
from hallucination_bench import consistency_v2 as cons
from hallucination_bench import report_v2
from hallucination_bench import taxonomy_v2 as tx
from hallucination_bench.cache import Cache
from hallucination_bench.evals.wsbench_chat import run as wrun
from hallucination_bench.evals.wsbench_chat import score_v2
from hallucination_bench.judges import claimmap_v2 as cm
from hallucination_bench.units import split_cell


def test_12_leaves_and_the_verdict_table():
    assert tx.LEAF_NAMES == (
        "non_claim",
        "correct",
        "detail",
        "false_claim",
        "substitution_sibling",
        "substitution_category",
        "model_state",
        "unverifiable",
        "supplementary_true",
        "supplementary_false",
        "ambiguous_topic",
        "off_topic",
    )
    assert tx.VERDICTS == {
        "hallucination": ("detail", "false_claim"),
        "probe_candidate": (
            "substitution_sibling",
            "substitution_category",
            "model_state",
            "supplementary_false",
            "ambiguous_topic",
        ),
        "correct": ("correct", "supplementary_true"),
        "unresolved": ("unverifiable", tx.UNKNOWN_INCORRECT),
        "junk": ("off_topic",),
        "non_claim": ("non_claim",),
    }
    mapped = [lf for leaves in tx.VERDICTS.values() for lf in leaves]
    assert len(mapped) == len(set(mapped))
    assert set(mapped) == {*tx.LEAF_NAMES, tx.UNKNOWN_INCORRECT}


def test_every_old_leaf_maps_to_a_rev16_leaf_with_tags():
    old = {
        "generic",
        "filler",
        "formatting",
        "junk_text",
        "lens_meta",
        "restates_prefix",
        "stated_after",
        "entailed",
        "world_true",
        "possible_belief",
        "other_on_topic",
        "supplementary_unsettled",
        "unsupported_assertion",
        "unrealised_option",
        "unrealised_prediction",
    }
    assert set(tx.LEAF_RENAMES) == old
    assert set(tx.LEAF_RENAMES.values()) <= set(tx.LEAVES)
    assert tx.MERGE_TAGS["stated_after"] == {"source": "after"}
    assert tx.MERGE_TAGS["lens_meta"] == {"reason": "lens_meta"}
    assert tx.MERGE_TAGS["supplementary_unsettled"] == {"world": True}


def test_rev16_anchors_and_tags_in_the_prompt():
    s = cm.SYSTEM
    assert '"RLC in the context of **Android**"' in s
    assert "if it's from a flood or sewage backup" in s
    assert '"my better half is your development environment"' in s
    assert '"the spill has been cleaned up"' in s and '"welcome to Oakville"' in s
    assert '"3. Baking soda" where item 3 is coffee grounds' in s
    assert "Predictions get **no special category**" in s
    assert "Correct beats everything uncontradicted." in s
    for t in ("reason", "source", "same_slot_alternative", "world", "prediction"):
        assert f"- `{t}` " in s
    props = cm.SCHEMA["schema"]["properties"]["units"]["items"]["properties"]
    assert props["reason"]["anyOf"][0]["enum"] == list(tx.REASONS)
    assert props["source"]["anyOf"][0]["enum"] == list(tx.SOURCES)
    for old in ("unrealised_option", "supplementary_unsettled", "restates_prefix", "other_on"):
        assert f"`{old}" not in s  # old leaf names are gone


def test_parse_keeps_leaf_tags_only_on_their_leaf():
    units = split_cell(["Hmm. It was said. Next comes a spill. Maybe so."])[0]
    raw = {
        "units": [
            e("R0.U1", "non_claim", reason="filler", source="after", prediction=True),
            e("R0.U2", "correct", source="prefix", reason="filler"),
            e("R0.U3", "unverifiable", prediction=True, same_slot_alternative=True),
            e("R0.U4", "unverifiable", world=True, reason="generic", source="bogus"),
        ],
        "rationale": "",
    }
    segs, _ = cm.parse(raw, units)
    got = [
        (s.leaf, s.reason, s.source, s.same_slot_alternative, s.world, s.prediction) for s in segs
    ]
    assert got == [
        ("non_claim", "filler", None, False, False, False),
        ("correct", None, "prefix", False, False, False),
        ("unverifiable", None, None, True, False, True),
        ("unverifiable", None, None, False, True, False),
    ]


def test_rest_piece_takes_the_units_tags():
    units = split_cell(["Hmm, it was said there."])[0]
    raw = {
        "units": [
            e(
                "R0.U1",
                "non_claim",
                source="prefix",
                splits=[sp("Hmm,", "non_claim", reason="filler")],
                rest_leaf="correct",
            )
        ],
        "rationale": "",
    }
    segs, _ = cm.parse(raw, units)
    assert [(s.leaf, s.reason, s.source) for s in segs] == [
        ("non_claim", "filler", None),
        ("correct", None, "prefix"),
    ]


def test_old_gold_is_merged_with_tags():
    c = cal.skeleton(
        {
            "case_id": "old",
            "part": "synthetic",
            "rules": [],
            "prompt": "q",
            "response": "a⟦READ⟧ b",
            "readouts": ["The tokens include x and it is cold. Fine."],
        }
    )
    g0, g1 = c["gold"]["units"]
    g0["splits"] = [{"text": "The tokens include", "leaf": "filler"}]
    g0["rest_leaf"] = "stated_after"
    g1["leaf"] = "supplementary_unsettled"
    assert cal.validate_case(c) == []
    m = cal.remap_case(c)
    h0, h1 = m["gold"]["units"]
    assert h0["splits"][0] == {
        "text": "The tokens include",
        "leaf": "non_claim",
        "tags": {"reason": "filler"},
    }
    assert h0["rest_leaf"] == "correct" and h0["rest_tags"] == {"source": "after"}
    assert h1["leaf"] == "unverifiable" and h1["tags"] == {"world": True}


def test_dev_v2_rev16_anchors_and_dev2_36_reverted():
    cases = {c["case_id"]: c for c in cal.load_cases("dev")}
    raw = (cal.CAL_DIR / "dev_v2.jsonl").read_text()
    for old in tx.LEAF_RENAMES:
        assert f'"leaf": "{old}"' not in raw
    assert cases["dev2-36-better-half"]["gold"]["units"][0]["leaf"] == "false_claim"
    assert cases["dev2-42-rlc-android"]["gold"]["units"][0]["leaf"] == "detail"
    assert cases["dev2-43-tobacco-smell"]["gold"]["units"][0]["leaf"] == "false_claim"
    spill = cases["dev2-44-laptop-spill"]["gold"]["units"]
    assert [g["leaf"] for g in spill] == ["unverifiable", "unverifiable", "correct"]
    soda = cases["dev2-45-fridge-odour"]["gold"]["units"][0]
    assert soda["leaf"] == "unverifiable" and soda["tags"]["same_slot_alternative"] is True
    cov = cal.coverage(list(cases.values()))["segments_per_leaf"]
    assert all(n >= 3 for n in cov.values()), cov


# ------------------------------------------------------------------ reporting


def test_false_claim_and_detail_are_headline_metrics(tmp_path):
    recs = [
        _rec("a", 44, [[("false_claim", 30), ("non_claim", 10)]], cls="hallucinated"),
        _rec("b", 44, [[("detail", 5), ("correct", 5)]], cls="consistent"),
        _rec("c", 44, [[("unverifiable", 8)]], cls="consistent"),
    ]
    acc = score_v2.AccumulatorV2()
    for r in recs:
        acc.add(r)
    b = score_v2.block(acc.cells, n_boot=50)
    cmp_ = b["v5c_comparison"]
    assert cmp_["v2_false_claim_rate"]["value"] == pytest.approx(1 / 3)
    assert cmp_["v2_detail_rate"]["value"] == pytest.approx(1 / 3)
    summary = {
        "config": {"kind": "prose", "pre_gate": True, "map_anchor": "none"},
        "overall": b,
        "by_layer": {"44": b},
        "health": {
            **b["health"],
            "failed_cell_rate": 0.0,
            "unlocated_subspan_rate": None,
            "missing_probe_target_rate": None,
        },
        "spend_map": {"usd": 0.0},
    }
    md, _ = report_v2.build({"toy": summary}, "m")
    head = next(line for line in md.splitlines() if line.startswith("| arm | n readouts | hall"))
    assert "↳ false_claim | ↳ detail" in head
    share_row = md.split("Headline: character-weighted verdict shares, per arm")[1].splitlines()
    toy = next(line for line in share_row if line.startswith("| toy"))
    # hallucination 0.25 [..] then false_claim 0.25 then detail 0.1667, all with CIs
    cells = [c.strip() for c in toy.split("|")[1:-1]]
    assert cells[2].startswith("41.7") and cells[3].startswith("25.0") and "[" in cells[3]
    assert cells[4].startswith("16.7") and "[" in cells[4]
    assert "↳ v2 false_claim" in md and "↳ v2 detail" in md


# ------------------------------------------------------------------ consistency


def test_kappa_agreement_and_flagged_pairs():
    pairs = [("correct", "correct")] * 8 + [("unverifiable", "correct")] * 2
    pairs += [("unverifiable", "unverifiable")] * 10
    res = cons.agreement(pairs)
    assert res["n_units"] == 20 and res["raw_agreement"] == pytest.approx(0.9)
    assert res["kappa"] == pytest.approx(0.8)
    assert res["confusion"]["unverifiable|correct"] == 2
    sa = res["self_agreement"]["correct"]
    assert sa["n_r0"] == 8 and sa["n_r1"] == 10 and sa["dice"] == pytest.approx(16 / 18)
    (flag,) = res["flagged_pairs"]
    assert flag["pair"] == ["correct", "unverifiable"] and flag["n_swaps"] == 2
    assert flag["share_a"] == pytest.approx(0.2) and flag["share_b"] == pytest.approx(2 / 12)
    assert cons.cohen_kappa([]) is None
    assert cons.parse_arms("s3d-rl600:44, jlens:44") == [("s3d-rl600", 44), ("jlens", 44)]


def test_consistency_makes_two_fresh_calls_per_cell_and_caches(tmp_path, prose_readouts):
    out = tmp_path / "out"
    _run([("toy", prose_readouts)], out, MapFake(), stages=["map"])  # writes arm dir + cache
    arm = wrun.open_arm("toy", prose_readouts, out, wrun.Models(), readonly=True)
    picked = cons.sample_cells(arm, 44, 3, 0, wrun.Models())
    assert len(picked) == 3
    assert picked == cons.sample_cells(arm, 44, 3, 0, wrun.Models())  # seeded
    work = [("toy", arm, c, s) for c, s in picked]
    fake = MapFake({"map": default_map})

    def go(llm):
        with Cache(tmp_path / "cons.jsonl") as cache:
            return asyncio.run(
                cons.map_twice(work, models=wrun.Models(), cache=cache, llm=llm, usage=cons.Usage())
            )

    cells = go(fake)
    assert fake.n("map") == 6  # 3 cells x 2 repeats
    keys = sorted(json.loads(x)["key"] for x in (tmp_path / "cons.jsonl").read_text().splitlines())
    assert keys[0].startswith("cons:toy:") and keys[0].endswith(":r0")
    assert {k[-2:] for k in keys} == {"r0", "r1"}
    fake2 = MapFake()
    assert go(fake2) == cells and fake2.n("map") == 0
    res = cons.score(cells)
    assert res["overall"]["raw_agreement"] == 1.0 and res["n_cells"] == 3
    md = cons.render_markdown(res)
    assert "κ" in md and "Between-run confusion" in md and "Flagged pairs" in md
    arm.cache.close()
