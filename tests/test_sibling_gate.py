"""Plan 0009 (scoring rev 21): the sibling gate. Synthetic cells only; no LLM / Modal calls."""

import asyncio
import json
from pathlib import Path

import pytest
from fixtures import score_v2_rev20 as rev20

from hallucination_bench import report_v2
from hallucination_bench import taxonomy_v2 as tx
from hallucination_bench.cache import Cache
from hallucination_bench.evals.sibling_gate import fill_judge as F
from hallucination_bench.evals.sibling_gate import ladder as L
from hallucination_bench.evals.sibling_gate import resample as R
from hallucination_bench.evals.sibling_gate import texts as T
from hallucination_bench.evals.sibling_gate import units as sibling_units
from hallucination_bench.evals.wsbench_chat import score_v2
from hallucination_bench.llm import CallResult, Usage

SIB = "substitution_sibling"


# ---------------------------------------------------------------- synthetic cells


def seg(readout, start, n, leaf, **kw):
    d = {
        "unit": f"R{readout}.U{start}",
        "readout": readout,
        "start": start,
        "end": start + n,
        "text": "x" * n,
        "leaf": leaf,
        "raw_leaf": leaf,
        "status": "ok",
        "established": False,
        "kind": "unit",
        "alt_leaf": None,
        "hedged": False,
        "world_false": False,
        "cruxy": False,
        "dispute": False,
        "pair": None,
        "why": "",
        "contradicted_by": None,
    }
    d.update(kw)
    return d


def rec(item, layer, pos, segs, n_readouts=3, cls="consistent", quotes=None):
    key = sibling_units.cell_key(item, layer, pos)
    return {
        "key": key,
        "id": item,
        "layer": layer,
        "pos": pos,
        "status": "judged",
        "stages": {"map": "ok"},
        "samples": ["Google built me. Claude. OpenAI made me." + " " * 60] * n_readouts,
        "verdict": [{"idx": i, "class": cls, "quotes": quotes} for i in range(n_readouts)],
        "map": {"segments": segs, "counts": {"n_units": len(segs)}},
    }


def pair(said, actual="the engineers"):
    return {"pair": {"said": said, "actual": actual}}


def cells():
    """Sibling units (one repeated across readouts), a disputed raw sibling, a disputed segment
    whose alt is a sibling, category substitutions and correct content."""
    return [
        rec(
            "a",
            44,
            10,
            [
                seg(0, 0, 10, SIB, **pair("Google")),
                seg(0, 11, 20, "correct"),
                seg(1, 0, 10, SIB, **pair("google.")),  # same unit (normalised said)
                seg(1, 11, 5, "substitution_category", **pair("Paris", "x")),
                seg(2, 0, 8, "correct"),
            ],
            quotes={"true": ["Claude"], "disputed": ["Google built"]},
        ),
        rec(
            "a",
            48,
            10,  # same pair at another layer: another unit
            [seg(0, 0, 10, SIB, **pair("Google")), seg(1, 0, 4, "detail")],
            n_readouts=2,
        ),
        rec(
            "b",
            44,
            22,
            [
                seg(0, 0, 6, SIB, status="disputed", alt_leaf="false_claim", **pair("OpenAI")),
                seg(0, 7, 6, "unverifiable", status="disputed", alt_leaf=SIB),
                seg(1, 0, 9, SIB, **pair("OpenAI")),
                seg(1, 10, 9, "junk", reason="filler"),
            ],
            n_readouts=2,
            cls="hallucinated",
        ),
        rec("c", 44, 5, [seg(0, 0, 12, "false_claim"), seg(0, 13, 3, SIB, **pair("Bing"))], 1),
    ]


def labels_for(recs, final_of):
    """{segkey: final} with ``final_of(unit) -> final``, via the production unit builder."""
    out = {}
    for r in recs:
        for u in sibling_units.units_of_cell("arm", r):
            for s in u["segments"]:
                out[(s["key"], s["readout"], s["start"], s["end"])] = final_of(u)
    return out


# ---------------------------------------------------------------- taxonomy (rev 21)


def test_verdicts_unchanged_and_gated_leaf_only_in_scoring_maps():
    assert tx.VERDICTS[tx.V_HALLUCINATION] == ("detail", "false_claim", "substitution_category")
    assert SIB in tx.PROBE_CANDIDATE_LEAVES and tx.SIBLING_GATED not in tx.ALL_LEAVES
    assert tx.verdict_of(SIB) == tx.V_PROBE
    assert tx.verdict_of(tx.SIBLING_GATED) == tx.V_HALLUCINATION
    assert tx.group_of(tx.SIBLING_GATED) == tx.group_of(SIB)
    assert score_v2.LEAF_ROWS == (*tx.ALL_LEAVES, tx.SIBLING_GATED)
    assert score_v2.HALLUCINATION_COMPONENTS == (
        "false_claim",
        "detail",
        "substitution_category",
        tx.SIBLING_GATED,
    )
    assert tx.SIBLING_GATED in score_v2.HALLUCINATION_LEAVES
    assert tx.SIBLING_GATED in score_v2.FLAGS and "n_cmp_v2_gated" in score_v2.COLUMNS


def test_effective_leaf():
    assert tx.effective_leaf(SIB, "unstable") == tx.SIBLING_GATED
    assert tx.effective_leaf(SIB, "probe_unconfirmed") == tx.SIBLING_GATED
    for st in ("probe_faithful", "probe_nearby", "cross_method", "probe_na", "resample_failed"):
        assert tx.effective_leaf(SIB, st) == SIB
    assert tx.effective_leaf(SIB, None) == SIB
    assert tx.effective_leaf("false_claim", "unstable") == "false_claim"


def test_hedged_origin_keeps_testing_the_raw_rev20_components():
    s = {"hedged": True, "leaf": "unverifiable", "raw_leaf": "false_claim"}
    assert score_v2.hedged_origin(s) == "hallucination"
    assert score_v2.hedged_origin({**s, "raw_leaf": SIB}) == "substitution"


# ---------------------------------------------------------------- units


def test_units_normalise_group_and_skip_disputed_and_alt_only():
    assert sibling_units.norm(" Google. ") == "google" and sibling_units.norm("...") == "..."
    assert sibling_units.norm("ＧＯＯＧＬＥ") == "google"  # NFKC + casefold
    us = [u for r in cells() for u in sibling_units.units_of_cell("arm", r)]
    keys = sorted((u["cell"], u["said_norm"], len(u["segments"])) for u in us)
    assert keys == [
        ("a__L044__p10", "google", 2),  # repeats across readouts = one unit
        ("a__L048__p10", "google", 1),  # another layer = another unit
        ("b__L044__p22", "openai", 1),  # the disputed raw sibling forms no unit
        ("c__L044__p5", "bing", 1),
    ]
    names = sibling_units.item_sibling_names(us)
    assert names["a"]["google"]["layers"] == {44, 48}


def test_load_labels_and_family(tmp_path):
    p = tmp_path / "arm.jsonl"
    p.write_text(
        json.dumps(
            {"final": "unstable", "segments": [{"key": "k", "readout": 0, "start": 1, "end": 4}]}
        )
        + "\n"
    )
    assert sibling_units.load_labels("arm", tmp_path) == {("k", 0, 1, 4): "unstable"}
    assert sibling_units.load_labels("none", tmp_path) is None
    assert sibling_units.family_of("s3d-rl600-all11") == sibling_units.family_of("s3d-sft251")
    assert sibling_units.family_of("mlnla-rl600-all11") != sibling_units.family_of("nla-rl-L42")
    assert sibling_units.family_of("jlens") == "jlens"


# ---------------------------------------------------------------- the label join (score_v2)


def test_label_join_alt_view_and_null_final():
    recs = cells()
    lab = labels_for(recs, lambda u: "unstable" if u["said_norm"] != "bing" else None)
    by = {r["key"]: score_v2.slim(r, lab) for r in recs}
    a = by["a__L044__p10"].readouts
    assert a[0].segs[0][:2] == (tx.SIBLING_GATED, tx.SIBLING_GATED)  # ok segment: alt gated too
    assert a[1].segs[1][0] == "substitution_category"
    b = by["b__L044__p22"].readouts
    assert b[0].segs[0][:2] == (SIB, "false_claim")  # disputed raw sibling stays raw
    assert b[0].segs[1][:2] == ("unverifiable", SIB)  # alt-only sibling stays raw
    assert b[1].segs[0][0] == tx.SIBLING_GATED
    c = by["c__L044__p5"].readouts
    assert c[0].segs[1][0] == SIB  # null final keeps the raw leaf
    # agreement spans stay raw
    raw = score_v2.slim(recs[0])
    assert by["a__L044__p10"].agreement == raw.agreement


def test_no_labels_summary_is_identical_to_rev20():
    recs = cells()
    acc20 = rev20.AccumulatorV2()
    acc21 = score_v2.AccumulatorV2()
    acc_empty = score_v2.AccumulatorV2(sibling_labels={})
    for r in recs:
        acc20.add(r)
        acc21.add(r)
        acc_empty.add(r)
    cfg = {"x": 1}
    s20 = rev20.summarize(acc20, config=cfg, spend={}, n_boot=50)
    for s21 in (
        score_v2.summarize(acc21, config=cfg, spend={}, n_boot=50),
        score_v2.summarize(acc_empty, config=cfg, spend={}, n_boot=50),
    ):

        def same(x, y, path=""):
            if isinstance(x, dict):
                for k, v in x.items():
                    assert k in y, path + "/" + str(k)
                    same(v, y[k], path + "/" + str(k))
            else:
                assert x == y, path

        same(s20, s21)
        assert s21["overall"]["char_share_leaf"][tx.SIBLING_GATED]["value"] == 0
        assert s21["overall"]["rates"][tx.SIBLING_GATED]["all"]["value"] == 0


def test_labels_move_siblings_to_hallucination_and_alt_never_below():
    recs = cells()
    lab = labels_for(recs, lambda u: "unstable")
    acc0, acc = score_v2.AccumulatorV2(), score_v2.AccumulatorV2(sibling_labels=lab)
    for r in recs:
        acc0.add(r)
        acc.add(r)
    for c in acc.cells:
        for r in c.readouts:
            f = score_v2.readout_flags(r)
            assert f["hallucination_alt"] >= f["hallucination"]
    s0 = score_v2.summarize(acc0, config={}, spend={}, n_boot=50)["overall"]
    s1 = score_v2.summarize(acc, config={}, spend={}, n_boot=50)["overall"]
    h0 = s0["char_share_verdict"]["hallucination"]["value"]
    h1 = s1["char_share_verdict"]["hallucination"]["value"]
    assert h1 > h0
    assert s1["char_share_leaf"][tx.SIBLING_GATED]["value"] > 0
    # raw sibling row is net of gated; group shares unchanged
    assert s1["char_share_leaf"][SIB]["value"] < s0["char_share_leaf"][SIB]["value"]
    for g, v in s0["char_share_group"].items():
        assert s1["char_share_group"][g]["value"] == pytest.approx(v["value"])
    assert (
        s1["rates"]["hallucination_alt"]["all"]["value"]
        >= (s1["rates"]["hallucination"]["all"]["value"])
    )
    assert s1["v5c_comparison"]["v2_substitution_sibling_gated_rate"]["value"] > 0
    assert s1["agreement"] == s0["agreement"]


# ---------------------------------------------------------------- rung 1


def ans(*kinds):
    out = []
    for i, k in enumerate(kinds, 1):
        a, objs = (k.split(":", 1)[0], [k.split(":", 1)[1]]) if ":" in k else (k, [])
        out.append({"block": i, "answer": a, "objects": objs})
    return out


def test_rung1_rule_branches():
    c = F.counts(ans("other:Canada", "other:Russia", "other:UK", *["none"] * 7))
    assert F.rung1(c) == {"status": "unstable", "branch": "A", "diverse": True}
    c = F.counts(ans(*["actual"] * 4, "said", *["none"] * 5))
    assert F.rung1(c)["status"] == "unstable" and F.rung1(c)["branch"] == "B"
    c = F.counts(ans(*["actual"] * 4, "other:a", "other:b", *["none"] * 4))
    assert F.rung1(c)["branch"] == "both"
    c = F.counts(ans("said", "said", "said", *["actual"] * 7))
    assert F.rung1(c)["status"] == "stable"
    assert F.rung1(F.counts(ans(*["none"] * 10)))["status"] == "sparse"
    # one repeated other object: sparse
    c = F.counts(ans(*["other:Canada"] * 5, *["none"] * 5))
    assert c["n_other"] == 1 and F.rung1(c)["status"] == "sparse"


def test_rung1_sensitivity_rows():
    # n_said 2 + 1 actual + 1 other block: rev-1 unstable (2+1+1 >= 4), primary sparse
    c = F.counts(ans("said", "said", "actual", "other:x", *["none"] * 6))
    assert F.rung1(c)["status"] == "sparse"
    s = F.sensitivity_unstable(c)
    assert s["rung1_rev1"] and not s["rung1_branch_a_only"]
    # branch-B-only unstable: (ii) sends it to rung 2
    c = F.counts(ans(*["actual"] * 5, *["none"] * 5))
    s = F.sensitivity_unstable(c)
    assert F.rung1(c)["status"] == "unstable" and not s["rung1_branch_a_only"]
    # n_said = 2: unstable, but not under (iii)
    c = F.counts(ans("said", "said", "other:a", "other:b", *["none"] * 6))
    s = F.sensitivity_unstable(c)
    assert F.rung1(c)["status"] == "unstable" and not s["rung1_said_le1"]
    # n_said = 3: stable, but unstable under (iv)
    c = F.counts(ans("said", "said", "said", "other:a", "other:b", *["none"] * 5))
    s = F.sensitivity_unstable(c)
    assert F.rung1(c)["status"] == "stable" and s["rung1_said_le3"]


def test_needs_rung2_includes_sensitivity_only_units():
    def r1(*kinds):
        u = {"unit": "u", "arm": "s3d-rl600"}
        return F.rung1_row(u, "ok", {"answers": ans(*kinds), "n_failed_quotes": 0})

    assert F.needs_rung2(r1(*["none"] * 10))  # sparse
    assert F.needs_rung2(r1("said", "said", "other:a", "other:b", *["none"] * 6))  # n_said 2
    assert F.needs_rung2(r1(*["actual"] * 5, *["none"] * 5))  # branch B only
    assert not F.needs_rung2(r1("other:a", "other:b", "other:c", *["none"] * 7))  # A, n_said 0
    assert F.needs_rung2({"rung1": "deterministic"})
    assert not F.needs_rung2({"rung1": "resample_failed"})


def test_sensitivity_rows_use_rung2_or_probe_na():
    r1 = F.rung1_row(
        {"unit": "u", "arm": "a"},
        "ok",
        {"answers": ans("said", "said", "actual", "other:x", *["none"] * 6), "n_failed_quotes": 0},
    )
    out = L.sensitivity(r1, {"status": "probe_faithful"}, "probe_faithful", {})
    assert out["rung1_rev1"] == "unstable" and out["rung1_branch_a_only"] == "probe_faithful"
    r1u = F.rung1_row(
        {"unit": "u", "arm": "a"},
        "ok",
        {"answers": ans("other:a", "other:b", *["none"] * 8), "n_failed_quotes": 0},
    )
    out = L.sensitivity(r1u, None, "unstable", {})
    assert out["rung1_branch_a_only"] == "unstable"
    # (i): n_said 0 + 2 other blocks < 4 -> moved to rung 2, which never ran -> probe_na
    assert out["rung1_rev1"] == "probe_na"


# ---------------------------------------------------------------- slot-fill judge


def test_fill_validate_parse_and_quote_check():
    blocks = ["- Google built me\n- I am an AI", "- the team that made me", "- nothing here"]
    raw = {
        "blocks": [
            {"block": 1, "answer": "said", "objects": [], "quote": "google built me"},
            {"block": 2, "answer": "actual", "objects": [], "quote": "not in the block"},
            {"block": 3, "answer": "none", "objects": [], "quote": ""},
        ],
        "note": "",
    }
    assert F.validate(raw, 3) and not F.validate(raw, 4)
    bad = {**raw, "blocks": [*raw["blocks"][:2], {**raw["blocks"][2], "answer": "other"}]}
    assert not F.validate(bad, 3)  # other without an object
    p = F.parse(raw, blocks)
    assert [a["answer"] for a in p["answers"]] == ["said", "none", "none"]
    assert p["n_failed_quotes"] == 1


def test_fill_stage_uses_the_cache_and_fails_to_resample_failed(tmp_path, monkeypatch):
    item = {
        "id": "a",
        "prompt": "Who made you?",
        "response": "The engineers at Alibaba made me.",
        "sites": [{"pos": 10, "char": 4, "token": " engineers"}],
    }
    u = sibling_units.units_of_cell("s3d-rl600", cells()[0])[0]
    blocks = [f"- Google block {i}" for i in range(10)]
    calls = []

    class Fake:
        async def call(self, *, model, system, user, schema):
            calls.append(user)
            return CallResult({"blocks": [], "note": ""}, Usage())  # always invalid

    async def go(llm):
        with Cache(tmp_path / "c.jsonl") as cache:
            return await F.run_fill(
                [u, {**u, "unit": "tok", "arm": "jlens"}],
                {"s3d-rl600": {("a", 44, 10): blocks}},
                {"a": item},
                llm,
                cache,
            )

    rows, _ = asyncio.run(go(Fake()))
    assert len(calls) == 1 + F.INVALID_RETRIES
    assert {r["rung1"] for r in rows} == {"resample_failed", "deterministic"}
    assert "=== block 10 ===" in calls[0] and "Google" in calls[0]
    rows, _ = asyncio.run(go(None))  # cache replay, no calls
    assert rows[0]["rung1"] == "resample_failed" and len(calls) == 3


# ---------------------------------------------------------------- resample


def test_requests_pools_and_overgeneration():
    us = [
        {"arm": "s3d-rl600", "id": "a", "layer": 44, "pos": 1},
        {"arm": "s3d-rl600-all11", "id": "a", "layer": 44, "pos": 1},  # same pool + cell
        {"arm": "s3d-rl600-all11", "id": "a", "layer": 20, "pos": 7},
        {"arm": "jlens", "id": "a", "layer": 44, "pos": 1},  # deterministic: no pool
    ]
    r = R.requests(us)["s3d-rl600"]
    assert set(R.requests(us)) == {"s3d-rl600"}
    assert r.n_requested == 2 and r.layers == [20, 44] and r.positions() == {"a": [1, 7]}
    assert r.n_generated == 4  # 2 positions x 2 layers


def test_trimmed_manifest_copies_and_per_entry_positions():
    orig = {
        "model_id": "m",
        "layers": [42],
        "prompts": [
            {
                "label": "a",
                "family": "f",
                "file": "/acts/acts-x-L42/a.safetensors",
                "n_pos": 50,
                "tokens": [],
                "targets": [],
                "look_for": "",
                "eval_positions": [1, 2, 3],
                "units": None,
                "contract": None,
            },
            {
                "label": "b",
                "family": "f",
                "file": "/acts/acts-x-L42/b.safetensors",
                "n_pos": 9,
                "tokens": [],
                "targets": [],
                "look_for": "",
                "eval_positions": [5],
            },
        ],
    }
    man, copies = R.trimmed_manifest(orig, R.POOLS["nla-rl"], {"a": [3, 40]})
    assert [e["label"] for e in man["prompts"]] == ["a"] and man["layers"] == [42]
    assert man["prompts"][0]["eval_positions"] == [3, 40]
    assert man["prompts"][0]["file"] == "a.safetensors"
    assert copies == [("/acts-x-L42/a.safetensors", "/acts-sibgate-nla-rl/a.safetensors")]
    assert R.check_manifest(man, [42]) == []
    assert R.check_manifest(man, [44])  # no plane
    assert R.expected_keys(man, [42]) == {("a", 42, 3), ("a", 42, 40)}
    with pytest.raises(ValueError):
        R.trimmed_manifest(orig, R.POOLS["nla-rl"], {"b": [9]})


def test_collect_keeps_requested_keys_and_block_shape():
    rows = [
        {"id": "a", "layer": 44, "pos": 1, "samples": [f"- s{i}" for i in range(10)]},
        {"id": "a", "layer": 48, "pos": 1, "samples": ["- z"] * 10},  # over-generated
        {"id": "b", "layer": 44, "pos": 2, "samples": ["- y"] * 4},  # short
    ]
    out, st = R.collect(rows, [("a", 44, 1), ("b", 44, 2), ("c", 44, 3)], {"k": 10})
    assert [(r["id"], r["layer"]) for r in out] == [("a", 44)] and out[0]["config"] == {"k": 10}
    assert st["n_dropped_overgenerated"] == 1 and st["n_short"] == 1 and st["n_missing"] == 1
    assert not st["keys_equal_request"]
    orig = ["- a b c\n- d e f\n- g"] * 20
    assert R.block_shape_check(orig, ["- a b c\n- d e f\n- h"] * 20)["ok"]
    assert not R.block_shape_check(orig, ["one long line " * 10] * 20)["ok"]


# ---------------------------------------------------------------- texts / token class


def test_punctuation_token_class_key():
    assert T.punct_class(".") == T.PUNCT_CLASSES.index(".")
    assert T.punct_class(" ,") == T.PUNCT_CLASSES.index(",")
    assert T.punct_class(".\n\n") == T.PUNCT_CLASSES.index(".")
    assert T.punct_class("\n\n") == T.PUNCT_CLASSES.index("\n")
    assert T.punct_class("。") == T.PUNCT_CLASSES.index("。")
    assert T.punct_class(' "') == T.OTHER and T.punct_class("—") == T.OTHER
    assert T.is_punct_token(" ,") and T.is_punct_token("\n") and not T.is_punct_token(" the")
    assert not T.is_punct_token("  ")
    merge = T.merge_small({0: 60, 1: 10})
    assert merge[0] == 0 and merge[1] == T.OTHER and merge[T.OTHER] == T.OTHER
    ending, off = T.class_key([".", ";", "!"], merge)
    assert ending == [0, 0, 0] and off == [0, T.OTHER, T.OTHER]
    toks = ["Hi", ",", " Alibaba", " made", " me", ".", " Yes", "!"]
    starts = [0, 2, 3, 11, 16, 19, 20, 24]
    assert T.read_positions(toks, starts, reply_start=0, mention_char=3) == [5, 7]
    assert T.read_positions(toks, starts, 0, 0, max_n=1) == [1]


def test_text_check_naming_ngrams_duplicates():
    namer = T.Namer({"ChatGPT": ["OpenAI's ChatGPT"], "Claude": []})
    idx = T.item_ngram_index([{"prompt": "one two three four five six seven eight nine"}])
    seen = set()
    assert T.text_check(namer, "ChatGPT", "explicit", "I like ChatGPT.", idx, seen) is None
    assert T.text_check(namer, "ChatGPT", "explicit", "ChatGPT or Claude", idx, seen).startswith(
        "names_other"
    )
    assert T.text_check(namer, "ChatGPT", "implied", "the bot by OpenAI", idx, seen) is None
    assert T.text_check(namer, "ChatGPT", "implied", "chatgpt!", idx, seen).startswith("names")
    assert T.text_check(namer, None, "neutral", "Claude", idx, seen).startswith("names")
    long = "so one two three four five six seven eight ok"
    assert T.text_check(namer, None, "neutral", long, idx, seen) == "ngram8"
    seen.add(T.normalize_text("hello there"))
    assert T.text_check(namer, None, "neutral", "Hello  there", idx, seen) == "duplicate"


# ---------------------------------------------------------------- ladder


CONCEPTS = [f"c{i}" for i in range(12)]


def probes(z_at, valid=True, null_ok=True):
    """A class whose z table is ``z_at(item, layer, pos) -> {concept: z}`` (default 0)."""
    cp = L.ClassProbes("K", list(CONCEPTS))
    cp.z = L.ZTable(CONCEPTS)
    for la in L.PROBE_LAYERS:
        cp.null_ok[la] = null_ok
        for c in CONCEPTS:
            cp.validity[(c, la)] = 0.95 if valid else 0.5
        for item in ("i", "j"):
            for p in (1, 2, 3, 4):
                zz = z_at(item, la, p)
                cp.z.add(item, la, p, [zz.get(c, 0.0) for c in CONCEPTS])
    return cp


def unit(gold="c1", layer=44, arm="s3d-rl600-all11"):
    return L.UnitProbe("u", arm, "i", layer, 2, "K", "c0", gold, [f"c{i}" for i in range(2, 10)])


def gold_readable(item, la, p):
    return {"c1": 5.0}


def test_neighbours():
    assert L.neighbours(44) == (40, 42, 48)
    assert L.neighbours(40) == (36, 42, 44)
    assert L.neighbours(42) == (40, 44)
    assert L.neighbours(20) == (24,) and L.neighbours(60) == (56,)


def test_fire_nearby_and_strict_precedence():
    def fire_at(la):
        return lambda i, L_, p: {"c1": 5.0, **({"c0": 9.0} if (L_ == la and p == 2) else {})}

    idx = {("i", "c0"): [("nla-rl-L42", 44, 3)]}  # another family names S
    assert L.rung2(unit(), probes(fire_at(44)), idx)["status"] == "probe_faithful"
    assert L.rung2(unit(), probes(fire_at(48)), idx)["status"] == "probe_nearby"
    assert L.rung2(unit(), probes(fire_at(52)), idx)["status"] == "cross_method"
    assert L.rung2(unit(), probes(fire_at(52)), {})["status"] == "probe_unconfirmed"
    # n/a first: never cross_method / unconfirmed
    for cp in (probes(fire_at(52), valid=False), probes(fire_at(52), null_ok=False), None):
        r = L.rung2(unit(), cp, idx)
        assert r["status"] == "probe_na"
    unread = probes(lambda i, la, p: {})  # the gold never beats its distractors
    assert L.rung2(unit(), unread, idx) == {
        **L.rung2(unit(), unread, idx),
        "status": "probe_na",
        "na_reason": "unreadable",
    }
    small = probes(fire_at(52))
    small.concepts = CONCEPTS[:8]
    assert L.rung2(unit(), small, idx)["na_reason"] == "small_class"


def test_no_gold_never_probe_unconfirmed():
    idx = {("i", "c0"): [("jlens", 44, 3)]}
    for z in (lambda i, la, p: {}, lambda i, la, p: {"c1": 5.0}):
        for ix in (idx, {}):
            r = L.rung2(unit(gold=None), probes(z), ix)
            assert r["status"] == "probe_na" and r["no_gold"]
    fired = L.rung2(unit(gold=None), probes(lambda i, la, p: {"c0": 9.0}), {})
    assert fired["status"] == "probe_faithful"


def test_l42_token_mismatch_disables_nearby():
    def fire44(i, la, p):
        return {"c1": 5.0, **({"c0": 9.0} if la == 44 else {})}

    u = unit(layer=42, arm="nla-rl-L42")
    assert L.rung2(u, probes(fire44), {})["status"] == "probe_nearby"
    r = L.rung2(u, probes(fire44), {}, l42_tokens_ok=False)
    assert r["status"] == "probe_na" and r["na_reason"] == "l42_token_mismatch"


def test_cross_method_family_exclusion():
    u = unit()
    idx = {
        ("i", "c0"): [
            ("s3d-rl600", 44, 2),  # same family: excluded
            ("s3d-rl600-all11", 44, 2),  # the unit's own arm
            ("mlnla-rl600-all11", 52, 2),  # different family, but layer 52 > 44 + 4
        ]
    }
    assert L.cross_arms(u, idx) == []
    assert L.cross_arms(u, idx, same_family=True) == ["s3d-rl600"]
    idx[("i", "c0")].append(("mlnla-rl600-all11", 48, 4))
    assert L.cross_arms(u, idx) == ["mlnla-rl600-all11"]
    assert L.cross_arms(u, idx, same_pos=True) == []
    assert L.cross_arms(u, idx, discounted={"mlnla-rl600-all11"}) == []


def test_distractors_exclusions_and_determinism():
    cent = {c: [float(i), 1.0] for i, c in enumerate(CONCEPTS)}
    excl = {"c0", "c2", "c3"}
    a = L.select_distractors("u1", "c1", CONCEPTS, cent, excl)
    assert len(a["distractors"]) == 8 and not (set(a["distractors"]) & (excl | {"c1"}))
    assert a == L.select_distractors("u1", "c1", CONCEPTS, cent, excl)
    assert L.distractors_sha256({"u1": a}) == L.distractors_sha256({"u1": dict(a)})


def test_final_status_and_label_row():
    r1 = {"rung1": "stable", "counts": {"n_said": 5}, "branch": None, "diverse": False}
    assert L.final_status(r1, None) is None  # rung 2 unfinished: raw leaf kept
    assert L.final_status(r1, {"status": "probe_unconfirmed"}) == "probe_unconfirmed"
    assert L.final_status({"rung1": "unstable"}, {"status": "probe_faithful"}) == "unstable"
    u = sibling_units.units_of_cell("arm", cells()[0])[0]
    row = L.label_row(u, {**r1, "rung1": "unstable", "branch": "A", "diverse": True}, None, None)
    assert row["final"] == "unstable" and row["flags"]["diverse"]
    assert row["segments"][0] == {"key": "a__L044__p10", "readout": 0, "start": 0, "end": 10}


# ---------------------------------------------------------------- report


def test_report_consumers_and_sibling_section(tmp_path):
    d = tmp_path / "wsbench_chat" / "m" / "arm"
    d.mkdir(parents=True)
    recs = cells()
    with (d / "cells.jsonl").open("w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    ld = tmp_path / "labels"
    us = [u for r in recs for u in sibling_units.units_of_cell("arm", r)]
    rows = [
        L.label_row(u, {"rung1": "unstable", "branch": "A", "diverse": True}, None, None)
        if u["said_norm"] == "google"
        else L.label_row(u, {"rung1": "stable"}, {"status": "probe_faithful"}, None)
        for u in us
    ]
    L.write_labels(rows, ld)
    raw = report_v2.probe_candidates([d])
    gated = report_v2.probe_candidates([d], labels_dir=ld)
    n_google = sum(len(u["segments"]) for u in us if u["said_norm"] == "google")
    assert len(raw) - len(gated) == n_google
    assert all(r["leaf"] in tx.PROBE_CANDIDATE_LEAVES for r in gated)
    assert {r["sibling_status"] for r in gated if r["leaf"] == SIB} >= {"probe_faithful"}
    ex = "\n".join(report_v2.hallucination_examples([d], labels_dir=ld))
    assert tx.SIBLING_GATED in ex
    sec = "\n".join(report_v2.sibling_gate_section(ld, ["arm"]))
    assert "Sibling units by final status" in sec and "Sensitivity rows" in sec
    assert report_v2.sibling_gate_section(tmp_path / "nolabels", ["arm"]) == []


def test_vendored_probe_modules_are_verbatim():
    src = Path("/workspace/hb-um-ladder/src/hallucination_bench/evals/sibling_probe")
    if not src.exists():
        pytest.skip("camila/um-ladder-v2 checkout absent")
    here = Path(__file__).resolve().parents[1] / "src/hallucination_bench/evals/sibling_gate"
    assert (here / "probes.py").read_text() == (src / "probes.py").read_text()
    a = (here / "um_probe.py").read_text().splitlines()
    b = (src / "um_probe.py").read_text().splitlines()
    diff = [(x, y) for x, y in zip(a, b, strict=True) if x != y]
    assert diff == [
        (
            "    from hallucination_bench.evals.sibling_gate import probes as P",
            "    from hallucination_bench.evals.sibling_probe import probes as P",
        )
    ]
