"""Plan 0003 stage 2: probes, judge glue, scoring, verdict, report (synthetic data only)."""

import asyncio
import inspect
import json

import numpy as np
import pytest

from hallucination_bench import taxonomy_v2 as tx
from hallucination_bench.cache import Cache, fingerprint
from hallucination_bench.evals.sibling_probe import judge as J
from hallucination_bench.evals.sibling_probe import probes as P
from hallucination_bench.evals.sibling_probe import score as S
from hallucination_bench.evals.sibling_probe.concepts import load_inventory
from hallucination_bench.judges import claimmap_v2 as cm
from hallucination_bench.judges.v5c import READ_MARK
from hallucination_bench.readouts import Cell

# ------------------------------------------------------------------ synthetic probe data


def make_texts(C=4, n=24, d=64, signal=3.0, n_neutral=12, seed=0, lens=(5, 7)):
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(C, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    tmpl = {e: rng.normal(size=(L, d)) * 5 for e, L in enumerate(lens)}
    meta, wins = [], []
    for c in [*range(C), P.NEUTRAL]:
        for j in range(n if c >= 0 else n_neutral):
            e = j % len(lens)
            w = tmpl[e] + rng.normal(size=tmpl[e].shape)
            if c >= 0:
                w = w + signal * dirs[c]
            kind = "neutral" if c < 0 else ("implied" if (j // 2) % 2 == 0 else "explicit")
            meta.append(
                {"id": f"t{c}_{j}", "concept": None if c < 0 else c, "request_idx": e, "kind": kind}
            )
            wins.append(w)
    folds = P.assign_folds(meta, salt="cat")
    ts = P.TextSet.build(
        wins,
        [m["id"] for m in meta],
        [P.NEUTRAL if m["concept"] is None else m["concept"] for m in meta],
        [m["kind"] for m in meta],
        [m["request_idx"] for m in meta],
        [folds[m["id"]] for m in meta],
    )
    return ts, dirs, meta


def test_diffmean_recovers_planted_direction():
    ts, dirs, _ = make_texts(C=4, d=64, signal=4.0)
    cf = P.cross_fit(ts, 4)
    for mdl in cf.models:
        # d_c = m_c - mean of the others = (dir_c - mean other dirs) in expectation
        want = dirs - (dirs.sum(0, keepdims=True) - dirs) / 3
        cos = np.sum(mdl.U * want / np.linalg.norm(want, axis=1, keepdims=True), axis=1)
        assert np.all(cos > 0.9), cos
    va = P.validity(ts, cf.oof, 4)
    assert np.all(va > 0.95)


def test_assign_folds_balanced_seeded_and_includes_neutral():
    _, _, meta = make_texts()
    f1 = P.assign_folds(meta, salt="cat")
    f2 = P.assign_folds(meta, salt="cat")
    assert f1 == f2
    assert set(f1) == {m["id"] for m in meta}
    assert any(f1[m["id"]] is not None for m in meta if m["concept"] is None)
    for c in [0, 1, 2, 3, None]:
        for e in (0, 1):
            ids = [m["id"] for m in meta if m["concept"] == c and m["request_idx"] == e]
            cnt = np.bincount([f1[i] for i in ids], minlength=4)
            assert cnt.max() - cnt.min() <= 1
    tot = np.bincount(list(f1.values()), minlength=4)
    assert tot.max() - tot.min() <= 1
    assert P.assign_folds(meta, salt="other") != f1


def test_folds_never_leak():
    """Every OOF row is scored by a model that never saw its text, and on pure noise the OOF
    validity AUROC is near chance while the in-sample one is inflated (d >> n)."""
    ts, _, _ = make_texts(C=3, n=16, d=400, signal=0.0, n_neutral=8)
    cf = P.cross_fit(ts, 3)
    for k, mdl in enumerate(cf.models):
        fold_k = np.flatnonzero(ts.fold == k)
        assert not np.isin(fold_k, mdl.train).any()
        assert set(mdl.train) == set(np.flatnonzero((ts.fold != k) & (ts.concept >= 0)))
    oof = P.validity(ts, cf.oof, 3)
    insample = P.fit_fold(ts, np.arange(len(ts.ids)), 3)
    ins = P.validity(ts, insample.score(ts.H, ts.ending, ts.offset), 3)
    assert np.all(ins > 0.9)
    assert np.all(np.abs(oof - 0.5) < 0.2), oof
    # thresholds of fold k come from fold k's rows only
    ths = P.fold_thresholds(ts, cf, 3)
    rf = ts.row_fold
    for k in range(4):
        rc = ts.row_concept[rf == k]
        want = P.q95(cf.oof[rf == k][(rc != 0) & (rc >= 0), 0])
        assert ths[k].tau_cat[0] == pytest.approx(want)


def test_tau_is_the_95th_percentile():
    rng = np.random.default_rng(1)
    S_ = rng.normal(size=(1000, 2))
    rc = np.repeat([0, 1, P.NEUTRAL, 1], 250)
    ri = np.tile([True, False], 500)
    th = P.thresholds_from(S_, rc, ri, 2)
    assert th.tau_cat[0] == pytest.approx(np.quantile(S_[rc == 1, 0], 0.95))
    assert th.tau_neu[1] == pytest.approx(np.quantile(S_[rc == P.NEUTRAL, 1], 0.95))
    assert th.tau_ceil[0] == pytest.approx(np.quantile(S_[(rc == 1) & ri, 0], 0.95))
    x = rng.normal(size=50)
    w = rng.integers(0, 4, size=50)
    assert P.weighted_quantile(x, w) == pytest.approx(np.quantile(np.repeat(x, w), 0.95))


def test_tau_cg_and_z_use_gold_implied_rows():
    rc = np.array([0] * 4 + [1] * 4 + [2] * 4)
    ri = np.array([True, True, False, False] * 3)
    S_ = np.zeros((12, 3))
    S_[:, 2] = [0, 0, 0, 0, 1, 3, 100, 100, 9, 9, 9, 9]  # probe 2 on g=1 texts: implied 1, 3
    th = P.thresholds_from(S_, rc, ri, 3)
    assert th.tau_cg[2, 1] == pytest.approx(np.quantile([1, 3], 0.95))
    assert th.tau_cg_all[2, 1] == pytest.approx(np.quantile([1, 3, 100, 100], 0.95))
    assert th.z_mu[2, 1] == pytest.approx(2.0)
    assert th.z_sd[2, 1] == pytest.approx(np.std([1, 3], ddof=1))
    # eval scoring: one identity "model", score 5 on probe 2 at a g=1 cell
    mdl = P.FoldModel(
        D=np.eye(3), U=np.eye(3), keys={(0, 0): 0}, muU=np.zeros((1, 3)), sigma=np.ones((1, 3))
    )
    H = np.array([[0.0, 0.0, 5.0]])
    es = P.score_eval([mdl, mdl], [th, th], H, np.array([0]), np.array([0]), np.array([1]))
    assert es.z[0, 2] == pytest.approx((5 - 2) / np.std([1, 3], ddof=1))
    assert np.isnan(es.z[0, 1])  # c = g
    assert es.det_cg[0, 2] == 1.0
    assert es.det_ceil[0, 1] in (0.0, 1.0) and np.isnan(es.det_ceil[0, 2])


def test_score_eval_averages_fold_indicators():
    mdl = P.FoldModel(np.eye(2), np.eye(2), {(0, 0): 0}, np.zeros((1, 2)), np.ones((1, 2)))
    base = P.thresholds_from(np.zeros((4, 2)), np.array([0, 0, 1, 1]), np.ones(4, bool), 2)
    lo, hi = P.Thresholds(**base.__dict__), P.Thresholds(**base.__dict__)
    lo.tau_cat = np.array([0.0, 0.0])
    hi.tau_cat = np.array([10.0, 10.0])
    es = P.score_eval(
        [mdl] * 4,
        [lo, lo, lo, hi],
        np.array([[1.0, 1.0]]),
        np.array([0]),
        np.array([0]),
        np.array([0]),
    )
    assert es.det_cat[0, 1] == pytest.approx(0.75)


def test_unembed_cosine():
    D = np.array([[1.0, 0.0], [0.0, 2.0]])
    W = np.array([[3.0, 0.0], [1.0, 1.0]])
    c = P.unembed_cosines(D, W)
    assert c[0, 0] == pytest.approx(1.0)
    assert c[1, 1] == pytest.approx(1 / np.sqrt(2))


def test_auroc_ties_and_empty():
    assert P.auroc(np.array([2.0, 3.0]), np.array([1.0])) == 1.0
    assert P.auroc(np.array([1.0]), np.array([1.0])) == 0.5
    assert np.isnan(P.auroc(np.array([]), np.array([1.0])))


# ------------------------------------------------------------------ pair classes (M / U)


def rec(kind="prose", mentions=(), token_hits=None, summary="s", gold="g", named=()):
    return {
        "gold": gold,
        "named": list(named),
        "kind": kind,
        "empty": False,
        "mentions": [dict(m) for m in mentions],
        "token_hits": token_hits or {},
        "summary": summary if kind == "tokens" else None,
    }


def m(concept, verdict):
    return {"concept": concept, "verdict": verdict, "leaf": "x", "scope": None}


def test_classify_pair_prose():
    r = rec(
        mentions=[
            m("a", tx.V_PROBE),
            m("b", tx.V_CORRECT),
            m("c", tx.V_NON_CLAIM),
            m("d", tx.FAILED),
            m("e", tx.V_JUNK),
            m("e", tx.V_NON_CLAIM),
            m("f", tx.V_CORRECT),
            m("f", tx.V_HALLUCINATION),
        ],
        named=["n"],
    )
    assert S.classify_pair(r, "g") == "gold"
    assert S.classify_pair(r, "n") == "named"
    assert S.classify_pair(r, "a") == "M"
    assert S.classify_pair(r, "b") == "correct"
    assert S.classify_pair(r, "c") == "non_claim"
    assert S.classify_pair(r, "d") == "failed"
    assert S.classify_pair(r, "e") == "M"
    assert S.classify_pair(r, "f") == "correct"  # any correct mention wins
    assert S.classify_pair(r, "z") == "unmentioned"
    for v in (tx.V_HALLUCINATION, tx.V_UNRESOLVED, tx.V_JUNK):
        assert S.classify_pair(rec(mentions=[m("a", v)]), "a") == "M"
    assert S.classify_pair(rec(mentions=[m("a", None)]), "a") == "pending"


def test_classify_pair_jlens_token_and_summary_rule():
    r = rec(
        kind="tokens",
        mentions=[m("a", tx.V_PROBE), m("c", tx.V_PROBE)],
        token_hits={"a": ["Ġa"], "b": ["Ġb"]},
    )
    assert S.classify_pair(r, "a") == "M"  # token AND summary
    assert S.classify_pair(r, "b") == "token_only"
    assert S.classify_pair(r, "c") == "summary_only"
    assert S.classify_pair(r, "z") == "unmentioned"
    assert S.classify_pair(r, "a", token_eligible=False) == "ineligible"
    nos = rec(kind="tokens", token_hits={}, summary=None)
    assert S.classify_pair(nos, "z") == "no_summary"
    assert S.gold_mentioned(
        rec(kind="tokens", mentions=[m("g", tx.V_CORRECT)], token_hits={"g": ["x"]})
    )
    assert not S.gold_mentioned(rec(kind="tokens", mentions=[m("g", tx.V_CORRECT)]))


# ------------------------------------------------------------------ synthetic arm for score


def synth_arm(effect=1.0, n_items=30, seed=0, kind="prose"):
    """Items with gold g0 and siblings s1..s3, 2 layers, 6 positions; siblings are mentioned (M)
    at random cells and their z is shifted by ``effect`` there."""
    rng = np.random.default_rng(seed)
    concepts = {"cat": ["g0", "s1", "s2", "s3"]}
    items = {
        f"i{k}": {"id": f"i{k}", "category": "cat", "gold": "g0", "named": []}
        for k in range(n_items)
    }
    cells, scores = [], {}
    for iid in items:
        for L in (1, 2):
            for pos in range(6):
                ments = []
                for c in ("s1", "s2", "s3"):
                    mentioned = rng.random() < 0.25
                    z = rng.normal() + 0.3 * pos + (effect if mentioned else 0.0)
                    if mentioned:
                        ments.append(
                            {
                                "concept": c,
                                "verdict": tx.V_PROBE,
                                "leaf": "substitution_sibling",
                                "scope": "user",
                                "segment": 0,
                            }
                        )
                    det = float(z > 1.0)
                    scores[(iid, L, pos, c)] = (z, z, det, det, det, det, float("nan"))
                gm = rng.random() < 0.5
                if gm:
                    ments.append(
                        {
                            "concept": "g0",
                            "verdict": tx.V_CORRECT,
                            "leaf": "entailed",
                            "scope": None,
                            "segment": 0,
                        }
                    )
                scores[(iid, L, pos, "g0")] = (
                    3.0,
                    float("nan"),
                    1.0,
                    float("nan"),
                    float("nan"),
                    1.0,
                    1.0,
                )
                cells.append(
                    {
                        "id": iid,
                        "layer": L,
                        "pos": pos,
                        "offset": 5 - pos,
                        "category": "cat",
                        "gold": "g0",
                        "named": [],
                        "kind": kind,
                        "empty": False,
                        "mentions": ments,
                        "token_hits": {},
                        "response_named": [],
                        "summary": None,
                        "segments": [],
                    }
                )
    return cells, scores, items, concepts


def all_pass_gates(concepts, layers=(1, 2)):
    val = {(cat, c, L): 1.0 for cat, cs in concepts.items() for c in cs for L in layers}
    return S.Gates(val, dict(val), {k: list(v) for k, v in concepts.items()})


def test_beta1_beta2_recover_planted_effect():
    cells, scores, _items, concepts = synth_arm(effect=1.0)
    p, _g = S.build_pairs(cells, concepts, {}, scores, all_pass_gates(concepts))
    W = S.boot_weights(len(p.items), 200, 0)
    b1 = S.run_beta1(p, W)
    b2 = S.run_beta2(p, W)
    assert b1["est"] == pytest.approx(1.0, abs=0.2) and b1["lo"] > 0
    assert b2["est"] == pytest.approx(1.0, abs=0.25) and b2["lo"] > 0
    cells0, scores0, *_ = synth_arm(effect=0.0, seed=3)
    p0, _ = S.build_pairs(cells0, concepts, {}, scores0, all_pass_gates(concepts))
    b10 = S.run_beta1(p0, W)
    assert b10["lo"] < 0 < b10["hi"]


def test_beta1_exact_on_noiseless_data():
    """z = 2*mentioned + offset effect + group effect exactly -> beta1 = 2."""
    group = np.array([0, 0, 0, 1, 1, 1, 2, 2])
    offset = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    mentioned = np.array([1, 0, 0, 0, 1, 0, 1, 0], bool)
    z = 2 * mentioned + 0.5 * offset + np.array([1.0, 1, 1, -3, -3, -3, 7, 7])
    item = np.array([0, 0, 0, 1, 1, 1, 2, 2])
    est, _boots = S.beta1(z, mentioned, offset, group, item, S.boot_weights(3, 50, 1))
    assert est == pytest.approx(2.0)


def test_bootstrap_is_seeded_and_deterministic():
    a = S.boot_weights(10, 100, 7)
    b = S.boot_weights(10, 100, 7)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, S.boot_weights(10, 100, 8))
    assert np.all(a.sum(1) == 10)
    cells, scores, _items, concepts = synth_arm(effect=0.5, n_items=12)
    p, g = S.build_pairs(cells, concepts, {}, scores, all_pass_gates(concepts))
    s1 = S.arm_summary(p, g, n_boot=100, seed=0)
    s2 = S.arm_summary(p, g, n_boot=100, seed=0)
    assert json.dumps(s1, sort_keys=True) == json.dumps(s2, sort_keys=True)


def test_excluded_mention_types_on_neither_side():
    cells, scores, _items, concepts = synth_arm(effect=1.0, n_items=6)
    # turn one M mention into a correct one: that pair must leave both M and U
    r = next(c for c in cells if any(x["concept"] == "s1" for x in c["mentions"]))
    for x in r["mentions"]:
        if x["concept"] == "s1":
            x["verdict"] = tx.V_CORRECT
    p, _ = S.build_pairs(cells, concepts, {}, scores, all_pass_gates(concepts))
    sel = S.matched_sample(p)
    i = np.flatnonzero((p.cell == cells.index(r)) & (p.concept == "s1"))
    assert p.cls[i[0]] == "correct" and not sel[i[0]]


def test_gates_and_eligibility():
    concepts = {"cat": ["a", "b", "c", "d"]}
    val = {("cat", c, 1): 0.9 for c in "abcd"} | {("cat", "d", 1): 0.5}
    tr = {("cat", "a", 1): 0.8, ("cat", "b", 1): 0.75, ("cat", "c", 1): 0.6, ("cat", "d", 1): 0.9}
    g = S.Gates(val, tr, concepts)
    assert g.catlayer_pass("cat", 1)  # 3/4 pass transfer
    assert g.eligible("cat", "a", 1)
    assert g.drop_reason("cat", "c", 1) == "transfer"
    assert g.drop_reason("cat", "d", 1) == "validity"
    g2 = S.Gates(val, {k: 0.5 for k in tr} | {("cat", "a", 1): 0.9}, concepts)
    assert g2.drop_reason("cat", "a", 1) == "category_layer_transfer"


def test_transfer_auroc():
    items = {"x": {"category": "cat", "gold": "a"}, "y": {"category": "cat", "gold": "b"}}
    scores = {
        ("x", 1, 0, "a"): (2.0,),
        ("y", 1, 0, "a"): (1.0,),
        ("x", 1, 0, "b"): (0.0,),
        ("y", 1, 0, "b"): (1.0,),
    }
    t = S.transfer_auroc(scores, items, {"cat": ["a", "b"]})
    assert t[("cat", "a", 1)] == 1.0 and t[("cat", "b", 1)] == 1.0


def test_test3_and_halluc_rate_on_planted_data():
    cells, scores, _items, concepts = synth_arm(effect=1.5)
    p, g = S.build_pairs(cells, concepts, {}, scores, all_pass_gates(concepts))
    s = S.arm_summary(p, g, n_boot=100, seed=0)
    assert s["test3"]["tau_cg"]["est"] > 0.2 and s["test3"]["tau_cg"]["lo"] > 0
    det_m = s["detection"]["tau_cat"]["M"]["est"]
    assert s["halluc"]["tau_cat"]["all"]["rate"]["est"] == pytest.approx(1 - det_m, abs=1e-3)
    assert s["detection"]["tau_cat"]["ceiling"]["est"] == 1.0
    assert det_m > s["detection"]["tau_cat"]["U_matched"]["est"]
    assert s["within_item_auroc"]["est"] > 0.6
    assert (
        s["halluc"]["tau_cat"]["scope_user"]["rate"]["n"]
        == s["halluc"]["tau_cat"]["all"]["rate"]["n"]
    )


# ------------------------------------------------------------------ verdict table


def ci(lo, hi):
    return {"est": (lo + hi) / 2, "lo": lo, "hi": hi}


POS, NEG, NULL = ci(0.1, 0.5), ci(-0.5, -0.1), ci(-0.1, 0.1)


@pytest.mark.parametrize(
    ("b1", "b2", "ctrl", "want"),
    [
        (POS, POS, False, "supported"),
        (POS, POS, True, "supported"),
        (POS, NEG, True, "contradicted"),
        (NEG, NULL, True, "contradicted"),
        (NULL, NULL, True, "not supported"),
        (POS, NULL, True, "inconclusive (mixed)"),
        (NULL, POS, True, "inconclusive (mixed)"),
        (NULL, NULL, False, "inconclusive (control failed)"),
        (NEG, NEG, False, "inconclusive (control failed)"),
    ],
)
def test_verdict_cases(b1, b2, ctrl, want):
    assert S.verdict(b1, b2, ctrl) == want


def test_verdict_table_uses_jlens_control():
    arms = {
        "jlens": {"beta1": POS, "beta2": POS},
        "s3d-rl600": {"beta1": NULL, "beta2": NULL, "strong": {"status": "not met"}},
        "s3d-rl600@token_eligible": {"beta1": POS, "beta2": POS, "strong": {"status": "met"}},
        "nla-rl-L42": {"beta1": POS, "beta2": NULL},
    }
    rows = {(r["arm"], r["subset"]): r for r in S.verdict_table(arms)}
    assert rows[("s3d-rl600", "all transfer-passing")]["verdict"] == "not supported"
    assert rows[("s3d-rl600", "token-eligible")]["strongly_supported"] is True
    assert rows[("nla-rl-L42", "all transfer-passing")]["verdict"] == "inconclusive (mixed)"
    arms["jlens"] = {"beta1": POS, "beta2": NULL}
    rows = {(r["arm"], r["subset"]): r for r in S.verdict_table(arms)}
    assert rows[("s3d-rl600", "all transfer-passing")]["verdict"] == (
        "inconclusive (control failed)"
    )


# ------------------------------------------------------------------ judge glue


ITEM = {
    "id": "um-sd-country-01",
    "category": "country",
    "gold": "south_korea",
    "named": [],
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "I love kimchi. Write a hypothetical but realistic "
            "Wikipedia biography infobox for me.",
        },
    ],
    "response": "Here is an infobox: Born in Seoul.",
}


def test_judge_renders_exactly_claimmap_prompt():
    from hallucination_bench.units import split_cell

    samples = ["The user is Korean.\n- They might be from China or Japan."]
    system, user, _units = J.render_map_prompt(ITEM, samples, "prose")
    want_units = cm.plain_units(split_cell(samples)[0])
    ws, wu = cm.render_prompt(
        ITEM["messages"][1]["content"],
        READ_MARK + ITEM["response"],
        want_units,
        kind="prose",
        anchor="none",
    )
    assert (system, user) == (ws, wu)
    assert system == cm.SYSTEM
    assert f"<response>\n{READ_MARK}Here is an infobox" in user
    _, tuser, _ = J.render_map_prompt(ITEM, ["Korea kimchi"], "tokens")
    assert cm.TOKENS_NOTE in tuser


def test_judge_module_copies_no_claimmap_or_summarizer_text():
    from hallucination_bench import summarizer

    src = inspect.getsource(J)
    for text in (
        cm.SYSTEM_INTRO[:80],
        cm.TOKENS_NOTE[:60],
        summarizer.INTERP_SYSTEM[:60],
        "A language model was asked",
    ):
        assert text not in src


def test_coverage_sample_seeded_five_percent():
    keys = {"a": [f"k{i}" for i in range(100)], "b": [f"j{i}" for i in range(7)]}
    s1 = J.coverage_sample(keys, arm="x")
    assert s1 == J.coverage_sample(keys, arm="x")
    assert len([k for k in s1 if k.startswith("k")]) == 5
    assert len([k for k in s1 if k.startswith("j")]) == 1
    assert s1 != J.coverage_sample(keys, arm="y")


def test_attribute_largest_overlap():
    segs = [
        {"readout": 0, "start": 0, "end": 10},
        {"readout": 0, "start": 10, "end": 30},
        {"readout": 1, "start": 0, "end": 30},
    ]
    assert J.attribute(0, 8, 14, segs) == 1
    assert J.attribute(1, 2, 5, segs) == 2
    assert J.attribute(0, 40, 45, segs) is None


def test_process_cell_from_cache_attributes_mentions(tmp_path):
    """Rebuild mode (no LLM): a cached claim map is attributed to the alias hits; the sibling
    hallucination candidate's scope comes from its cached scope call."""
    samples = ["The user is Korean. They may live in China."]
    cell = Cell(ITEM["id"], 36, 40, tuple(samples), None, None)
    meta = {"n": 60}
    system, user, units = J.render_map_prompt(ITEM, samples, "prose")
    models = J.JudgeModels()
    cache = Cache(tmp_path / "cache.jsonl")
    ans = {"units": [], "rationale": "r"}
    for u in units:
        leaf = "entailed" if "Korean" in u.text else "substitution_sibling"
        ans["units"].append(
            {
                "id": u.id,
                "leaf": leaf,
                "hedged": False,
                "world_false": False,
                "cruxy": False,
                "pair": None,
                "probe_target": "China",
                "why": "w",
                "splits": [],
                "rest_leaf": None,
            }
        )
    fp = fingerprint(cm.PROMPT_VERSION, models.judge, models.effort, system, user)
    cache.put(f"{cell.key}:M", fp, {"result": ans, "meta": {}})
    china_seg = next(u for u in units if "China" in u.text)
    s_sys, s_user = J.render_scope(ITEM["messages"][1]["content"], samples[0], china_seg.text)
    cache.put(
        f"{cell.key}:S:0:{china_seg.start}-{china_seg.end}",
        fingerprint(J.SCOPE_VERSION, models.judge, models.effort, s_sys, s_user),
        {"result": {"scope": "user", "why": "w"}, "meta": {}},
    )
    ja = J.JudgeArm("s3d-rl600", tmp_path, cache, "prose")
    r = asyncio.run(
        J.process_cell(ja, cell, ITEM, meta, None, models, in_sample=False, summarize_all=True)
    )
    assert r["prefiltered"] and r["judged"] and r["status"] == "judged"
    assert r["offset"] == 19
    by = {h["concept"]: h for h in r["mentions"]}
    assert by["south_korea"]["verdict"] == tx.V_CORRECT
    assert by["china"]["verdict"] == tx.V_PROBE
    assert by["china"]["scope"] == "user"
    assert S.classify_pair(r, "china") == "M"
    assert S.classify_pair(r, "japan") == "unmentioned"
    assert S.gold_mentioned(r)


def test_jlens_cell_without_token_hit_not_prefiltered(tmp_path):
    cell = Cell(ITEM["id"], 36, 40, None, ("Ġkimchi", "Ġfood"), (2.0, 1.0))
    ja = J.JudgeArm("jlens", tmp_path, Cache(tmp_path / "c.jsonl"), "tokens")
    r = asyncio.run(
        J.process_cell(
            ja, cell, ITEM, {"n": 60}, None, J.JudgeModels(), in_sample=False, summarize_all=True
        )
    )
    assert not r["prefiltered"] and not r["judged"]
    assert r["status"] == "pending_summary"
    assert S.classify_pair(r, "china") == "no_summary"
    inv = load_inventory()["country"]
    tok = next(t for c in inv.concepts if c.id == "china" for t in c.token_aliases)
    assert J.token_hits(J.matcher("country"), ["Ġfood", tok.replace(" ", "Ġ")]) == {
        "china": [tok.replace(" ", "Ġ")]
    }


def test_render_cover_and_scope_have_no_placeholders():
    _, u = J.render_cover("tree", "sycamore")
    assert "{" not in u and "a kind of tree" in u and "sycamore" in u
    _s, u = J.render_scope("msg {readout}", "ro", "seg")
    assert "msg {readout}" in u and "<segment>\nseg\n</segment>" in u


# ------------------------------------------------------------------ score_all + report


def test_score_all_and_report_end_to_end(tmp_path):
    from hallucination_bench.evals.sibling_probe import report as R

    cells, scores, items, concepts = synth_arm(effect=1.0, n_items=10)
    for it in items.values():
        it["gold"] = "g0"
    validity = {(cat, c, L): 1.0 for cat, cs in concepts.items() for c in cs for L in (1, 2)}
    # transfer needs c-gold items: give s1..s3 their own items with high own-probe scores
    for c in ("s1", "s2", "s3"):
        items[f"own-{c}"] = {"id": f"own-{c}", "category": "cat", "gold": c, "named": []}
        for L in (1, 2):
            scores[(f"own-{c}", L, 0, c)] = (9.0,) * 7
            for o in concepts["cat"]:
                if o != c:
                    scores[(f"own-{c}", L, 0, o)] = (-9.0,) * 7
    for (iid, L, pos, c), v in list(scores.items()):
        if c == "g0" and not iid.startswith("own"):
            scores[(iid, L, pos, c)] = (9.0, *v[1:])
    summ = S.score_all(
        {"s3d-rl600": cells, "jlens": cells},
        scores,
        validity,
        items,
        concepts,
        {("cat", c): True for c in concepts["cat"]},
        n_boot=50,
        log=lambda s: None,
    )
    assert "s3d-rl600@token_eligible" in summ["arms"]
    assert summ["verdict"][0]["verdict"] == "supported"
    S.write_summary(summ, tmp_path)
    res = R.write_report(tmp_path, tmp_path / "plots")
    assert (tmp_path / "report.md").read_text().startswith("# Plan 0003")
    assert len(res["plots"]) >= 3


def test_assign_folds_balanced_by_kind():
    """Review fix: every fold gets ~equal implied (and explicit) texts per concept."""
    meta = [
        {"id": f"t{c}_{kind}_{j}", "concept": c, "request_idx": j % 2, "kind": kind}
        for c in range(3)
        for kind, n in (("explicit", 10), ("implied", 6))
        for j in range(n)
    ] + [
        {"id": f"n{j}", "concept": None, "request_idx": j % 2, "kind": "neutral"} for j in range(9)
    ]
    f = P.assign_folds(meta, salt="cat")
    for c in range(3):
        for kind in ("explicit", "implied"):
            ids = [m["id"] for m in meta if m["concept"] == c and m["kind"] == kind]
            cnt = np.bincount([f[i] for i in ids], minlength=4)
            assert cnt.max() - cnt.min() <= 2, (c, kind, cnt)
            for e in (0, 1):
                sub = [m["id"] for m in meta if m["concept"] == c and m["kind"] == kind]
                sub = [i for i in sub if next(m for m in meta if m["id"] == i)["request_idx"] == e]
                cnt = np.bincount([f[i] for i in sub], minlength=4)
                assert cnt.max() - cnt.min() <= 1, (c, kind, e, cnt)
        ids = [m["id"] for m in meta if m["concept"] == c]
        cnt = np.bincount([f[i] for i in ids], minlength=4)
        assert cnt.max() - cnt.min() <= 1
    tot = np.bincount(list(f.values()), minlength=4)
    assert tot.max() - tot.min() <= 1


def test_named_gold_excluded_from_golds_and_transfer_positives():
    """Review fix: an item whose gold is named in its prompt adds no gold-mention / ceiling row
    and is not a transfer positive for its gold; it stays a negative for other concepts."""
    cells, scores, items, concepts = synth_arm(effect=1.0, n_items=6)
    for r in cells:
        if r["id"] == "i0":
            r["named"] = ["g0"]
    p, g = S.build_pairs(cells, concepts, {}, scores, all_pass_gates(concepts), sorted(items))
    i0 = sorted(items).index("i0")
    assert not np.isin(i0, g.item).any()
    assert np.isin(i0, p.item).any()  # its sibling pairs are kept
    assert g.item.size == sum(r["id"] != "i0" for r in cells)
    items = {
        "x": {"category": "cat", "gold": "a", "named": ["a"]},
        "y": {"category": "cat", "gold": "b", "named": []},
        "w": {"category": "cat", "gold": "a", "named": []},
    }
    scores = {
        ("x", 1, 0, "a"): (5.0,),
        ("w", 1, 0, "a"): (0.5,),
        ("y", 1, 0, "a"): (1.0,),
        ("x", 1, 0, "b"): (2.0,),
        ("w", 1, 0, "b"): (0.0,),
        ("y", 1, 0, "b"): (1.0,),
    }
    t = S.transfer_auroc(scores, items, {"cat": ["a", "b"]})
    assert t[("cat", "a", 1)] == 0.0  # positives = w only (x dropped): 0.5 < 1.0
    assert t[("cat", "b", 1)] == 0.5  # x stays a negative for b: 1.0 vs {2.0, 0.0}


def test_pair_table_and_adjudicable_macro():
    cells, scores, _items, concepts = synth_arm(effect=1.0, n_items=12)
    concepts2 = {"cat": concepts["cat"], "weak": ["w0", "w1"]}
    gates = S.Gates(
        {("cat", c, L): 1.0 for c in concepts["cat"] for L in (1, 2)}
        | {("weak", c, L): 0.5 for c in ("w0", "w1") for L in (1, 2)},
        {("cat", c, L): 1.0 for c in concepts["cat"] for L in (1, 2)}
        | {("weak", c, L): 1.0 for c in ("w0", "w1") for L in (1, 2)},
        concepts2,
    )
    assert gates.pass_share("cat") == 1.0 and gates.pass_share("weak") == 0.0
    p, g = S.build_pairs(cells, concepts, {}, scores, gates)
    s = S.arm_summary(p, g, n_boot=50, seed=0, gates=gates)
    assert s["category_gates"]["cat"] == {"layers": [1, 2], "pass_share": 1.0, "adjudicable": True}
    h = s["halluc"]["tau_cat"]["all"]
    assert h["by_category"]["cat"]["adjudicable"] is True
    assert h["macro_adjudicable"]["est"] == h["macro"]["est"]
    m = (p.cls == "M") & p.eligible
    pt = s["pair_table"]
    assert [r["n"] for r in pt] == sorted((r["n"] for r in pt), reverse=True)
    assert sum(r["n"] for r in pt) == int(m.sum())
    r = pt[0]
    sel = m & (p.concept == r["sibling"])
    assert r["gold"] == "g0" and r["sibling_gates_all_layers"] is True
    assert r["det_cat"] == pytest.approx(p.vals["det_cat"][sel].mean(), abs=1e-4)
    assert r["z_mean"] == pytest.approx(p.vals["z"][sel].mean(), abs=1e-4)
    # a category below 50% pass share is flagged and left out of the adjudicable macro
    weak = S.Gates(
        {
            ("cat", c, L): (1.0 if c in ("g0", "s1") or L == 1 else 0.5)
            for c in concepts["cat"]
            for L in (1, 2)
        },
        {("cat", c, L): 1.0 for c in concepts["cat"] for L in (1, 2)},
        concepts,
    )
    assert weak.pass_share("cat") == 0.75
    weak.validity |= {("cat", c, 1): 0.5 for c in ("s2", "s3")}
    assert weak.pass_share("cat") == 0.5
    weak.validity[("cat", "s1", 1)] = 0.5
    p2, g2 = S.build_pairs(cells, concepts, {}, scores, weak)
    s2 = S.arm_summary(p2, g2, n_boot=50, seed=0, gates=weak)
    assert s2["category_gates"]["cat"]["adjudicable"] is False
    h2 = s2["halluc"]["tau_cat"]["all"]
    assert h2["by_category"]["cat"]["adjudicable"] is False
    assert h2["macro_adjudicable"]["est"] is None and h2["macro_adjudicable"]["n"] == 0
