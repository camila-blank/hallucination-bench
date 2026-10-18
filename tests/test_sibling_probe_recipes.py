"""Plan 0003 rev 11: V2 (shrinkage-whitened DiffMean) port, recipe tables, recipe sensitivity."""

import json

import numpy as np
import pytest
from test_sibling_probe_stage2 import make_texts, synth_arm

from hallucination_bench.evals.sibling_probe import probe_io as PIO
from hallucination_bench.evals.sibling_probe import probes as P
from hallucination_bench.evals.sibling_probe import report as R
from hallucination_bench.evals.sibling_probe import score as S


def correlated_texts(C=3, n=24, d=24, seed=3):
    """Texts whose within-class noise is strongly correlated across dimensions (a nuisance
    direction that dwarfs the class signal along the raw DiffMean), the case whitening fixes."""
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(C, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    nuis = rng.normal(size=d)
    nuis /= np.linalg.norm(nuis)
    meta, wins = [], []
    for c in [*range(C), P.NEUTRAL]:
        for j in range(n if c >= 0 else 8):
            e = j % 2
            w = np.zeros((6, d))
            w = w + 1.5 * dirs[c] * (c >= 0)
            w = w + rng.normal(size=(6, 1)) * 6.0 * nuis  # large shared nuisance
            w = w + 0.3 * rng.normal(size=(6, d))
            kind = "neutral" if c < 0 else ("implied" if j % 4 < 2 else "explicit")
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
    return ts


def test_whitened_direction_is_shrinkage_lda_of_the_diffmean():
    ts, _, _ = make_texts(C=3, n=24, d=16, signal=2.0)
    wh = P.Whitener(ts, 3)
    fs = frozenset({0, 1, 2})
    tr = wh.train_texts(fs)
    # S_w by hand: positions of the concept texts in the training folds, minus the mean of their
    # (concept, ending, offset) group, dof n - K*C (K = (ending, offset) keys of the WHOLE set)
    rows = np.flatnonzero(np.isin(ts.text, tr))
    H = ts.H[rows].astype(np.float64)
    grp = list(zip(ts.row_concept[rows], ts.ending[rows], ts.offset[rows], strict=True))
    R_ = np.empty_like(H)
    for g in set(grp):
        ix = np.array([i for i, x in enumerate(grp) if x == g])
        R_[ix] = H[ix] - H[ix].mean(axis=0)
    K = len(set(zip(ts.ending.tolist(), ts.offset.tolist(), strict=True)))
    assert K == wh.K
    S_w = R_.T @ R_ / (len(rows) - K * 3)
    D0 = P._diffmean(ts, tr, 3)
    for lam in (0.01, 1.0):
        a = lam * np.trace(S_w) / S_w.shape[0]
        want = np.linalg.solve(S_w + a * np.eye(S_w.shape[0]), D0.T).T
        np.testing.assert_allclose(wh.directions(fs, lam), want, rtol=1e-6, atol=1e-9)
    # a huge shrinkage returns the raw DiffMean direction (up to scale)
    big = wh.directions(fs, 1e9)
    cos = np.sum(big * D0, axis=1) / np.linalg.norm(big, axis=1) / np.linalg.norm(D0, axis=1)
    assert np.all(cos > 0.999999)


def test_whitener_uses_concept_texts_of_training_folds_only():
    ts, _, _ = make_texts(C=3, n=24, d=16, signal=2.0)
    wh = P.Whitener(ts, 3)
    fs = frozenset({0, 1})
    tr = wh.train_texts(fs)
    assert set(ts.fold[tr].tolist()) <= {0, 1} and (ts.concept[tr] >= 0).all()
    d1 = wh.directions(fs, 0.1)
    # moving a held-out fold's (or a neutral) text must not change the training-fold model
    ts2, _, _ = make_texts(C=3, n=24, d=16, signal=2.0)
    held = np.flatnonzero((ts2.fold >= 2) | (ts2.concept < 0))
    for t in held:
        ts2.H[ts2.text == t] += 100.0
    object.__setattr__(ts2, "_means", None)
    d2 = P.Whitener(ts2, 3).directions(fs, 0.1)
    np.testing.assert_allclose(d1, d2, rtol=1e-6, atol=1e-8)


def test_v2_beats_v0_with_correlated_noise_and_picks_lam_from_grid():
    ts = correlated_texts()
    C = 3
    v0 = P.run_category_layer(ts, C, recipe="v0")
    v2 = P.run_category_layer(ts, C, recipe="v2")
    assert v2.hyper is not None and len(v2.hyper) == P.N_FOLDS
    assert all(h in P.LAMS_W for h in v2.hyper)
    assert np.nanmean(v2.valid_auroc) > np.nanmean(v0.valid_auroc) + 0.1
    assert np.nanmean(v2.valid_auroc) > 0.9
    # held-out folds are scored by their own model only
    for k, m in enumerate(v2.cf.models):
        assert not np.isin(np.flatnonzero(ts.fold == k), m.train).any()
    # sigma is recomputed with the whitened unit direction: scores are unit-variance-ish
    # at the training positions of every fold
    assert np.allclose(np.linalg.norm(v2.cf.models[0].U, axis=1), 1.0)
    with pytest.raises(ValueError):
        P.run_category_layer(ts, C, recipe="v9")


def test_recipe_gates_and_layout():
    assert P.VALIDITY_BY_RECIPE == {"v2": 0.85, "v0": 0.70}
    assert P.RECIPES[0] == "v2" and P.TRANSFER_MIN_AUROC == 0.70
    assert S.Gates({}, {}, {}).min_validity == 0.85


def _fake_summary(verdict, rate, ceiling=0.8, excess=0.9):
    ci = lambda e, w=0.03: {"est": e, "lo": e - w, "hi": e + w, "n": 10}  # noqa: E731
    arm = {
        "halluc": {th: {"all": {"rate": ci(rate), "ceiling": ci(ceiling)}} for th in S.THRESHOLDS},
        "normalised": {
            th: {"rate_normalised": ci(1 - (1 - rate) / ceiling), "ratio": ci(excess)}
            for th in S.THRESHOLDS
        },
        "detection": {
            th: {"M": ci(1 - rate), "U_matched": ci(0.1), "ceiling": ci(ceiling)}
            for th in S.THRESHOLDS
        },
    }
    return {
        "arms": {"s3d-rl600": arm},
        "verdict": [{"arm": "s3d-rl600", "subset": "all transfer-passing", "verdict": verdict}],
    }


def test_recipe_sensitivity_rule():
    same = S.recipe_sensitivity(
        {"v2": _fake_summary("supported", 0.5), "v0": _fake_summary("supported", 0.52)}
    )
    assert same["status"] == "robust to recipe" and not same["reasons"]
    diff = S.recipe_sensitivity(
        {"v2": _fake_summary("supported", 0.5), "v0": _fake_summary("not supported", 0.5)}
    )
    assert diff["status"] == "recipe-sensitive" and "verdict" in diff["reasons"][0]
    # raw rates that differ only through the gold-miss floor do not fire; normalised CIs do
    raw = S.recipe_sensitivity(
        {
            "v2": _fake_summary("supported", 0.30, ceiling=0.9),
            "v0": _fake_summary("supported", 0.55, ceiling=0.5, excess=0.9),
        }
    )
    assert raw["rates"][0]["raw"]["overlap"] is False
    far = S.recipe_sensitivity(
        {
            "v2": _fake_summary("supported", 0.2, excess=0.9),
            "v0": _fake_summary("supported", 0.2, excess=0.3),
        }
    )
    assert far["status"] == "recipe-sensitive"
    assert S.recipe_sensitivity({"v2": _fake_summary("supported", 0.5)})["status"] == (
        "single recipe"
    )


def test_score_recipes_and_report_end_to_end(tmp_path):
    cells, scores, items, concepts = synth_arm(effect=1.0, n_items=10)
    for it in items.values():
        it["gold"] = "g0"
    validity = {(c_, c, L): 0.9 for c_, cs in concepts.items() for c in cs for L in (1, 2)}
    rows = {k: {"fpr_neutral": 0.05} for k in validity}
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
    v0_validity = {k: 0.75 for k in validity}  # passes 0.70, fails 0.85
    inputs = {"v2": (scores, validity, rows), "v0": (scores, v0_validity, rows)}
    summ = S.score_recipes(
        {"s3d-rl600": cells, "jlens": cells},
        inputs,
        items,
        concepts,
        {("cat", c): True for c in concepts["cat"]},
        n_boot=40,
        log=lambda s: None,
    )
    assert summ["primary_recipe"] == "v2" and set(summ["recipes"]) == {"v2", "v0"}
    # V2 is gated at 0.85 (its own validity row is the main row); V0 keeps the 0.70 gate
    assert "s3d-rl600@validity085" not in summ["recipes"]["v2"]["arms"]
    assert "s3d-rl600@validity085" in summ["recipes"]["v0"]["arms"]
    assert summ["recipes"]["v2"]["arms"]["s3d-rl600"]["coverage"]["n_M_eligible"] > 0
    assert summ["recipes"]["v0"]["arms"]["s3d-rl600"]["coverage"]["n_M_eligible"] > 0
    assert summ["recipes"]["v0"]["arms"]["s3d-rl600@validity085"]["coverage"]["n_M_eligible"] == 0
    assert summ["recipe_sensitivity"]["status"] in ("robust to recipe", "recipe-sensitive")
    assert summ["calibration"]["v2"]["by_category"]["cat"]["neutral_fpr"] == 0.05
    S.write_summary(summ, tmp_path)
    res = R.write_report(tmp_path, tmp_path / "plots")
    md = (tmp_path / "report.md").read_text()
    assert "Recipe sensitivity" in md and "gold-miss floor" in md
    assert len(res["plots"]) >= 3


def test_verbatim_transfer_uses_only_um_sd_items():
    concepts = {"cat": ["a", "b"]}
    items = {
        "um-sd-1": {"id": "um-sd-1", "category": "cat", "gold": "a", "named": []},
        "um-sd-2": {"id": "um-sd-2", "category": "cat", "gold": "b", "named": []},
        "gen-1": {"id": "gen-1", "category": "cat", "gold": "a", "named": []},
    }
    sc = {}
    for iid, gold in (("um-sd-1", "a"), ("um-sd-2", "b"), ("gen-1", "a")):
        for c in "ab":
            v = 1.0 if c == gold else -1.0
            if iid == "gen-1":
                v = -v  # would break the AUROC if it leaked in
            sc[(iid, 44, 0, c)] = (v,) + (0.0,) * 6
    out = S.verbatim_transfer({"v2": sc, "v0": sc}, items, concepts, n_boot=20)
    assert out["n_items"] == 2 and out["recipes"]["all_layers"]["v2"]["est"] == 1.0


def test_pipeline_v2_matches_sweep_when_outputs_present():
    """Offline check on the real run (skipped without the Modal outputs): pipeline V2 / V0
    validity and transfer equal the sweep's, max abs diff < 1e-3."""
    dirs = PIO.recipe_dirs(PIO.PROBES_DIR)
    if "v2" not in dirs or not (PIO.SWEEP_DIR / "config.json").exists():
        pytest.skip(
            f"no pipeline V2 outputs / sweep results under {PIO.PROBES_DIR} / {PIO.SWEEP_DIR} "
            "(gitignored; produced by the Modal probe run)"
        )
    for recipe in dirs:
        r = PIO.compare_with_sweep(PIO.PROBES_DIR, PIO.SWEEP_DIR, recipe)
        assert r["n"] > 400, json.dumps(r)
        assert r["max_abs_diff_validity"] < 1e-3, json.dumps(r)
        assert r["max_abs_diff_transfer"] < 1e-3, json.dumps(r)
