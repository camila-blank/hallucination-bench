"""Plan 0003 rev 11: V2 (shrinkage-whitened DiffMean) port, recipe tables, recipe sensitivity."""

import numpy as np
import pytest
from test_sibling_probe_stage2 import make_texts

from hallucination_bench.evals.sibling_probe import probes as P


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
