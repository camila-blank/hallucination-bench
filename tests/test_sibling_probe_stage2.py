"""Plan 0003 stage 2: probes, judge glue, scoring, verdict, report (synthetic data only)."""

import numpy as np
import pytest

from hallucination_bench.evals.sibling_probe import probes as P

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
