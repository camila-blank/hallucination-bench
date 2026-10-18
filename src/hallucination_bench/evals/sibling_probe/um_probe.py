"""Plan 0006 §4b: the two probe types per layer, their per-fold null standardisation, validity
and the implied-text centroids. numpy (+ scikit-learn for type (b)); imports nothing from the
package except :mod:`probes` (both files are copied into the Modal job).

Vocabulary (one layer): a :class:`probes.TextSet` of probe texts (concept index ``c >= 0`` or
``NEUTRAL``), 4 folds by text (``probes.assign_folds``).

* **type (a)**, whitened DiffMean: plan 0003 rev 11 V2, ``probes.cross_fit_whitened``,
  unchanged (lambda by inner CV);
* **type (b)**, multinomial logistic regression over the concepts (neutral texts are not a
  class): per-offset centring (the equal-weighted mean over concepts of the training positions
  at each offset), then a standardiser and PCA-256 fitted on the training fold, then L2
  multinomial logistic regression with C chosen by inner leave-one-training-fold-out CV over
  :data:`LOGREG_CS` (criterion: mean over concepts of the position-level implied AUROC). Score =
  the class logit.
* **null z** (both types, per fold model k): ``z_c = (score_c - m[c, o]) / sd[c, o]`` with m, sd
  the mean / sd of score_c over fold k's out-of-fold **implied** positions of the OTHER concepts
  at offset o. Eval scores are z'd per fold model, then averaged over the 4 models.
* **validity** per concept: OOF AUROC of z_c, positives = c's implied positions, negatives =
  every other concept's implied positions (neutral texts excluded).
* **centroids** (§4c): per concept, the mean over its implied texts' window positions,
  grand-mean-centred (minus the mean of the concept means).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

try:  # inside the Modal job the module is copied as a top-level file
    from hallucination_bench.evals.sibling_probe import probes as P
except ImportError:  # pragma: no cover
    import sibprobes as P  # type: ignore[no-redef]


def _key(ts: P.TextSet) -> np.ndarray:
    """Position key for centring and the null (plan 0011): ``ending * 1000 + offset``. With one
    request sentence (ending 0) it equals the offset, so one-ending results are unchanged."""
    return np.asarray(ts.ending, dtype=np.int64) * 1000 + np.asarray(ts.offset, dtype=np.int64)


def eval_key(ending: np.ndarray, offset: np.ndarray) -> np.ndarray:
    return np.asarray(ending, dtype=np.int64) * 1000 + np.asarray(offset, dtype=np.int64)


LOGREG_CS = (10.0, 30.0, 100.0, 300.0)
PCA_K = 512
TYPES = ("a", "b")


# ------------------------------------------------------------------ type (b): logistic


@dataclass
class LogregModel:
    """Fitted on one fold's training texts. ``mu`` [O, d] per-offset centres (offset index via
    ``offsets``), ``mean`` / ``scale`` the standardiser, ``comp`` [k, d] PCA components (applied
    after standardising), ``coef`` [C, k], ``intercept`` [C]."""

    offsets: np.ndarray
    mu: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    pca_mean: np.ndarray
    comp: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray
    C_reg: float

    def logits(self, H: np.ndarray, offset: np.ndarray) -> np.ndarray:
        oi = {int(o): i for i, o in enumerate(self.offsets)}
        idx = np.array([oi.get(int(o), -1) for o in offset])
        out = np.full((H.shape[0], self.coef.shape[0]), np.nan)
        ok = idx >= 0
        if not ok.any():
            return out
        X = (np.asarray(H[ok], np.float64) - self.mu[idx[ok]] - self.mean) / self.scale
        Z = (X - self.pca_mean) @ self.comp.T
        out[ok] = Z @ self.coef.T + self.intercept
        return out


def _offset_centres(
    ts: P.TextSet, rows: np.ndarray, n_concepts: int
) -> tuple[np.ndarray, np.ndarray]:
    rc = ts.row_concept
    offs = np.unique(_key(ts)[rows])
    mu = np.zeros((len(offs), ts.H.shape[1]))
    for i, o in enumerate(offs):
        at = rows & (_key(ts) == o)
        per_c = [
            ts.H[at & (rc == c)].mean(axis=0) for c in range(n_concepts) if np.any(at & (rc == c))
        ]
        mu[i] = np.mean(per_c, axis=0)
    return offs, mu


@dataclass
class _Prep:
    offsets: np.ndarray
    mu: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    pca_mean: np.ndarray
    comp: np.ndarray

    def transform(self, H: np.ndarray, offset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        oi = {int(o): i for i, o in enumerate(self.offsets)}
        idx = np.array([oi.get(int(o), -1) for o in offset])
        ok = idx >= 0
        X = (np.asarray(H[ok], np.float32) - self.mu[idx[ok]] - self.mean) / self.scale
        return (X - self.pca_mean) @ self.comp.T, ok


def _prep(
    ts: P.TextSet, train_texts: np.ndarray, n_concepts: int, seed: int = 0
) -> tuple[_Prep, np.ndarray, np.ndarray]:
    """Per-offset centring, standardiser and PCA on the training texts' concept positions;
    returns the prep and the transformed training rows (Z, y)."""
    from sklearn.decomposition import PCA

    intr = np.zeros(len(ts.ids), bool)
    intr[train_texts] = True
    rows = intr[ts.text] & (ts.row_concept >= 0)
    offs, mu = _offset_centres(ts, rows, n_concepts)
    mu = mu.astype(np.float32)
    oi = {int(o): i for i, o in enumerate(offs)}
    X = ts.H[rows].astype(np.float32) - mu[[oi[int(o)] for o in _key(ts)[rows]]]
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0).astype(np.float32)
    X = (X - mean) / scale
    k = min(PCA_K, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=k, svd_solver="randomized", random_state=seed).fit(X)
    Z = pca.transform(X)
    pr = _Prep(
        offs, mu, mean, scale, pca.mean_.astype(np.float32), pca.components_.astype(np.float32)
    )
    return pr, Z, ts.row_concept[rows]


def _fit_lr(pr: _Prep, Z: np.ndarray, y: np.ndarray, n_concepts: int, C_reg: float) -> LogregModel:
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=C_reg, max_iter=1000, tol=1e-4)
    clf.fit(Z, y)
    if list(clf.classes_) != list(range(n_concepts)):
        raise ValueError(f"logreg classes {clf.classes_} != 0..{n_concepts - 1}")
    return LogregModel(
        pr.offsets, pr.mu, pr.mean, pr.scale, pr.pca_mean, pr.comp, clf.coef_, clf.intercept_, C_reg
    )


def fit_logreg(
    ts: P.TextSet, train_texts: np.ndarray, n_concepts: int, C_reg: float, seed: int = 0
) -> LogregModel:
    pr, Z, y = _prep(ts, train_texts, n_concepts, seed)
    return _fit_lr(pr, Z, y, n_concepts, C_reg)


def _mean_implied_auroc(S: np.ndarray, rows: np.ndarray, ts: P.TextSet, n_concepts: int) -> float:
    rc, imp = ts.row_concept, ts.row_implied
    vals = []
    for c in range(n_concepts):
        pos = S[rows & imp & (rc == c), c]
        neg = S[rows & imp & (rc != c) & (rc >= 0), c]
        if pos.size and neg.size:
            vals.append(P.auroc(pos, neg))
    return float(np.nanmean(vals)) if vals else float("nan")


@dataclass
class LogregCrossFit:
    models: list[LogregModel]
    oof: np.ndarray  # [P, C] logits, each row by its own fold's model
    C_per_fold: list[float]
    inner: list[dict[float, float]]


def cross_fit_logreg(
    ts: P.TextSet, n_concepts: int, n_folds: int = P.N_FOLDS, Cs: Sequence[float] = LOGREG_CS
) -> LogregCrossFit:
    T = np.arange(len(ts.ids))
    rf = ts.row_fold
    oof = np.full((ts.H.shape[0], n_concepts), np.nan)
    models, chosen, tables = [], [], []
    for k in range(n_folds):
        F = [f for f in range(n_folds) if f != k]
        sc: dict[float, list[float]] = {c: [] for c in Cs}
        for j in F:
            tr = T[np.isin(ts.fold, [f for f in F if f != j]) & (ts.concept >= 0)]
            r = (rf == j) & (ts.row_concept >= 0)
            pr, Z, y = _prep(ts, tr, n_concepts)  # shared by every C (it does not depend on C)
            for c in Cs:
                m = _fit_lr(pr, Z, y, n_concepts, c)
                S = np.full(oof.shape, np.nan)
                S[r] = m.logits(ts.H[r], _key(ts)[r])
                sc[c].append(_mean_implied_auroc(S, r, ts, n_concepts))
        table = {c: float(np.mean(v)) for c, v in sc.items()}
        best = max(table, key=lambda c: table[c])
        tr = T[np.isin(ts.fold, F) & (ts.concept >= 0)]
        m = fit_logreg(ts, tr, n_concepts, best)
        rows = rf == k
        oof[rows] = m.logits(ts.H[rows], _key(ts)[rows])
        models.append(m)
        chosen.append(float(best))
        tables.append(table)
    return LogregCrossFit(models, oof, chosen, tables)


# ------------------------------------------------------------------ null z


@dataclass
class NullStats:
    """Per fold model: m, sd [n_offsets, C] over that fold's OOF implied positions of the other
    concepts (``offsets`` gives the offset of each row of m / sd)."""

    offsets: np.ndarray
    m: list[np.ndarray]
    sd: list[np.ndarray]

    def z(self, k: int, S: np.ndarray, offset: np.ndarray) -> np.ndarray:
        oi = {int(o): i for i, o in enumerate(self.offsets)}
        idx = np.array([oi.get(int(o), -1) for o in offset])
        out = np.full(S.shape, np.nan)
        ok = idx >= 0
        out[ok] = (S[ok] - self.m[k][idx[ok]]) / self.sd[k][idx[ok]]
        return out


def null_stats(
    ts: P.TextSet, oof: np.ndarray, n_concepts: int, n_folds: int = P.N_FOLDS
) -> NullStats:
    rf, rc, imp = ts.row_fold, ts.row_concept, ts.row_implied
    offs = np.unique(_key(ts)[rc >= 0])
    ms, sds = [], []
    for k in range(n_folds):
        m = np.full((len(offs), n_concepts), np.nan)
        sd = np.full((len(offs), n_concepts), np.nan)
        for i, o in enumerate(offs):
            base = (rf == k) & imp & (rc >= 0) & (_key(ts) == o)
            for c in range(n_concepts):
                x = oof[base & (rc != c), c]
                x = x[np.isfinite(x)]
                if x.size > 1:
                    m[i, c] = x.mean()
                    s = x.std(ddof=1)
                    sd[i, c] = s if s > 0 else np.nan
        ms.append(m)
        sds.append(sd)
    return NullStats(offs, ms, sds)


def oof_z(ts: P.TextSet, oof: np.ndarray, ns: NullStats, n_folds: int = P.N_FOLDS) -> np.ndarray:
    """OOF scores z'd by their own fold's null."""
    out = np.full(oof.shape, np.nan)
    rf = ts.row_fold
    for k in range(n_folds):
        r = rf == k
        out[r] = ns.z(k, oof[r], _key(ts)[r])
    return out


def validity_implied(ts: P.TextSet, Z: np.ndarray, n_concepts: int) -> np.ndarray:
    rc, imp = ts.row_concept, ts.row_implied
    return np.array(
        [
            P.auroc(Z[imp & (rc == c), c], Z[imp & (rc != c) & (rc >= 0), c])
            for c in range(n_concepts)
        ]
    )


def implied_centroids(ts: P.TextSet, n_concepts: int) -> np.ndarray:
    """[C, d] grand-mean-centred implied-text centroids (mean over each concept's implied
    texts' window positions)."""
    rc, imp = ts.row_concept, ts.row_implied
    cent = np.stack(
        [ts.H[imp & (rc == c)].astype(np.float64).mean(axis=0) for c in range(n_concepts)]
    )
    return cent - cent.mean(axis=0, keepdims=True)


def score_eval(
    models_a: Sequence[P.FoldModel],
    models_b: Sequence[LogregModel],
    ns_a: NullStats,
    ns_b: NullStats,
    H: np.ndarray,
    ending: np.ndarray,
    offset: np.ndarray,
) -> dict[str, np.ndarray]:
    """Eval positions -> fold-averaged null z for both types, [P, C] each."""
    key = eval_key(ending, offset)
    za = np.nanmean(
        [ns_a.z(k, m.score(H, ending, offset), key) for k, m in enumerate(models_a)], axis=0
    )
    zb = np.nanmean([ns_b.z(k, m.logits(H, key), key) for k, m in enumerate(models_b)], axis=0)
    return {"a": za, "b": zb}
