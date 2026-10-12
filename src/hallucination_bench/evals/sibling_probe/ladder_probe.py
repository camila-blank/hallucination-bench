"""Plan 0005 step 3-4: per-layer entity-bank probes (the Ali case-study method, extended).

One (bank, layer) at a time (:func:`fit_bank_layer`):

* rows: bank entities that pass the knowledge filter, all 3 templates (training); references
  and the items are template-0 rows (the item's own wording);
* outer folds: GroupKFold(5) over entities, so every reference row and the item are scored by
  fold models that never saw them (items: the mean of the 5 fold models);
* probes, scored per class:
  - ``logreg``: standardise -> PCA(k) -> multinomial logistic (C = 0.01, Ali), the score is the
    class logit; k in {128, 256, 512} chosen per outer fold by an inner GroupKFold(3) over the
    training entities (criterion: mean one-vs-rest template-0 AUROC). PCA components are
    nested, so one PCA-512 fit gives all three k;
  - ``dom``: one-vs-rest difference of means, projected after centring on the fold's training
    mean (Ali);
  - ``wdm``: shrinkage-whitened DiffMean (plan 0003 rev 11 V2): d = (S_w + a I)^-1 d_dom,
    a = lam tr(S_w) / D, S_w the pooled within-(class, template) covariance of the training
    rows; lam by the same inner CV. Computed through the n x n Gram matrix (n < D).
* outputs per class: held-out template-0 AUROC (class vs other classes); per item: the item's
  scores, and its empirical p vs same-gold references (concept scores) and vs other-class
  references (gold score, the positive control).

sklearn is imported lazily (``uv run --with scikit-learn``); nothing here runs at import.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

PCA_KS = (128, 256, 512)
LR_C = 0.01
LAMS_W = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
N_OUTER = 5
N_INNER = 3
PROBE_TYPES = ("logreg", "dom", "wdm")


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):  # average ranks for ties
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def mean_auroc(S: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    vals = [
        auroc(S[y == c, c], S[y != c, c])
        for c in range(n_classes)
        if (y == c).any() and (y != c).any()
    ]
    return float(np.nanmean(vals)) if vals else float("nan")


def emp_p(score: float, refs: np.ndarray) -> float:
    refs = np.asarray(refs, float)
    return float((1 + np.sum(refs >= score)) / (len(refs) + 1))


# ------------------------------------------------------------------ DiffMean variants


def dom_fit(X: np.ndarray, y: np.ndarray, C: int) -> tuple[np.ndarray, np.ndarray]:
    mu0 = X.mean(0)
    dirs = np.stack(
        [
            X[y == c].mean(0) - X[y != c].mean(0) if (y == c).any() else np.zeros(X.shape[1])
            for c in range(C)
        ]
    )
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
    return mu0, dirs


class WhitenedDM:
    """Shrinkage-whitened DiffMean on one training set; ``directions(lam)`` is cheap after the
    Gram eigendecomposition. Groups for S_w: (class, template)."""

    def __init__(self, X: np.ndarray, y: np.ndarray, t: np.ndarray, C: int) -> None:
        X = X.astype(np.float64)
        self.mu0 = X.mean(0)
        g = y * 10 + t
        R = X.copy()
        for gid in np.unique(g):
            m = g == gid
            R[m] -= X[m].mean(0)
        self.m = max(len(X) - len(np.unique(g)), 1)
        self.R = R
        G = R @ R.T
        E, U = np.linalg.eigh(G)
        self.E, self.U = np.maximum(E, 0), U
        self.trS = float(np.trace(G)) / self.m  # tr(S_w)
        self.D = X.shape[1]
        self.d0 = np.stack(
            [
                X[y == c].mean(0) - X[y != c].mean(0) if (y == c).any() else np.zeros(X.shape[1])
                for c in range(C)
            ]
        )
        self.Rd0 = R @ self.d0.T  # [n, C]

    def directions(self, lam: float) -> np.ndarray:
        a = lam * self.trS / self.D
        # (R^T R / m + a I)^-1 v = (1/a) [v - R^T (m a I + R R^T)^-1 R v]
        inner = self.U @ ((self.U.T @ self.Rd0) / (self.m * a + self.E)[:, None])
        W = (self.d0.T - self.R.T @ inner) / a  # [D, C]
        W /= np.maximum(np.linalg.norm(W, axis=0, keepdims=True), 1e-12)
        return W.T


# ------------------------------------------------------------------ one (bank, layer)


def _logreg_models(Xtr, ytr, ks, C):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Xtr)
    Z = sc.transform(Xtr)
    kmax = min(max(ks), Z.shape[0] - 1, Z.shape[1])
    pca = PCA(n_components=kmax, random_state=0, svd_solver="randomized").fit(Z)
    P = pca.transform(Z)
    out = {}
    for k in ks:
        kk = min(k, kmax)
        clf = LogisticRegression(C=C, max_iter=3000).fit(P[:, :kk], ytr)
        out[k] = (sc, pca, kk, clf)
    return out


def _logreg_score(model, X, C=None):
    """Class logits [n, C]; a class absent from the training fold scores -inf-like (-1e6)."""
    sc, pca, kk, clf = model
    d = clf.decision_function(pca.transform(sc.transform(X))[:, :kk])
    if C is None or len(clf.classes_) == C:
        return d
    out = np.full((len(X), C), -1e6)
    out[:, clf.classes_] = d
    return out


def fit_bank_layer(
    X: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    groups: np.ndarray,
    X_items: np.ndarray,
    n_classes: int,
    ks: Sequence[int] = PCA_KS,
    lams: Sequence[float] = LAMS_W,
) -> dict[str, Any]:
    """Cross-fitted scores. Returns {"oof": {probe: [n, C]}, "items": {probe: [m, C]},
    "hyper": {"k": [...], "lam": [...]}, "inner": {...}}."""
    from sklearn.model_selection import GroupKFold

    X = X.astype(np.float32)
    C = n_classes
    oof = {p: np.zeros((len(X), C)) for p in PROBE_TYPES}
    its = {p: np.zeros((len(X_items), C)) for p in PROBE_TYPES}
    hyper: dict[str, list[Any]] = {"k": [], "lam": []}
    inner_tab: dict[str, list[Any]] = {"k": [], "lam": []}
    outer = list(GroupKFold(n_splits=N_OUTER).split(X, y, groups))
    for tr, te in outer:
        # inner selection on the training entities only
        inner = list(GroupKFold(n_splits=N_INNER).split(X[tr], y[tr], groups[tr]))
        sk = {k: [] for k in ks}
        sl = {lam: [] for lam in lams}
        for itr, ite in inner:
            a, b = tr[itr], tr[ite]
            t0 = t[b] == 0
            mods = _logreg_models(X[a], y[a], ks, LR_C)
            for k in ks:
                sk[k].append(mean_auroc(_logreg_score(mods[k], X[b][t0], C), y[b][t0], C))
            w = WhitenedDM(X[a], y[a], t[a], C)
            for lam in lams:
                S = (X[b][t0] - w.mu0) @ w.directions(lam).T
                sl[lam].append(mean_auroc(S, y[b][t0], C))
        kbest = max(ks, key=lambda k: np.nanmean(sk[k]))
        lbest = max(lams, key=lambda v: np.nanmean(sl[v]))
        hyper["k"].append(int(kbest))
        hyper["lam"].append(float(lbest))
        inner_tab["k"].append({str(k): round(float(np.nanmean(v)), 4) for k, v in sk.items()})
        inner_tab["lam"].append({str(k): round(float(np.nanmean(v)), 4) for k, v in sl.items()})
        mod = _logreg_models(X[tr], y[tr], [kbest], LR_C)[kbest]
        oof["logreg"][te] = _logreg_score(mod, X[te], C)
        its["logreg"] += _logreg_score(mod, X_items, C) / N_OUTER
        mu0, dirs = dom_fit(X[tr], y[tr], C)
        oof["dom"][te] = (X[te] - mu0) @ dirs.T
        its["dom"] += ((X_items - mu0) @ dirs.T) / N_OUTER
        w = WhitenedDM(X[tr], y[tr], t[tr], C)
        Wd = w.directions(lbest)
        oof["wdm"][te] = (X[te] - w.mu0) @ Wd.T
        its["wdm"] += ((X_items - w.mu0) @ Wd.T) / N_OUTER
    return {"oof": oof, "items": its, "hyper": hyper, "inner": inner_tab}


def class_aurocs(
    oof: Mapping[str, np.ndarray], y: np.ndarray, t: np.ndarray, classes: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Held-out template-0 AUROC per probe and class (class vs other classes)."""
    m = t == 0
    return {
        p: {c: auroc(S[m & (y == i), i], S[m & (y != i), i]) for i, c in enumerate(classes)}
        for p, S in oof.items()
    }


def item_cell(
    item_scores: Mapping[str, np.ndarray],
    oof: Mapping[str, np.ndarray],
    y: np.ndarray,
    t: np.ndarray,
    classes: Sequence[str],
    gold: str,
    aurocs: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """The probe cell of one item at one layer (input to ``ladder.fires``)."""
    ci = {c: i for i, c in enumerate(classes)}
    m0 = t == 0
    if gold not in ci:
        return {"classes": list(classes), "na_reason": "gold not a bank class"}
    g = ci[gold]
    same = m0 & (y == g)
    other = m0 & (y != g)
    cell: dict[str, Any] = {
        "classes": list(classes),
        "n_same": int(same.sum()),
        "auroc": {p: dict(aurocs[p]) for p in item_scores},
        "p_concept": {},
        "rank_concept": {},
        "p_gold": {},
        "rank_gold": {},
        "z_concept": {},
        "score": {},
    }
    for p, s in item_scores.items():
        S = oof[p]
        cell["p_concept"][p] = {c: emp_p(s[i], S[same, i]) for c, i in ci.items()}
        cell["rank_concept"][p] = {
            c: float(np.mean(S[same, i] < s[i])) if same.any() else None for c, i in ci.items()
        }
        cell["z_concept"][p] = {
            c: float((s[i] - S[same, i].mean()) / (S[same, i].std() + 1e-9)) if same.any() else None
            for c, i in ci.items()
        }
        cell["p_gold"][p] = emp_p(s[g], S[other, g])
        cell["rank_gold"][p] = float(np.mean(S[other, g] < s[g]))
        cell["score"][p] = {c: float(s[i]) for c, i in ci.items()}
    return cell


def gather_rows(runs: Sequence[tuple[str, int, int]], gids: np.ndarray, li: int) -> np.ndarray:
    """Rows ``gids`` (global ids: run offset + row index) at layer index ``li`` from memory-mapped
    ``acts.npy`` files [n, n_layers, d]."""
    out = None
    for path, off, n in runs:
        sel = np.flatnonzero((gids >= off) & (gids < off + n))
        if not sel.size:
            continue
        local = gids[sel] - off
        srt = np.argsort(local)
        part = np.asarray(np.load(path, mmap_mode="r")[local[srt], li], dtype=np.float32)
        if out is None:
            out = np.zeros((len(gids), part.shape[1]), dtype=np.float32)
        out[sel[srt]] = part
    assert out is not None
    return out


def probe_task(a: tuple, threads: int = 1) -> tuple[str, int, dict[str, Any], dict[str, Any]]:
    """One (bank, layer): fit, class AUROCs, item cells. ``a`` = (bank, layer, layer index,
    runs, rows, y, t, groups, item_rows, classes, item_golds, item_ids)."""
    from threadpoolctl import threadpool_limits

    b, L, li, runs, rows, y, t, groups, item_rows, classes, item_golds, item_ids = a
    y, t, groups = np.asarray(y), np.asarray(t), np.asarray(groups)
    X = gather_rows(runs, np.asarray(rows), li)
    Xi = gather_rows(runs, np.asarray(item_rows), li)
    with threadpool_limits(threads):
        r = fit_bank_layer(X, y, t, groups, Xi, len(classes))
    au = class_aurocs(r["oof"], y, t, classes)
    cells = {
        iid: item_cell({p: r["items"][p][j] for p in PROBE_TYPES}, r["oof"], y, t, classes, g, au)
        for j, (iid, g) in enumerate(zip(item_ids, item_golds, strict=True))
    }
    print(
        f"  [probes] {b} L{L} k={r['hyper']['k']} lam={r['hyper']['lam']} median AUROC "
        + " ".join(f"{p}={np.nanmedian(list(au[p].values())):.3f}" for p in PROBE_TYPES),
        flush=True,
    )
    return b, L, {"auroc": au, "hyper": r["hyper"], "inner": r["inner"]}, cells
