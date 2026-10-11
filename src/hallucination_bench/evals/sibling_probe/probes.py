"""DiffMean concept probes, cross-fit by category-level folds (plan 0003 §Phase E). Pure numpy.

This module is imported by the Modal job (``scripts/sibling_probe/probes_modal.py``, copied into
the container as a single file), so it imports nothing from the package.

Vocabulary (one category, one layer L):

* a **text** is a probe training text (concept index ``c >= 0``) or a neutral text
  (``c = NEUTRAL``); its **ending** is the request-sentence index; its read window is the
  ``from_last_sentence_start`` positions, aligned across rows by **offset from the end**
  (``offset = n_pos - 1 - pos``; 0 = the last template token);
* **folds**: every text of the category (all concepts AND the neutral texts) gets one seeded fold
  id in ``0..n_folds-1``, balanced by (concept, ending, kind) (:func:`assign_folds`);
* **fold model k** (:func:`fit_fold`) is fitted on the concept texts outside fold k only:
  ``d_c`` = mean of c's per-text window means - the equal-weighted mean of the other concepts'
  per-concept means; ``mu[e, o]`` = equal-weighted mean over concepts of the training residuals at
  (ending e, offset o); ``sigma[e, o, c]`` = sd of the projected training residuals
  ``(h - mu[e, o]) . d_c / |d_c|`` at (e, o), pooled over the category's concepts;
* **score** ``s_c = ((h - mu[e, o]) . d_c / |d_c|) / sigma[e, o, c]`` (:meth:`FoldModel.score`);
* **out-of-fold** (OOF) scores: every fold-k text is scored by model k only (:func:`cross_fit`);
* **thresholds** per fold k, from fold k's OOF rows only (:func:`fold_thresholds`): ``tau_cat``
  (95th pct of s_c over the other concepts' positions), ``tau_cg`` / ``tau_cg_all`` (95th pct of
  s_c over g's implied / all positions), ``tau_neu`` (neutral positions), ``tau_ceil`` (95th pct
  of s_g over the other concepts' implied positions = tau(g | g') pooled over g'), and the z
  statistics ``z_mu`` / ``z_sd`` of s_c over g's implied positions;
* **eval scoring** (:func:`score_eval`): each eval position is scored by every fold model k and
  thresholded against fold k's own thresholds; detection indicators and z are averaged over folds.
"""

from __future__ import annotations

import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

NEUTRAL = -1
N_FOLDS = 4
FOLD_SEED = 0
Q = 0.95
# plan 0003 rev 10: lowered from 0.85 (position-level OOF AUROC; set before any judge/score result)
VALIDITY_MIN_AUROC = 0.70
VALIDITY_STRICT_AUROC = 0.85  # robustness row only
TRANSFER_MIN_AUROC = 0.70
KINDS = ("explicit", "implied", "neutral")

# ------------------------------------------------------------------ statistics


def q95(x: np.ndarray) -> float:
    """The 95th percentile (numpy's default linear interpolation); NaN for an empty set."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, Q)) if x.size else float("nan")


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score of a positive > score of a negative) + 0.5 P(tie) (Mann-Whitney); NaN if a side
    is empty."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not pos.size or not neg.size:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(allv.size, dtype=np.float64)
    sv = allv[order]
    # average ranks over ties
    i = 0
    n = sv.size
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    r_pos = ranks[: pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def weighted_quantile(x: np.ndarray, w: np.ndarray, q: float = Q) -> float:
    """Quantile of ``x`` under integer frequency weights ``w`` (equal to :func:`q95` on the
    expanded sample)."""
    keep = w > 0
    x, w = x[keep], w[keep]
    if not x.size:
        return float("nan")
    order = np.argsort(x, kind="mergesort")
    x, w = x[order], w[order]
    n = int(w.sum())
    h = (n - 1) * q
    lo, hi = int(np.floor(h)), int(np.ceil(h))
    cw = np.cumsum(w)
    xlo = x[np.searchsorted(cw, lo + 1)]
    xhi = x[np.searchsorted(cw, hi + 1)]
    return float(xlo + (h - lo) * (xhi - xlo))


def threshold_ci(
    scores: np.ndarray, text_of_row: np.ndarray, *, n_boot: int = 500, seed: int = 0
) -> tuple[float, float]:
    """95% text-level bootstrap CI of the 95th percentile of ``scores`` (rows grouped by text)."""
    if not scores.size:
        return float("nan"), float("nan")
    texts, inv = np.unique(text_of_row, return_inverse=True)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        cnt = rng.multinomial(texts.size, np.full(texts.size, 1.0 / texts.size))
        out[b] = weighted_quantile(scores, cnt[inv])
    return float(np.nanquantile(out, 0.025)), float(np.nanquantile(out, 0.975))


# ------------------------------------------------------------------ folds


def assign_folds(
    texts: Sequence[Mapping[str, object]],
    *,
    n_folds: int = N_FOLDS,
    seed: int = FOLD_SEED,
    salt: str = "",
) -> dict[str, int]:
    """Seeded fold id per text id, balanced by (concept, ending, kind): inside each stratum the
    texts are permuted and dealt round-robin, starting where the previous stratum stopped.
    Strata are dealt in sorted order, so the strata of one (concept, ending) are consecutive and
    the (concept, ending) and fold totals also differ by at most one; each fold gets ~equal
    implied (and explicit) texts per concept. ``texts`` rows carry ``id``, ``concept`` (None for
    neutral), ``request_idx`` and ``kind`` (missing -> ''). ``salt`` (the category name)
    decorrelates categories."""
    strata: dict[tuple[str, int, str], list[str]] = {}
    for t in texts:
        c = t.get("concept")
        k = (
            "\0neutral" if c is None else str(c),
            int(t["request_idx"]),  # type: ignore[arg-type]
            str(t.get("kind") or ""),
        )
        strata.setdefault(k, []).append(str(t["id"]))
    rng = np.random.default_rng([seed, zlib.crc32(salt.encode())])
    out: dict[str, int] = {}
    start = 0
    for k in sorted(strata):
        ids = sorted(strata[k])
        perm = rng.permutation(len(ids))
        for j, i in enumerate(perm):
            out[ids[i]] = (start + j) % n_folds
        start = (start + len(ids)) % n_folds
    return out


# ------------------------------------------------------------------ data container


@dataclass
class TextSet:
    """The window positions of a set of texts at one layer, stacked.

    ``H`` [P, d] float32 residuals; per row: ``text`` (index into the per-text arrays),
    ``ending``, ``offset``. Per text: ``ids``, ``concept`` (index or NEUTRAL), ``kind``,
    ``ending``, ``fold`` (-1 for eval items)."""

    H: np.ndarray
    text: np.ndarray
    ending: np.ndarray
    offset: np.ndarray
    ids: list[str]
    concept: np.ndarray
    kind: list[str]
    t_ending: np.ndarray
    fold: np.ndarray

    @classmethod
    def build(
        cls,
        windows: Sequence[np.ndarray],
        ids: Sequence[str],
        concept: Sequence[int],
        kind: Sequence[str],
        ending: Sequence[int],
        fold: Sequence[int] | None = None,
        offsets: Sequence[Sequence[int]] | None = None,
    ) -> TextSet:
        """``windows[i]`` [n_i, d]: text i's window positions in order (offset n_i - 1 .. 0
        unless ``offsets`` is given)."""
        rows_t, rows_e, rows_o = [], [], []
        for i, w in enumerate(windows):
            n = int(w.shape[0])
            off = np.arange(n - 1, -1, -1) if offsets is None else np.asarray(offsets[i])
            rows_t.append(np.full(n, i))
            rows_e.append(np.full(n, int(ending[i])))
            rows_o.append(off)
        d = int(windows[0].shape[1]) if windows else 0
        return cls(
            H=np.concatenate([np.asarray(w, dtype=np.float32) for w in windows])
            if windows
            else np.zeros((0, d), np.float32),
            text=np.concatenate(rows_t).astype(np.int64) if windows else np.zeros(0, np.int64),
            ending=np.concatenate(rows_e).astype(np.int64) if windows else np.zeros(0, np.int64),
            offset=np.concatenate(rows_o).astype(np.int64) if windows else np.zeros(0, np.int64),
            ids=list(ids),
            concept=np.asarray(concept, dtype=np.int64),
            kind=list(kind),
            t_ending=np.asarray(ending, dtype=np.int64),
            fold=np.asarray(fold if fold is not None else [-1] * len(ids), dtype=np.int64),
        )

    @property
    def row_concept(self) -> np.ndarray:
        return self.concept[self.text]

    @property
    def row_fold(self) -> np.ndarray:
        return self.fold[self.text]

    @property
    def row_implied(self) -> np.ndarray:
        imp = np.array([k == "implied" for k in self.kind], dtype=bool)
        return imp[self.text]

    def text_means(self) -> np.ndarray:
        """[T, d] per-text mean over its window positions."""
        cached = getattr(self, "_means", None)
        if cached is not None:
            return cached
        T = len(self.ids)
        cnt = np.bincount(self.text, minlength=T).astype(np.float64)
        if self.text.size and np.all(np.diff(self.text) >= 0) and np.all(cnt > 0):
            starts = np.concatenate([[0], np.cumsum(cnt)[:-1]]).astype(np.int64)
            out = np.add.reduceat(self.H.astype(np.float64), starts, axis=0)
        else:
            out = np.zeros((T, self.H.shape[1]), np.float64)
            np.add.at(out, self.text, self.H)
        out = out / np.maximum(cnt, 1)[:, None]
        object.__setattr__(self, "_means", out)
        return out


# ------------------------------------------------------------------ one fold model


@dataclass
class FoldModel:
    """Fitted on one fold's training texts. ``D`` [C, d] raw DiffMean directions, ``U`` unit
    directions; ``keys`` (ending, offset) -> row of ``muU`` [K, C] (mu . u_c) and ``sigma``
    [K, C]; ``train`` the text indices it saw (directions AND centring AND scale)."""

    D: np.ndarray
    U: np.ndarray
    keys: dict[tuple[int, int], int]
    muU: np.ndarray
    sigma: np.ndarray
    train: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))

    def score(self, H: np.ndarray, ending: np.ndarray, offset: np.ndarray) -> np.ndarray:
        """[P, C] position-aligned scores; NaN at an (ending, offset) the model never saw."""
        HU = np.asarray(H, dtype=np.float64) @ self.U.T
        ki = np.array(
            [self.keys.get((int(e), int(o)), -1) for e, o in zip(ending, offset, strict=True)],
            dtype=np.int64,
        )
        out = np.full(HU.shape, np.nan)
        ok = ki >= 0
        out[ok] = (HU[ok] - self.muU[ki[ok]]) / self.sigma[ki[ok]]
        return out


def fit_fold(ts: TextSet, train_texts: np.ndarray, n_concepts: int) -> FoldModel:
    """DiffMean + centring + scale on the concept texts in ``train_texts`` (text indices;
    neutral texts in it are ignored)."""
    train_texts = np.asarray(train_texts, dtype=np.int64)
    tr = train_texts[ts.concept[train_texts] >= 0]
    V = ts.text_means()
    C = n_concepts
    m = np.zeros((C, ts.H.shape[1]))
    for c in range(C):
        sel = tr[ts.concept[tr] == c]
        if not sel.size:
            raise ValueError(f"concept {c} has no training texts")
        m[c] = V[sel].mean(axis=0)
    others = (m.sum(axis=0, keepdims=True) - m) / max(C - 1, 1)
    D = m - others
    U = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
    in_train = np.zeros(len(ts.ids), dtype=bool)
    in_train[tr] = True
    row_tr = in_train[ts.text]
    rc = ts.row_concept
    keys: dict[tuple[int, int], int] = {}
    mus, sds = [], []
    pairs = sorted(set(zip(ts.ending[row_tr].tolist(), ts.offset[row_tr].tolist(), strict=True)))
    for e, o in pairs:
        at = row_tr & (ts.ending == e) & (ts.offset == o)
        per_c = [ts.H[at & (rc == c)].mean(axis=0) for c in range(C) if np.any(at & (rc == c))]
        mu = np.mean(per_c, axis=0)
        p = (ts.H[at].astype(np.float64) - mu) @ U.T
        sd = p.std(axis=0, ddof=1) if p.shape[0] > 1 else np.full(C, np.nan)
        keys[(e, o)] = len(mus)
        mus.append(mu @ U.T)
        sds.append(np.where(sd > 0, sd, np.nan))
    return FoldModel(D=D, U=U, keys=keys, muU=np.array(mus), sigma=np.array(sds), train=np.sort(tr))


# ------------------------------------------------------------------ cross-fitting


@dataclass
class CrossFit:
    models: list[FoldModel]
    oof: np.ndarray  # [P, C] every row scored by the model of its text's fold
    n_folds: int


def cross_fit(ts: TextSet, n_concepts: int, n_folds: int = N_FOLDS) -> CrossFit:
    """Fit one model per fold on the other folds' texts; score each fold's rows with its own
    model only."""
    models = []
    oof = np.full((ts.H.shape[0], n_concepts), np.nan)
    T = np.arange(len(ts.ids))
    rf = ts.row_fold
    for k in range(n_folds):
        mdl = fit_fold(ts, T[ts.fold != k], n_concepts)
        assert not np.isin(np.flatnonzero(ts.fold == k), mdl.train).any()
        models.append(mdl)
        rows = rf == k
        if rows.any():
            oof[rows] = mdl.score(ts.H[rows], ts.ending[rows], ts.offset[rows])
    return CrossFit(models, oof, n_folds)


def validity(ts: TextSet, oof: np.ndarray, n_concepts: int) -> np.ndarray:
    """[C] position-level OOF AUROC: c's positions vs the other concepts' (neutral excluded)."""
    rc = ts.row_concept
    return np.array(
        [auroc(oof[rc == c, c], oof[(rc != c) & (rc >= 0), c]) for c in range(n_concepts)]
    )


# ------------------------------------------------------------------ thresholds


@dataclass
class Thresholds:
    """One fold's thresholds (all [C] or [C, G=C], indexed [c] / [c, g])."""

    tau_cat: np.ndarray
    tau_cg: np.ndarray
    tau_cg_all: np.ndarray
    tau_neu: np.ndarray
    tau_ceil: np.ndarray
    z_mu: np.ndarray
    z_sd: np.ndarray
    n_cg: np.ndarray  # positions behind tau_cg / z per g

    def to_dict(self) -> dict[str, list]:
        return {
            k: np.where(np.isfinite(v), v, np.nan).round(6).tolist()
            for k, v in self.__dict__.items()
        }


def thresholds_from(
    S: np.ndarray, row_concept: np.ndarray, row_implied: np.ndarray, n_concepts: int
) -> Thresholds:
    """Thresholds and z statistics from a block of OOF rows ``S`` [N, C]."""
    C = n_concepts
    tau_cat = np.array([q95(S[(row_concept != c) & (row_concept >= 0), c]) for c in range(C)])
    tau_neu = np.array([q95(S[row_concept == NEUTRAL, c]) for c in range(C)])
    tau_ceil = np.array(
        [q95(S[(row_concept != c) & (row_concept >= 0) & row_implied, c]) for c in range(C)]
    )
    tau_cg = np.full((C, C), np.nan)
    tau_cg_all = np.full((C, C), np.nan)
    z_mu = np.full((C, C), np.nan)
    z_sd = np.full((C, C), np.nan)
    n_cg = np.zeros(C, dtype=np.int64)
    for g in range(C):
        imp = (row_concept == g) & row_implied
        allg = row_concept == g
        n_cg[g] = int(imp.sum())
        for c in range(C):
            x = S[imp, c]
            x = x[np.isfinite(x)]
            tau_cg[c, g] = q95(x)
            tau_cg_all[c, g] = q95(S[allg, c])
            if x.size > 1:
                z_mu[c, g] = x.mean()
                sd = x.std(ddof=1)
                z_sd[c, g] = sd if sd > 0 else np.nan
    return Thresholds(tau_cat, tau_cg, tau_cg_all, tau_neu, tau_ceil, z_mu, z_sd, n_cg)


def fold_thresholds(ts: TextSet, cf: CrossFit, n_concepts: int) -> list[Thresholds]:
    """Thresholds of each fold k from fold k's OOF rows only."""
    rf, rc, ri = ts.row_fold, ts.row_concept, ts.row_implied
    return [
        thresholds_from(cf.oof[rf == k], rc[rf == k], ri[rf == k], n_concepts)
        for k in range(cf.n_folds)
    ]


# ------------------------------------------------------------------ eval scoring


@dataclass
class EvalScores:
    """Per eval row (position) and concept c, averaged over fold models: ``s`` (raw score),
    ``z`` (z(c | g), NaN at c = g), ``det_cat`` / ``det_cg`` / ``det_cg_all`` / ``det_neu``
    (share of the fold models whose threshold the score exceeds), ``det_ceil`` (c = g only:
    against tau(g | g')), all [P, C]."""

    s: np.ndarray
    z: np.ndarray
    det_cat: np.ndarray
    det_cg: np.ndarray
    det_cg_all: np.ndarray
    det_neu: np.ndarray
    det_ceil: np.ndarray


def _gt(s: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """1.0 / 0.0 indicator, NaN where either side is NaN."""
    out = (s > tau).astype(np.float64)
    out[~(np.isfinite(s) & np.isfinite(tau))] = np.nan
    return out


def score_eval(
    models: Sequence[FoldModel],
    ths: Sequence[Thresholds],
    H: np.ndarray,
    ending: np.ndarray,
    offset: np.ndarray,
    gold: np.ndarray,
) -> EvalScores:
    """Score eval positions (``gold`` [P] = the row's item gold concept index) with every fold
    model k, threshold against fold k's thresholds, average over folds."""
    P = H.shape[0]
    acc: dict[str, list[np.ndarray]] = {
        k: [] for k in ("s", "z", "det_cat", "det_cg", "det_cg_all", "det_neu", "det_ceil")
    }
    gold = np.asarray(gold, dtype=np.int64)
    for mdl, th in zip(models, ths, strict=True):
        s = mdl.score(H, ending, offset)
        C = s.shape[1]
        cs = np.arange(C)
        is_gold = cs[None, :] == gold[:, None]
        acc["s"].append(s)
        z = (s - th.z_mu[:, gold].T) / th.z_sd[:, gold].T
        z[is_gold] = np.nan
        acc["z"].append(z)
        acc["det_cat"].append(_gt(s, th.tau_cat[None, :]))
        cg = _gt(s, th.tau_cg[:, gold].T)
        cg[is_gold] = np.nan
        acc["det_cg"].append(cg)
        cga = _gt(s, th.tau_cg_all[:, gold].T)
        cga[is_gold] = np.nan
        acc["det_cg_all"].append(cga)
        acc["det_neu"].append(_gt(s, th.tau_neu[None, :]))
        ceil = _gt(s, th.tau_ceil[None, :])
        ceil[~is_gold] = np.nan
        acc["det_ceil"].append(ceil)
    import warnings

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices (c = g) stay NaN
        out = {
            k: np.nanmean(np.stack(v), axis=0) if v else np.zeros((P, 0)) for k, v in acc.items()
        }
    return EvalScores(**out)


# ------------------------------------------------------------------ unembedding check


def unembed_cosines(D: np.ndarray, W: np.ndarray) -> np.ndarray:
    """cos(d, w) for each direction row of ``D`` [C, d] and unembedding row of ``W`` [n, d]."""
    Dn = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
    Wn = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)
    return Dn @ Wn.T


# ------------------------------------------------------------------ one (category, layer)


@dataclass
class CategoryLayerResult:
    cf: CrossFit
    ths: list[Thresholds]
    valid_auroc: np.ndarray
    pooled: Thresholds  # thresholds from all folds' OOF rows (reporting only)


def run_category_layer(ts: TextSet, n_concepts: int, n_folds: int = N_FOLDS) -> CategoryLayerResult:
    cf = cross_fit(ts, n_concepts, n_folds)
    ths = fold_thresholds(ts, cf, n_concepts)
    va = validity(ts, cf.oof, n_concepts)
    pooled = thresholds_from(cf.oof, ts.row_concept, ts.row_implied, n_concepts)
    return CategoryLayerResult(cf, ths, va, pooled)
