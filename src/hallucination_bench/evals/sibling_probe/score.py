"""Sibling-probe scoring and the verdict (plan 0003 §Phase F, revision 9).

Unit = (cell, concept) per arm: arm x item x layer x pos x concept of the item's category; the
probe score is the cell's own layer. A **sibling** is a concept that is neither the gold nor
named in the prompt.

**Pair classes** (:func:`classify_pair`), from the judge's mention records:

* prose arms: no alias hit for c in the cell's readout text -> ``unmentioned``; otherwise the
  verdicts of c's hits decide, in this order: any pending -> ``pending``; any ``correct`` ->
  ``correct``; any not-correct claim verdict (``score.M_VERDICTS``) -> ``M`` (sibling
  hallucination candidate); any ``failed`` -> ``failed``; else ``non_claim``;
* J-lens (token-eligible concepts only; others are ``ineligible``): a mention needs BOTH a raw
  top-10 token-alias hit AND a summary alias hit (verdicts as above); token-only ->
  ``token_only``; summary-only -> ``summary_only``; neither -> ``unmentioned``; a cell without a
  summary -> ``no_summary`` (unmentioned cannot be established);
* every class other than ``M`` and ``unmentioned`` is excluded from both sides of Tests 1-3 and
  the within-item AUROC and only counted in its own row.

**Gates**: probe validity (OOF AUROC >= 0.70, plan rev 10) and the per-(concept, layer)
transfer gate (eval-site AUROC of c's probe score, c-gold items' cells vs the other-gold items'
cells of the category, >= 0.70); a (category, layer) is transfer-passing if at least half its
concepts pass transfer. A pair is **eligible** iff its (category, layer) is transfer-passing and
c passes both gates at that layer; headline numbers cover eligible pairs only.

**Estimands** (each per arm, item-clustered bootstrap, 2,000 draws, seeded):

* beta1 (Test 1): the ``mentioned`` coefficient of OLS z(c|g) ~ mentioned + offset dummies +
  (item, layer, c) fixed effects over M + U_matched (U_matched = the same (item, layer, c) at the
  arm's cells that do not mention c), by the within transform;
* beta2 (Test 2): mean over M cells of [mean z of mentioned siblings - mean z of unmentioned
  siblings at the same cell];
* within-item AUROC of z at mention vs non-mention cells per (item, layer, c), averaged;
* Test 3: LPM detected ~ mentioned + cell FE + (item, c) FE (alternating projections);
* probe-verified hallucination rate = 1 - detection rate over M (tau_cat headline, tau(c|g)
  strict, tau_neutral robustness; all-sibling-mentions row under tau_cat), with the ceiling
  (gold-equivalent reference: the gold probe at gold-mention cells; tau_cat -> det_cat,
  tau(c|g) -> tau(g|g'), tau_neutral -> det_neu) and the floor (detection over U_matched).

Every fixed-effect group is nested within an item, so the residualisation is the same under any
item reweighting; bootstraps therefore reuse per-item sufficient statistics (exact).
"""

import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from hallucination_bench import REPO_ROOT
from hallucination_bench import taxonomy_v2 as tx
from hallucination_bench.evals.sibling_probe.probes import (
    TRANSFER_MIN_AUROC,
    VALIDITY_MIN_AUROC,
    VALIDITY_STRICT_AUROC,
    auroc,
)

SCORE_DIR = REPO_ROOT / "outputs" / "sibling" / "score"
N_BOOT = 2000
BOOT_SEED = 0
M_VERDICTS = frozenset(
    {tx.V_HALLUCINATION, tx.V_PROBE, tx.V_ALTERNATIVE, tx.V_UNRESOLVED, tx.V_JUNK}
)
MENTION_CLASSES = ("M", "correct", "non_claim", "failed")  # proper mentions (judge verdicts)
THRESHOLDS: dict[str, str] = {  # row name -> eval-score field (detection indicator)
    "tau_cat": "det_cat",
    "tau_cg": "det_cg",
    "tau_neutral": "det_neu",
    "tau_cg_all": "det_cg_all",
}
CEILING_FIELD: dict[str, str] = {  # same threshold for the gold-equivalent reference
    "tau_cat": "det_cat",
    "tau_cg": "det_ceil",
    "tau_neutral": "det_neu",
    "tau_cg_all": "det_ceil",
}
PRIMARY, SECONDARY, CONTROL = "s3d-rl600", "nla-rl-L42", "jlens"
SCORE_FIELDS = ("s", "z", "det_cat", "det_cg", "det_cg_all", "det_neu", "det_ceil")

# ------------------------------------------------------------------ pair classes


def _verdict_class(verdicts: Iterable[str | None]) -> str:
    v = list(verdicts)
    if any(x is None for x in v):
        return "pending"
    if tx.V_CORRECT in v:
        return "correct"
    if any(x in M_VERDICTS for x in v):
        return "M"
    if tx.FAILED in v:
        return "failed"
    return "non_claim"


def classify_pair(rec: Mapping[str, Any], concept: str, *, token_eligible: bool = True) -> str:
    """The class of (cell, concept) for a judge cell record (see the module docstring); the
    gold and prompt-named concepts get ``gold`` / ``named``."""
    if concept == rec["gold"]:
        return "gold"
    if concept in set(rec.get("named") or ()):
        return "named"
    if rec.get("empty"):
        return "empty"
    ms = [m for m in rec.get("mentions") or () if m["concept"] == concept]
    if rec.get("kind") == "tokens":
        if not token_eligible:
            return "ineligible"
        tok = concept in (rec.get("token_hits") or {})
        if rec.get("summary") is None:
            return "no_summary"
        if tok and ms:
            return _verdict_class(m.get("verdict") for m in ms)
        if tok:
            return "token_only"
        if ms:
            return "summary_only"
        return "unmentioned"
    if not ms:
        return "unmentioned"
    return _verdict_class(m.get("verdict") for m in ms)


def gold_mentioned(rec: Mapping[str, Any]) -> bool:
    g = rec["gold"]
    hit = any(m["concept"] == g for m in rec.get("mentions") or ())
    if rec.get("kind") == "tokens":
        return hit and g in (rec.get("token_hits") or {})
    return hit


def pair_leaf_scope(rec: Mapping[str, Any], concept: str) -> tuple[str | None, str | None]:
    """The leaf of c's first M-verdict mention, and ``user`` if any of c's M-verdict mentions
    is in a ``scope = user`` segment (else ``other`` / None)."""
    ms = [
        m
        for m in rec.get("mentions") or ()
        if m["concept"] == concept and m.get("verdict") in M_VERDICTS
    ]
    if not ms:
        return None, None
    scopes = {m.get("scope") for m in ms}
    scope = "user" if "user" in scopes else ("other" if "other" in scopes else None)
    return ms[0].get("leaf"), scope


# ------------------------------------------------------------------ probe tables / gates


def index_scores(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, int, int, str], tuple]:
    def f(x: Any) -> float:
        return float("nan") if x is None else float(x)

    return {
        (r["id"], int(r["layer"]), int(r["pos"]), r["concept"]): tuple(
            f(r.get(k)) for k in SCORE_FIELDS
        )
        for r in rows
    }


def transfer_auroc(
    scores: Mapping[tuple[str, int, int, str], tuple],
    items: Mapping[str, Mapping[str, Any]],
    concepts: Mapping[str, Sequence[str]],
) -> dict[tuple[str, str, int], float]:
    """(category, concept, layer) -> AUROC of the fold-averaged score s_c at c-gold items' cells
    vs the other-gold items' cells of the category (same layer); NaN if a side is empty. An item
    whose gold is named in its prompt is not a positive for its gold (the prompt, not the
    implication, would carry the concept); it stays a negative for the other concepts."""
    by: dict[tuple[str, str, int], tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    for (iid, layer, _pos, c), v in scores.items():
        it = items.get(iid)
        if it is None:
            continue
        if it["gold"] == c and c in set(it.get("named") or ()):
            continue
        side = 0 if it["gold"] == c else 1
        by[(it["category"], c, layer)][side].append(v[0])
    out = {}
    layers = sorted({k[2] for k in by})
    for cat, cs in concepts.items():
        for c in cs:
            for L in layers:
                pos, neg = by.get((cat, c, L), ([], []))
                out[(cat, c, L)] = auroc(np.array(pos), np.array(neg))
    return out


@dataclass
class Gates:
    validity: dict[tuple[str, str, int], float]
    transfer: dict[tuple[str, str, int], float]
    concepts: dict[str, list[str]]
    min_validity: float = VALIDITY_MIN_AUROC

    def concept_pass(self, cat: str, c: str, L: int) -> bool:
        v = self.validity.get((cat, c, L), float("nan"))
        t = self.transfer.get((cat, c, L), float("nan"))
        return bool(v >= self.min_validity and t >= TRANSFER_MIN_AUROC)

    def transfer_pass(self, cat: str, c: str, L: int) -> bool:
        return bool(self.transfer.get((cat, c, L), float("nan")) >= TRANSFER_MIN_AUROC)

    def catlayer_pass(self, cat: str, L: int) -> bool:
        cs = self.concepts[cat]
        return sum(self.transfer_pass(cat, c, L) for c in cs) >= len(cs) / 2

    def eligible(self, cat: str, c: str, L: int) -> bool:
        return self.catlayer_pass(cat, L) and self.concept_pass(cat, c, L)

    def drop_reason(self, cat: str, c: str, L: int) -> str | None:
        if not self.catlayer_pass(cat, L):
            return "category_layer_transfer"
        if not self.validity.get((cat, c, L), float("nan")) >= self.min_validity:
            return "validity"
        if not self.transfer_pass(cat, c, L):
            return "transfer"
        return None

    def pass_share(self, cat: str, layers: Iterable[int] | None = None) -> float:
        """Share of the category's (concept, layer) probes passing both gates (eligible), over
        ``layers`` (default: every layer with a validity entry); NaN if there are none."""
        Ls = sorted(set(layers)) if layers is not None else None
        if Ls is None:
            Ls = sorted({k[2] for k in self.validity if k[0] == cat})
        cs = self.concepts.get(cat, [])
        n = len(cs) * len(Ls)
        return sum(self.eligible(cat, c, L) for c in cs for L in Ls) / n if n else float("nan")

    def table(self) -> list[dict[str, Any]]:
        rows = []
        for cat, cs in self.concepts.items():
            layers = sorted({k[2] for k in self.validity if k[0] == cat})
            for L in layers:
                for c in cs:
                    rows.append(
                        {
                            "category": cat,
                            "concept": c,
                            "layer": L,
                            "validity_auroc": _r(self.validity.get((cat, c, L))),
                            "transfer_auroc": _r(self.transfer.get((cat, c, L))),
                            "pass": self.eligible(cat, c, L),
                            "category_layer_pass": self.catlayer_pass(cat, L),
                        }
                    )
        return rows


def _r(x: Any, nd: int = 4) -> float | None:
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if np.isfinite(x) else None


# ------------------------------------------------------------------ pair tables


@dataclass
class Pairs:
    """Column arrays of one arm's sibling (cell, concept) pairs."""

    item: np.ndarray  # item index (into ``items``)
    items: list[str]
    cat: np.ndarray  # str
    layer: np.ndarray
    pos: np.ndarray
    offset: np.ndarray
    cell: np.ndarray  # cell index
    gold: np.ndarray  # str (the item's gold)
    concept: np.ndarray  # str
    cls: np.ndarray  # str
    leaf: np.ndarray  # str ('' when none)
    scope: np.ndarray  # str ('' when none)
    in_response: np.ndarray
    token_eligible: np.ndarray
    eligible: np.ndarray
    drop: np.ndarray  # str ('' when eligible)
    vals: dict[str, np.ndarray]  # SCORE_FIELDS -> float arrays

    def __len__(self) -> int:
        return int(self.item.size)

    def subset(self, m: np.ndarray) -> "Pairs":
        return Pairs(
            item=self.item[m],
            items=self.items,
            cat=self.cat[m],
            layer=self.layer[m],
            pos=self.pos[m],
            offset=self.offset[m],
            cell=self.cell[m],
            gold=self.gold[m],
            concept=self.concept[m],
            cls=self.cls[m],
            leaf=self.leaf[m],
            scope=self.scope[m],
            in_response=self.in_response[m],
            token_eligible=self.token_eligible[m],
            eligible=self.eligible[m],
            drop=self.drop[m],
            vals={k: v[m] for k, v in self.vals.items()},
        )


@dataclass
class Golds:
    """One row per cell: the gold probe at its own item's cell (cells of items whose gold is
    named in the prompt are left out)."""

    item: np.ndarray
    cat: np.ndarray
    layer: np.ndarray
    mentioned: np.ndarray
    eligible: np.ndarray
    vals: dict[str, np.ndarray]


def build_pairs(
    cells: Sequence[Mapping[str, Any]],
    concepts: Mapping[str, Sequence[str]],
    token_eligible: Mapping[tuple[str, str], bool],
    scores: Mapping[tuple[str, int, int, str], tuple],
    gates: Gates,
    item_order: Sequence[str] | None = None,
) -> tuple[Pairs, Golds]:
    items = list(item_order) if item_order is not None else sorted({r["id"] for r in cells})
    iidx = {k: i for i, k in enumerate(items)}
    cols: dict[str, list] = defaultdict(list)
    gcols: dict[str, list] = defaultdict(list)
    nan = (float("nan"),) * len(SCORE_FIELDS)
    for ci, r in enumerate(cells):
        cat, L, pos = r["category"], int(r["layer"]), int(r["pos"])
        it = iidx[r["id"]]
        resp = set(r.get("response_named") or ())
        for c in concepts[cat]:
            v = scores.get((r["id"], L, pos, c), nan)
            if c == r["gold"]:
                if c in set(r.get("named") or ()):
                    continue  # gold named in the prompt: no gold-mention / ceiling row
                gcols["item"].append(it)
                gcols["cat"].append(cat)
                gcols["layer"].append(L)
                gcols["mentioned"].append(gold_mentioned(r))
                gcols["eligible"].append(gates.eligible(cat, c, L))
                for k, x in zip(SCORE_FIELDS, v, strict=True):
                    gcols[k].append(x)
                continue
            te = bool(token_eligible.get((cat, c), False))
            cls = classify_pair(r, c, token_eligible=te)
            if cls in ("named", "gold"):
                continue
            leaf, scope = pair_leaf_scope(r, c) if cls == "M" else (None, None)
            drop = gates.drop_reason(cat, c, L)
            cols["item"].append(it)
            cols["cat"].append(cat)
            cols["layer"].append(L)
            cols["pos"].append(pos)
            cols["offset"].append(int(r["offset"]))
            cols["cell"].append(ci)
            cols["gold"].append(r["gold"])
            cols["concept"].append(c)
            cols["cls"].append(cls)
            cols["leaf"].append(leaf or "")
            cols["scope"].append(scope or "")
            cols["in_response"].append(c in resp)
            cols["token_eligible"].append(te)
            cols["eligible"].append(drop is None)
            cols["drop"].append(drop or "")
            for k, x in zip(SCORE_FIELDS, v, strict=True):
                cols[k].append(x)

    def arr(d: dict[str, list], k: str, dt: Any = None) -> np.ndarray:
        return np.array(d.get(k, []), dtype=dt)

    pairs = Pairs(
        item=arr(cols, "item", np.int64),
        items=items,
        cat=arr(cols, "cat", object),
        layer=arr(cols, "layer", np.int64),
        pos=arr(cols, "pos", np.int64),
        offset=arr(cols, "offset", np.int64),
        cell=arr(cols, "cell", np.int64),
        gold=arr(cols, "gold", object),
        concept=arr(cols, "concept", object),
        cls=arr(cols, "cls", object),
        leaf=arr(cols, "leaf", object),
        scope=arr(cols, "scope", object),
        in_response=arr(cols, "in_response", bool),
        token_eligible=arr(cols, "token_eligible", bool),
        eligible=arr(cols, "eligible", bool),
        drop=arr(cols, "drop", object),
        vals={k: arr(cols, k, np.float64) for k in SCORE_FIELDS},
    )
    golds = Golds(
        item=arr(gcols, "item", np.int64),
        cat=arr(gcols, "cat", object),
        layer=arr(gcols, "layer", np.int64),
        mentioned=arr(gcols, "mentioned", bool),
        eligible=arr(gcols, "eligible", bool),
        vals={k: arr(gcols, k, np.float64) for k in SCORE_FIELDS},
    )
    return pairs, golds


# ------------------------------------------------------------------ bootstrap machinery


def boot_weights(n_items: int, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> np.ndarray:
    """[n_boot, n_items] item multiplicities of an item-clustered bootstrap (seeded)."""
    if n_items == 0:
        return np.zeros((n_boot, 0))
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_items, np.full(n_items, 1.0 / n_items), size=n_boot).astype(np.float64)


def _ci(est: float, boots: np.ndarray, n: int, **extra: Any) -> dict[str, Any]:
    b = boots[np.isfinite(boots)]
    lo, hi = (
        (float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975)))
        if b.size
        else (
            float("nan"),
            float("nan"),
        )
    )
    return {"est": _r(est), "lo": _r(lo), "hi": _r(hi), "n": int(n), **extra}


def item_sums(item: np.ndarray, x: np.ndarray, n_items: int) -> np.ndarray:
    return np.bincount(item, weights=x, minlength=n_items)


def ratio_boot(
    item: np.ndarray, num: np.ndarray, den: np.ndarray, W: np.ndarray
) -> tuple[float, np.ndarray]:
    """Point estimate and bootstrap draws of sum(num) / sum(den) (rows grouped by item)."""
    n_items = W.shape[1]
    ni, di = item_sums(item, num, n_items), item_sums(item, den, n_items)
    with np.errstate(all="ignore"):
        est = ni.sum() / di.sum() if di.sum() else float("nan")
        return float(est), (W @ ni) / (W @ di)


def mean_ci(item: np.ndarray, x: np.ndarray, W: np.ndarray) -> dict[str, Any]:
    ok = np.isfinite(x)
    est, b = ratio_boot(item[ok], x[ok], np.ones(int(ok.sum())), W)
    return _ci(est, b, int(ok.sum()), n_items=int(np.unique(item[ok]).size))


# ------------------------------------------------------------------ fixed-effect regressions


def codes(groups: np.ndarray) -> np.ndarray:
    """Integer group codes (``np.unique`` inverse)."""
    return np.unique(groups, return_inverse=True)[1].ravel()


def demean(x: np.ndarray, groups: np.ndarray, *, coded: bool = False) -> np.ndarray:
    """``x`` minus its group mean (columns of a 2-D ``x`` separately)."""
    g = groups if coded else codes(groups)
    cnt = np.bincount(g).astype(np.float64)
    if x.ndim == 1:
        return x - (np.bincount(g, weights=x) / cnt)[g]
    out = np.empty_like(x, dtype=np.float64)
    for j in range(x.shape[1]):
        out[:, j] = x[:, j] - (np.bincount(g, weights=x[:, j]) / cnt)[g]
    return out


def demean2(
    x: np.ndarray, g1: np.ndarray, g2: np.ndarray, *, tol: float = 1e-10, max_iter: int = 1000
) -> np.ndarray:
    """Residual of ``x`` on two sets of fixed effects (alternating projections)."""
    c1, c2 = codes(g1), codes(g2)
    r = np.asarray(x, dtype=np.float64).copy()
    for _ in range(max_iter):
        prev = r
        r = demean(demean(r, c1, coded=True), c2, coded=True)
        if np.max(np.abs(r - prev), initial=0.0) < tol:
            break
    return r


def fe_ols_boot(
    y: np.ndarray, X: np.ndarray, item: np.ndarray, W: np.ndarray, coef: int = 0
) -> tuple[float, np.ndarray]:
    """Coefficient ``coef`` of OLS ỹ ~ X̃ (already residualised on item-nested fixed effects),
    and its item-bootstrap draws via per-item cross-products."""
    n_items, k = W.shape[1], X.shape[1]
    A = np.zeros((n_items, k, k))
    b = np.zeros((n_items, k))
    np.add.at(A, item, X[:, :, None] * X[:, None, :])
    np.add.at(b, item, X * y[:, None])

    def solve(Am: np.ndarray, bm: np.ndarray) -> float:
        sol, *_ = np.linalg.lstsq(Am, bm, rcond=None)
        return float(sol[coef])

    if not y.size or not np.any(X[:, coef] != 0):
        return float("nan"), np.full(W.shape[0], np.nan)
    est = solve(A.sum(0), b.sum(0))
    AW = np.einsum("bi,ijk->bjk", W, A)
    bW = W @ b
    boots = np.array([solve(AW[j], bW[j]) for j in range(W.shape[0])])
    return est, boots


def beta1(
    z: np.ndarray,
    mentioned: np.ndarray,
    offset: np.ndarray,
    group: np.ndarray,
    item: np.ndarray,
    W: np.ndarray,
) -> tuple[float, np.ndarray]:
    """The ``mentioned`` coefficient of z ~ mentioned + offset dummies + group FE (within
    transform; groups nested in items)."""
    offs = np.unique(offset)
    if offs.size > 1:
        ref = Counter(offset.tolist()).most_common(1)[0][0]
        D = np.stack([(offset == o).astype(np.float64) for o in offs if o != ref], axis=1)
        X = np.concatenate([mentioned.astype(np.float64)[:, None], D], axis=1)
    else:
        X = mentioned.astype(np.float64)[:, None]
    return fe_ols_boot(demean(z, group), demean(X, group), item, W)


def test3(
    det: np.ndarray,
    mentioned: np.ndarray,
    cell: np.ndarray,
    itemc: np.ndarray,
    item: np.ndarray,
    W: np.ndarray,
) -> tuple[float, np.ndarray]:
    """LPM ``mentioned`` coefficient with cell FE and (item, c) FE."""
    yr = demean2(det, cell, itemc)
    xr = demean2(mentioned.astype(np.float64), cell, itemc)
    return fe_ols_boot(yr, xr[:, None], item, W)


# ------------------------------------------------------------------ the tests on one arm


def _key(*cols: np.ndarray) -> np.ndarray:
    return np.array(["\x1f".join(map(str, t)) for t in zip(*cols, strict=True)], dtype=object)


def matched_sample(p: Pairs) -> np.ndarray:
    """Mask of M + U_matched among eligible pairs with finite z: the (item, layer, c) groups
    holding at least one M pair, their M pairs and their unmentioned pairs."""
    base = p.eligible & np.isin(p.cls, ["M", "unmentioned"])
    grp = _key(p.item, p.layer, p.concept)
    m_groups = set(grp[base & (p.cls == "M")].tolist())
    return base & np.array([g in m_groups for g in grp], dtype=bool)


def run_beta1(p: Pairs, W: np.ndarray) -> dict[str, Any]:
    m = matched_sample(p) & np.isfinite(p.vals["z"])
    q = p.subset(m)
    est, b = beta1(q.vals["z"], q.cls == "M", q.offset, _key(q.item, q.layer, q.concept), q.item, W)
    return _ci(
        est,
        b,
        len(q),
        n_m=int((q.cls == "M").sum()),
        n_groups=int(np.unique(_key(q.item, q.layer, q.concept)).size),
        n_items=int(np.unique(q.item).size),
    )


def test2_cells(p: Pairs) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """Per M cell: (item, mean z mentioned, mean z unmentioned, median z unmentioned, cats)."""
    ok = p.eligible & np.isfinite(p.vals["z"])
    by: dict[int, tuple[list[float], list[float], int, str]] = {}
    for i in np.flatnonzero(ok & np.isin(p.cls, ["M", "unmentioned"])):
        c = int(p.cell[i])
        if c not in by:
            by[c] = ([], [], int(p.item[i]), str(p.cat[i]))
        by[c][0 if p.cls[i] == "M" else 1].append(float(p.vals["z"][i]))
    rows = [
        (v[2], np.mean(v[0]), np.mean(v[1]), np.median(v[1]), v[3])
        for v in by.values()
        if v[0] and v[1]
    ]
    if not rows:
        return np.zeros(0, np.int64), np.zeros(0), np.zeros(0), np.zeros(0), []
    it, zm, zu, zmed, cats = zip(*rows, strict=True)
    return np.array(it), np.array(zm), np.array(zu), np.array(zmed), list(cats)


def run_beta2(p: Pairs, W: np.ndarray) -> dict[str, Any]:
    it, zm, zu, zmed, _ = test2_cells(p)
    est, b = ratio_boot(it, zm - zu, np.ones(it.size), W)
    out = _ci(est, b, int(it.size), n_items=int(np.unique(it).size))
    out["share_outrank_median"] = _r(float(np.mean(zm > zmed))) if it.size else None
    return out


def run_within_auroc(p: Pairs, W: np.ndarray) -> dict[str, Any]:
    ok = p.eligible & np.isfinite(p.vals["z"]) & np.isin(p.cls, ["M", "unmentioned"])
    grp = _key(p.item, p.layer, p.concept)
    by: dict[str, tuple[list[float], list[float], int]] = {}
    for i in np.flatnonzero(ok):
        g = grp[i]
        if g not in by:
            by[g] = ([], [], int(p.item[i]))
        by[g][0 if p.cls[i] == "M" else 1].append(float(p.vals["z"][i]))
    vals = [(v[2], auroc(np.array(v[0]), np.array(v[1]))) for v in by.values() if v[0] and v[1]]
    if not vals:
        return _ci(float("nan"), np.full(W.shape[0], np.nan), 0)
    it = np.array([v[0] for v in vals])
    a = np.array([v[1] for v in vals])
    est, b = ratio_boot(it, a, np.ones(a.size), W)
    return _ci(est, b, int(a.size), n_items=int(np.unique(it).size))


def run_test3(p: Pairs, W: np.ndarray, field: str) -> dict[str, Any]:
    ok = p.eligible & np.isin(p.cls, ["M", "unmentioned"]) & np.isfinite(p.vals[field])
    q = p.subset(ok)
    if not len(q):
        return _ci(float("nan"), np.full(W.shape[0], np.nan), 0)
    est, b = test3(q.vals[field], q.cls == "M", q.cell, _key(q.item, q.concept), q.item, W)
    return _ci(est, b, len(q), n_m=int((q.cls == "M").sum()))


# ------------------------------------------------------------------ detection / hallucination


def _det_ci(p: Pairs, mask: np.ndarray, field: str, W: np.ndarray) -> dict[str, Any]:
    return mean_ci(p.item[mask], p.vals[field][mask], W)


def ceiling_ci(
    g: Golds,
    field: str,
    W: np.ndarray,
    *,
    mentioned_only: bool = True,
    cats: Sequence[str] | None = None,
) -> dict[str, Any]:
    m = g.eligible.copy()
    if mentioned_only:
        m &= g.mentioned
    if cats is not None:
        m &= np.isin(g.cat, list(cats))
    return mean_ci(g.item[m], g.vals[field][m], W)


def _rate(ci: dict[str, Any]) -> dict[str, Any]:
    """1 - detection (CI flipped)."""
    out = dict(ci)
    for k_from, k_to in (("est", "est"), ("hi", "lo"), ("lo", "hi")):
        v = ci.get(k_from)
        out[k_to] = _r(1 - v) if v is not None else None
    return out


def halluc_rows(
    p: Pairs,
    g: Golds,
    W: np.ndarray,
    mask_m: np.ndarray,
    u_mask: np.ndarray,
    th: str,
    adjudicable: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Probe-verified hallucination rate under threshold ``th`` over ``mask_m`` pairs: overall,
    per category, macro over categories, with ceiling and floor. With ``adjudicable`` (category
    -> flag), each category row carries the flag and ``macro_adjudicable`` is the macro over the
    flagged categories only."""
    field = THRESHOLDS[th]
    cfield = CEILING_FIELD[th]
    cats = sorted(set(p.cat[mask_m].tolist()))
    out: dict[str, Any] = {
        "rate": _rate(_det_ci(p, mask_m, field, W)),
        "ceiling": ceiling_ci(g, cfield, W),
        "floor": _det_ci(p, u_mask, field, W),
        "by_category": {},
    }
    per_cat_boot, per_cat_est = [], []
    for c in cats:
        m = mask_m & (p.cat == c)
        ok = m & np.isfinite(p.vals[field])
        est, b = ratio_boot(p.item[ok], 1 - p.vals[field][ok], np.ones(int(ok.sum())), W)
        per_cat_est.append(est)
        per_cat_boot.append(b)
        out["by_category"][c] = {
            "adjudicable": None if adjudicable is None else bool(adjudicable.get(c, False)),
            "rate": _ci(est, b, int(ok.sum())),
            "ceiling": ceiling_ci(g, cfield, W, cats=[c]),
            "floor": _det_ci(p, u_mask & (p.cat == c), field, W),
        }
    if per_cat_est:
        with np.errstate(all="ignore"):
            mb = np.nanmean(np.stack(per_cat_boot), axis=0)
        out["macro"] = _ci(float(np.nanmean(per_cat_est)), mb, len(cats))
        if adjudicable is not None:
            keep = [i for i, c in enumerate(cats) if adjudicable.get(c, False)]
            if keep:
                with np.errstate(all="ignore"):
                    mba = np.nanmean(np.stack([per_cat_boot[i] for i in keep]), axis=0)
                est_a = float(np.nanmean([per_cat_est[i] for i in keep]))
            else:
                est_a, mba = float("nan"), np.full(W.shape[0], np.nan)
            out["macro_adjudicable"] = _ci(est_a, mba, len(keep))
    return out


def u_matched_mask(p: Pairs) -> np.ndarray:
    return matched_sample(p) & (p.cls == "unmentioned")


ADJUDICABLE_MIN_SHARE = 0.5  # review fix: below this (concept, layer) pass share -> too weak
N_PAIR_TABLE = 25


def category_gates(gates: Gates, p: Pairs) -> dict[str, dict[str, Any]]:
    """Per category: the (concept, layer) pass share over the layers of the arm's cells, and
    ``adjudicable`` = share >= 50% (else the probes are too weak to adjudicate)."""
    out = {}
    for cat in sorted(gates.concepts):
        Ls = sorted(set(p.layer[p.cat == cat].tolist()))
        share = gates.pass_share(cat, Ls) if Ls else float("nan")
        out[cat] = {
            "layers": Ls,
            "pass_share": _r(share),
            "adjudicable": bool(share >= ADJUDICABLE_MIN_SHARE),
        }
    return out


def pair_table(
    p: Pairs, m: np.ndarray, gates: Gates | None = None, top: int = N_PAIR_TABLE
) -> list[dict[str, Any]]:
    """The ``top`` (category, gold -> sibling) pairs by number of eligible M pairs ``m``: n,
    plain means over those pairs of the fold-averaged detection (tau_cat, tau(c|g)) and of
    z(c|g); ``sibling_gates_all_layers`` = the sibling's probe passes both gates at every layer
    where this (gold, sibling) has an M pair (eligible or not; None without ``gates``).
    ``u_*`` columns: the same means over the pair's matched unmentioned pairs (U_matched: same
    item, layer and concept, cells that do not mention it), the per-pair baseline."""
    m_all = p.cls == "M"
    u_all = u_matched_mask(p)
    keys = Counter(zip(p.cat[m].tolist(), p.gold[m].tolist(), p.concept[m].tolist(), strict=True))
    rows = []
    for (cat, gold, c), n in sorted(keys.items(), key=lambda kv: (-kv[1], kv[0]))[:top]:
        sel = m & (p.cat == cat) & (p.gold == gold) & (p.concept == c)
        sel_all = m_all & (p.cat == cat) & (p.gold == gold) & (p.concept == c)
        sel_u = u_all & (p.cat == cat) & (p.gold == gold) & (p.concept == c)
        layers_all = sorted(set(p.layer[sel_all].tolist()))

        def mean(f: str, sel: np.ndarray = sel) -> float | None:
            x = p.vals[f][sel]
            x = x[np.isfinite(x)]
            return _r(float(x.mean())) if x.size else None

        rows.append(
            {
                "category": cat,
                "gold": gold,
                "sibling": c,
                "n": int(n),
                "n_M_all": int(sel_all.sum()),
                "n_items": int(np.unique(p.item[sel]).size),
                "layers": sorted(set(p.layer[sel].tolist())),
                "det_cat": mean("det_cat"),
                "det_cg": mean("det_cg"),
                "z_mean": mean("z"),
                "n_u": int(sel_u.sum()),
                "u_det_cat": mean("det_cat", sel_u),
                "u_det_cg": mean("det_cg", sel_u),
                "u_z_mean": mean("z", sel_u),
                "sibling_gates_all_layers": (
                    None
                    if gates is None
                    else all(gates.concept_pass(cat, c, L) for L in layers_all)
                ),
            }
        )
    return rows


def arm_summary(
    p: Pairs,
    g: Golds,
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
    gates: Gates | None = None,
) -> dict:
    W = boot_weights(len(p.items), n_boot, seed)
    cg = category_gates(gates, p) if gates is not None else None
    adj = {k: v["adjudicable"] for k, v in cg.items()} if cg is not None else None
    sib_mention = np.isin(p.cls, list(MENTION_CLASSES))
    m_all = p.cls == "M"
    m = m_all & p.eligible
    u = u_matched_mask(p)
    out: dict[str, Any] = {
        "n_pairs": len(p),
        "class_counts": dict(Counter(p.cls.tolist())),
        "class_counts_eligible": dict(Counter(p.cls[p.eligible].tolist())),
        "coverage": {
            "n_M": int(m_all.sum()),
            "n_M_eligible": int(m.sum()),
            "dropped": dict(Counter(p.drop[m_all & ~p.eligible].tolist())),
            "share_dropped": _r(1 - m.sum() / m_all.sum()) if m_all.sum() else None,
        },
        "response_named": {
            "n_sibling_mentions": int((sib_mention & p.in_response).sum()),
            "by_class": dict(Counter(p.cls[sib_mention & p.in_response].tolist())),
        },
        "beta1": run_beta1(p, W),
        "beta2": run_beta2(p, W),
        "within_item_auroc": run_within_auroc(p, W),
        "test3": {
            "tau_cg": run_test3(p, W, "det_cg"),
            "tau_cat": run_test3(p, W, "det_cat"),
        },
        "detection": {},
        "halluc": {},
        "gold_detection": {
            "all_cells": ceiling_ci(g, "det_cat", W, mentioned_only=False),
            "gold_mention_cells": ceiling_ci(g, "det_cat", W),
            "n_gold_mention_cells": int((g.mentioned & g.eligible).sum()),
        },
    }
    for th, field in THRESHOLDS.items():
        det: dict[str, Any] = {
            "M": _det_ci(p, m, field, W),
            "U_matched": _det_ci(p, u, field, W),
            "ceiling": ceiling_ci(g, CEILING_FIELD[th], W),
            "ceiling_all_cells": ceiling_ci(g, CEILING_FIELD[th], W, mentioned_only=False),
            "by_layer": {},
            "by_category": {},
        }
        for L in sorted(set(p.layer[m].tolist())):
            det["by_layer"][str(L)] = {
                "M": _det_ci(p, m & (p.layer == L), field, W),
                "U_matched": _det_ci(p, u & (p.layer == L), field, W),
            }
        for c in sorted(set(p.cat[m].tolist())):
            det["by_category"][c] = {
                "M": _det_ci(p, m & (p.cat == c), field, W),
                "U_matched": _det_ci(p, u & (p.cat == c), field, W),
                "gold": ceiling_ci(g, CEILING_FIELD[th], W, cats=[c]),
            }
        out["detection"][th] = det
        out["halluc"][th] = {
            "all": halluc_rows(p, g, W, m, u, th, adj),
            "scope_user": halluc_rows(p, g, W, m & (p.scope == "user"), u, th),
            "by_leaf": {
                lf: _rate(_det_ci(p, m & (p.leaf == lf), field, W))
                for lf in sorted(set(p.leaf[m].tolist()))
            },
        }
    # all sibling mentions, judge verdict ignored (tau_cat)
    out["halluc_all_mentions"] = halluc_rows(p, g, W, sib_mention & p.eligible, u, "tau_cat")
    out["strong"] = strong_criterion(p, g, W, m, u)
    out["category_gates"] = cg
    out["pair_table"] = pair_table(p, m, gates)
    _it, zm, zu, _, cats = test2_cells(p)
    out["test2_points"] = [
        [_r(a, 3), _r(b, 3), c] for a, b, c in zip(zm.tolist(), zu.tolist(), cats, strict=True)
    ]
    return out


def strong_criterion(
    p: Pairs, g: Golds, W: np.ndarray, m: np.ndarray, u: np.ndarray
) -> dict[str, Any]:
    """(det(M) - det(U)) / (ceiling - det(U)) under tau_cat, with CIs; ``undefined`` if the
    denominator's CI includes 0."""
    f = "det_cat"
    okm, oku = m & np.isfinite(p.vals[f]), u & np.isfinite(p.vals[f])
    gm = g.eligible & g.mentioned & np.isfinite(g.vals[f])
    if not okm.any() or not oku.any() or not gm.any():
        return {"status": "undefined", "reason": "empty side"}
    dm, bm = ratio_boot(p.item[okm], p.vals[f][okm], np.ones(int(okm.sum())), W)
    du, bu = ratio_boot(p.item[oku], p.vals[f][oku], np.ones(int(oku.sum())), W)
    dc, bc = ratio_boot(g.item[gm], g.vals[f][gm], np.ones(int(gm.sum())), W)
    with np.errstate(all="ignore"):
        ratio, br = (dm - du) / (dc - du), (bm - bu) / (bc - bu)
    den = _ci(dc - du, bc - bu, int(gm.sum()))
    rat = _ci(ratio, br, int(okm.sum()))
    if den["lo"] is None or den["lo"] <= 0 <= den["hi"]:
        status = "undefined"
    elif rat["lo"] is not None and rat["est"] > 0.5 and rat["lo"] > 0.5:
        status = "met"
    else:
        status = "not met"
    return {"status": status, "ratio": rat, "denominator": den}


# ------------------------------------------------------------------ the verdict


def _sig(ci: Mapping[str, Any] | None) -> str:
    """``pos`` / ``neg`` (95% CI excludes 0 on that side), ``null``, or ``na``."""
    if not ci or ci.get("lo") is None or ci.get("hi") is None:
        return "na"
    if ci["lo"] > 0:
        return "pos"
    if ci["hi"] < 0:
        return "neg"
    return "null"


def control_passes(control: Mapping[str, Any] | None) -> bool:
    return bool(
        control and _sig(control.get("beta1")) == "pos" and _sig(control.get("beta2")) == "pos"
    )


def verdict(b1: Mapping[str, Any] | None, b2: Mapping[str, Any] | None, control_ok: bool) -> str:
    """The pre-registered decision table (plan rev 9)."""
    s1, s2 = _sig(b1), _sig(b2)
    if s1 == "pos" and s2 == "pos":
        return "supported"
    if not control_ok:
        return "inconclusive (control failed)"
    if "neg" in (s1, s2):
        return "contradicted"
    if s1 == "null" and s2 == "null":
        return "not supported"
    if {s1, s2} == {"pos", "null"}:
        return "inconclusive (mixed)"
    return "inconclusive (missing estimate)"


def verdict_table(arms: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rows for s3d (primary) and NLA (secondary): all transfer-passing concepts and the
    token-eligible subset (``<arm>@token_eligible``); the control is the J-lens arm (which only
    ever contains token-eligible concepts)."""
    ctrl = arms.get(CONTROL)
    ok = control_passes(ctrl)
    rows = []
    for arm, role in ((PRIMARY, "primary"), (SECONDARY, "secondary")):
        for key, subset in (
            (arm, "all transfer-passing"),
            (f"{arm}@token_eligible", "token-eligible"),
            (f"{arm}@validity085", "validity >= 0.85 (robustness)"),
        ):
            s = arms.get(key)
            if s is None:
                continue
            v = verdict(s.get("beta1"), s.get("beta2"), ok)
            strong = (s.get("strong") or {}).get("status")
            rows.append(
                {
                    "arm": arm,
                    "role": role,
                    "subset": subset,
                    "beta1": s.get("beta1"),
                    "beta2": s.get("beta2"),
                    "control_passes": ok,
                    "verdict": v,
                    "strongly_supported": (
                        "undefined"
                        if strong == "undefined"
                        else (v == "supported" and strong == "met")
                    ),
                }
            )
    if ctrl is not None:
        rows.append(
            {
                "arm": CONTROL,
                "role": "positive control",
                "subset": "token-eligible",
                "beta1": ctrl.get("beta1"),
                "beta2": ctrl.get("beta2"),
                "control_passes": ok,
                "verdict": "control passes" if ok else "control fails",
            }
        )
    return rows


# ------------------------------------------------------------------ driver


def score_all(
    arm_cells: Mapping[str, Sequence[Mapping[str, Any]]],
    scores: Mapping[tuple[str, int, int, str], tuple],
    validity: Mapping[tuple[str, str, int], float],
    items: Mapping[str, Mapping[str, Any]],
    concepts: Mapping[str, Sequence[str]],
    token_eligible: Mapping[tuple[str, str], bool],
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    gates = Gates(
        dict(validity),
        transfer_auroc(scores, items, concepts),
        {k: list(v) for k, v in concepts.items()},
    )
    out: dict[str, Any] = {"gates": gates.table(), "arms": {}, "n_boot": n_boot, "seed": seed}
    item_order = sorted(items)
    for arm, cells in arm_cells.items():
        p, g = build_pairs(cells, concepts, token_eligible, scores, gates, item_order)
        log(f"[score] {arm}: {len(p)} sibling pairs, {int((p.cls == 'M').sum())} M")
        out["arms"][arm] = arm_summary(p, g, n_boot=n_boot, seed=seed, gates=gates)
        if arm in (PRIMARY, SECONDARY):
            te = p.token_eligible
            out["arms"][f"{arm}@token_eligible"] = arm_summary(
                p.subset(te), g, n_boot=n_boot, seed=seed, gates=gates
            )
            # plan rev 10 robustness row: probes passing the original 0.85 validity gate
            strict = replace(gates, min_validity=VALIDITY_STRICT_AUROC)
            ps, gs = build_pairs(cells, concepts, token_eligible, scores, strict, item_order)
            out["arms"][f"{arm}@validity085"] = arm_summary(
                ps, gs, n_boot=n_boot, seed=seed, gates=strict
            )
    out["verdict"] = verdict_table(out["arms"])
    return out


def write_summary(summary: Mapping[str, Any], out_dir: Path = SCORE_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=1, default=str) + "\n")
    return path


def load_and_score(
    *,
    arms: Sequence[str] = (PRIMARY, SECONDARY, CONTROL),
    judge_root: Path | None = None,
    probes_dir: Path | None = None,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict[str, Any]:
    """Load the judge cells of every arm present, the probe job's eval scores and validity
    table, the bank and the inventory; score; record provenance."""
    from hallucination_bench.evals.sibling_probe import bank as B
    from hallucination_bench.evals.sibling_probe import judge as J
    from hallucination_bench.evals.sibling_probe import probe_io as PIO
    from hallucination_bench.evals.sibling_probe.concepts import load_inventory

    judge_root = Path(judge_root or J.JUDGE_ROOT)
    probes_dir = Path(probes_dir or PIO.PROBES_DIR)
    inv = load_inventory()
    items = {it["id"]: it for it in B.load_items()}
    concepts = {name: cat.ids for name, cat in inv.items()}
    te = {(name, c.id): c.token_eligible for name, cat in inv.items() for c in cat.concepts}
    validity = {
        k: float(v["auroc"]) for k, v in PIO.load_validity(probes_dir / "validity.jsonl").items()
    }
    scores = index_scores(PIO.iter_eval_scores(probes_dir / "eval_scores.jsonl.gz"))
    arm_cells, runs = {}, {}
    for a in arms:
        f = judge_root / a / "cells.jsonl"
        if not f.exists():
            print(f"[score] {a}: no {f} (skipped)")
            continue
        arm_cells[a] = J.load_cells(a, judge_root)
        rj = judge_root / a / "run.json"
        runs[a] = json.loads(rj.read_text()) if rj.exists() else {}
    out = score_all(arm_cells, scores, validity, items, concepts, te, n_boot=n_boot, seed=seed)
    psum = probes_dir / "summary.json"
    out["provenance"] = {
        "judge_runs": {
            a: {
                k: r.get(k)
                for k in (
                    "claimmap_prompt_version",
                    "taxonomy_version",
                    "summarizer_prompt_version",
                    "scope_prompt_version",
                    "cover_prompt_version",
                    "git_sha",
                    "model",
                    "effort",
                )
            }
            for a, r in runs.items()
        },
        "probes": json.loads(psum.read_text()) if psum.exists() else None,
        "git_sha": PIO.git_sha(),
        "inventory_only": True,
    }
    out["coverage_inventory"] = inventory_coverage(arm_cells)
    return out


def inventory_coverage(arm_cells: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Per arm x category: probe-candidate segments whose ``probe_target`` names an inventory
    sibling vs targets outside the inventory that the ``sibcover-v1`` call calls an instance
    (prefiltered cells, and the 5% sample of the rest scaled up), plus probe_target / alias
    agreement."""
    from hallucination_bench.evals.sibling_probe.judge import COVERAGE_FRACTION

    out: dict[str, Any] = {}
    for arm, cells in arm_cells.items():
        per: dict[str, Counter] = defaultdict(Counter)
        for r in cells:
            cat = r["category"]
            sample = bool(r.get("coverage_sample"))
            for i, s in enumerate(r.get("segments") or ()):
                if s.get("verdict") != tx.V_PROBE:
                    continue
                tc = s.get("target_concepts") or []
                sib_t = [c for c in tc if c != r["gold"] and c not in (r.get("named") or ())]
                if sib_t:
                    per[cat]["in_inventory_sample" if sample else "in_inventory"] += 1
                elif not tc and s.get("probe_target") and s.get("cover") is True:
                    per[cat]["out_of_inventory_sample" if sample else "out_of_inventory"] += 1
                if tc:
                    seg_alias = {
                        m["concept"] for m in r.get("mentions") or () if m.get("segment") == i
                    }
                    per[cat]["target_names_concept"] += 1
                    per[cat]["target_agrees_alias"] += bool(set(tc) & seg_alias)
        rows = {}
        for cat, c in per.items():
            inn = c["in_inventory"] + c["in_inventory_sample"] / COVERAGE_FRACTION
            outn = c["out_of_inventory"] + c["out_of_inventory_sample"] / COVERAGE_FRACTION
            rows[cat] = {
                **dict(c),
                "share_outside_inventory_est": _r(outn / (inn + outn)) if inn + outn else None,
                "probe_target_alias_agreement": _r(
                    c["target_agrees_alias"] / c["target_names_concept"]
                )
                if c["target_names_concept"]
                else None,
            }
        out[arm] = rows
    return out
