"""Plan 0009 §3d-e and §4: distractors, n/a, fire / nearby, cross-method, the matched null and
the labels file.

**Order of decisions (strict):** n/a -> fire -> nearby -> cross_method -> probe_unconfirmed.
An n/a unit is final ``probe_na`` and never reaches cross_method or probe_unconfirmed; a
``no_gold`` unit that does not fire (same layer or nearby) is ``probe_na`` (flag ``no_gold``) and
skips cross_method too, so ``no_gold`` never yields ``probe_unconfirmed``.

S is the probe of the unit's ``said_canonical`` concept (in C, §3a). Fire = at (id, L, pos) z_S
exceeds the z of every valid distractor (type (a), whitened DiffMean). Nearby = it fires at the
same (id, pos) at a neighbouring layer (:func:`neighbours`). L42 arms read their L40 / L44
neighbours from the 11-layer bank; if the step-0 token check of the two banks fails
(``l42_tokens_ok=False``) nearby is disabled for them and units that would fall through to
probe_unconfirmed become probe_na.

**Per-key null check (rev 8 / 8.1).** The null-check state is per (layer, key):
``ClassProbes.null_ok[layer][key]``. A failing key no longer takes down the class: only a unit
whose own eval-site key fails at L is ``probe_na`` (``null_fail``) at L, readability counts only
the item's sites on keys that pass at L (no such site = unreadable), and nearby counts only at a
neighbour layer where the unit's key passes. Units whose site falls in ``other`` are n/a by
design when ``other`` fails. Keys are in the class's merged key space (``ClassProbes.merge``,
raw token class -> key, fixed across layers); a key that cannot be resolved fails (fail-safe).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import units as U
from .classes import MIN_CLASS_SIZE, NO_CLASS
from .texts import OTHER, TOKEN_CLASSES

N_DISTRACTORS = 8
N_NEAREST = 4
MIN_DISTRACTORS = 6
READABLE_SHARE = 0.5
VALIDITY_MIN = 0.90
DISTRACTOR_LAYER = 44
GRID_LAYERS = tuple(range(20, 61, 4))
PROBE_LAYERS = tuple(sorted((*GRID_LAYERS, 42)))  # 12 capture layers (§3b)
CROSS_LAYER_WINDOW = 4
SHARED_PRIOR = 0.5

RUNG2_STATUSES = ("probe_faithful", "probe_nearby", "cross_method", "probe_unconfirmed", "probe_na")


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower() or "x"


def neighbours(layer: int) -> tuple[int, ...]:
    """All-layer arms: the grid layers at distance exactly 4, plus L42 for L40 and L44. L42
    (the L42 arms): L40 and L44."""
    if layer == 42:
        return (40, 44)
    out = [x for x in (layer - 4, layer + 4) if x in GRID_LAYERS]
    if layer in (40, 44):
        out.append(42)
    return tuple(sorted(out))


# ---------------------------------------------------------------- distractors (frozen)


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b, strict=True)) / (na * nb + 1e-12)


# Full-run finding: list-style claims put compound concepts ("students and departments",
# "JavaScript and C++") into C, and near-duplicates ("GPT" for ChatGPT, "students table" for
# students) slipped past the exact-name exclusion, so a distractor could carry the gold itself.
COMPOUND_RE = re.compile(r"\s(?:and|or|&)\s|[,/;]|、|和|与")
_WORD_RE = re.compile(r"\w+")
_STOP = frozenset(
    {"the", "of", "a", "an", "and", "or", "table", "tables", "system", "inc", "co", "ltd"}
)


def _words(s: str) -> set[str]:
    return {w for w in _WORD_RE.findall(s.casefold()) if w not in _STOP}


def is_compound(c: str) -> bool:
    return bool(COMPOUND_RE.search(c))


def overlaps(c: str, names: Iterable[str]) -> bool:
    """True if ``c`` is compound, shares a content word with any of ``names``, or one's
    normalised form contains the other's (both at least 3 characters)."""
    if is_compound(c):
        return True
    cw, cn = _words(c), U.norm(c)
    for n in names:
        if cw & _words(n):
            return True
        nn = U.norm(n)
        if len(cn) >= 3 and len(nn) >= 3 and (cn in nn or nn in cn):
            return True
    return False


def select_distractors(
    unit: str,
    anchor: str,
    concepts: Sequence[str],
    centroids: Mapping[str, Sequence[float]],
    exclude: Iterable[str],
) -> dict[str, Any]:
    """8 distractors from C: the 4 nearest ``anchor`` (the gold; ``said_canonical`` for no_gold
    units) by cosine of the grand-mean-centred implied centroids at L44, plus 4 seeded-random
    (seed = unit key). Never ``exclude`` (gold, said, the gold's compatible stratum,
    ``item_sibling_names[id]``)."""
    ex_names = [*exclude, anchor]
    ex = {U.norm(x) for x in ex_names}
    elig = sorted(c for c in concepts if U.norm(c) not in ex and not overlaps(c, ex_names))
    g = centroids.get(anchor)
    if g is not None:
        ranked = sorted(
            (c for c in elig if c in centroids), key=lambda c: (-_cos(centroids[c], g), c)
        )
    else:
        ranked = []
    nearest = ranked[:N_NEAREST]
    rest = [c for c in elig if c not in nearest]
    rng = random.Random(unit)
    rnd = sorted(rng.sample(rest, min(N_DISTRACTORS - len(nearest), len(rest))))
    return {
        "unit": unit,
        "anchor": anchor,
        "nearest": nearest,
        "random": rnd,
        "distractors": [*nearest, *rnd],
        "n_eligible": len(elig),
    }


def distractors_sha256(rows: Mapping[str, Any]) -> str:
    blob = json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------- probe context


class ZTable:
    """Fold-averaged null z of every concept of one class at (item, layer, pos), type (a)."""

    def __init__(self, concepts: Sequence[str]) -> None:
        self.concepts = list(concepts)
        self.ci = {c: i for i, c in enumerate(self.concepts)}
        self._z: dict[tuple[str, int, int], Sequence[float]] = {}
        self._key: dict[tuple[str, int, int], int] = {}
        self._pos: dict[tuple[str, int], set[int]] = defaultdict(set)

    def add(
        self, item: str, layer: int, pos: int, z: Sequence[float], key: int | None = None
    ) -> None:
        """``key``: the site's merged key as scored (``score/<class>/L<L>.npz`` ``key``)."""
        self._z[(item, int(layer), int(pos))] = z
        if key is not None:
            self._key[(item, int(layer), int(pos))] = int(key)
        self._pos[(item, int(layer))].add(int(pos))

    def key(self, item: str, layer: int, pos: int) -> int | None:
        return self._key.get((item, int(layer), int(pos)))

    def get(self, item: str, layer: int, pos: int, c: str) -> float:
        z = self._z.get((item, int(layer), int(pos)))
        if z is None or c not in self.ci:
            return math.nan
        return float(z[self.ci[c]])

    def positions(self, item: str, layer: int) -> list[int]:
        return sorted(self._pos.get((item, int(layer)), ()))


@dataclass
class ClassProbes:
    """One class: its concepts, validity per (concept, layer), the null check per (layer, key)
    (``null_ok[layer][key]``; a plain bool = the same verdict for every key at that layer), the
    raw-class -> key merge (fixed across layers; empty = identity), the z table and the L44
    implied centroids."""

    name: str
    concepts: list[str]
    validity: dict[tuple[str, int], float] = field(default_factory=dict)
    null_ok: dict[int, dict[int, bool] | bool] = field(default_factory=dict)
    z: ZTable | None = None
    centroids: dict[str, list[float]] = field(default_factory=dict)
    merge: dict[int, int] = field(default_factory=dict)

    def valid(self, c: str, layer: int) -> bool:
        return self.validity.get((c, int(layer)), 0.0) >= VALIDITY_MIN

    def merged(self, raw: int | None) -> int | None:
        if raw is None:
            return None
        return self.merge.get(int(raw), OTHER) if self.merge else int(raw)

    def key_ok(self, layer: int, key: int | None) -> bool:
        """The null check at (layer, key). A missing layer or an unresolved / unchecked key
        fails."""
        v = self.null_ok.get(int(layer))
        if isinstance(v, bool):
            return v
        if not v or key is None:
            return False
        return bool(v.get(int(key), False))

    def site_ok(self, item: str, layer: int, pos: int) -> bool:
        return self.key_ok(layer, self.z.key(item, layer, pos) if self.z is not None else None)


def beats_all(z: ZTable, item: str, layer: int, pos: int, c: str, others: Sequence[str]) -> bool:
    if not others:
        return False
    zc = z.get(item, layer, pos, c)
    if math.isnan(zc):
        return False
    for d in others:
        zd = z.get(item, layer, pos, d)
        if math.isnan(zd) or zc <= zd:
            return False
    return True


@dataclass
class UnitProbe:
    """What rung 2 knows about one unit."""

    unit: str
    arm: str
    item: str
    layer: int
    pos: int
    cls: str
    said: str  # said_canonical (S)
    gold: str | None  # None = no_gold
    distractors: list[str]
    key: int | None = None  # raw token class of the unit's eval-site token (texts.token_class)


def unit_key(cp: ClassProbes, u: UnitProbe, layer: int) -> int | None:
    """The unit's merged key: its site token's class through the class merge, else the key the
    score phase recorded at (item, layer, pos)."""
    if u.key is not None:
        return cp.merged(u.key)
    return cp.z.key(u.item, layer, u.pos) if cp.z is not None else None


def na_reason(cp: ClassProbes | None, u: UnitProbe, layer: int) -> str | None:
    """§3d n/a at ``layer`` (readability only for units with a gold)."""
    if u.cls == NO_CLASS:
        return "no_class"
    if cp is None:  # rev 8.2: a class below the pair cap has no probes (concepts.json omits it)
        return "class_too_small"
    if len(cp.concepts) < MIN_CLASS_SIZE:
        return "small_class"
    if cp.z is None or not cp.z.positions(u.item, layer):
        return "no_scores"
    if u.said not in cp.concepts or not cp.valid(u.said, layer):
        return "invalid_probe"
    dv = [d for d in u.distractors if cp.valid(d, layer)]
    if len(dv) < MIN_DISTRACTORS:
        return "few_distractors"
    if not cp.key_ok(layer, unit_key(cp, u, layer)):
        return "null_fail"  # only this unit's key fails; the class's other keys stay in use
    if u.gold is not None and readable_share(cp, u, layer) < READABLE_SHARE:
        return "unreadable"
    return None


def readable_share(cp: ClassProbes, u: UnitProbe, layer: int) -> float:
    """Over the item's eval positions at ``layer`` except the unit's own pos, on keys that pass
    the null check at ``layer`` (rev 8.1; none = unreadable): the share where the gold beats all
    its valid distractors (an invalid gold probe is never readable)."""
    if u.gold is None or cp.z is None or not cp.valid(u.gold, layer):
        return 0.0
    dv = [d for d in u.distractors if cp.valid(d, layer) and d != u.gold]
    ps = [p for p in cp.z.positions(u.item, layer) if p != u.pos and cp.site_ok(u.item, layer, p)]
    if not ps or len(dv) < MIN_DISTRACTORS:
        return 0.0
    hits = [beats_all(cp.z, u.item, layer, p, u.gold, dv) for p in ps]
    return sum(hits) / len(hits)


def fires(cp: ClassProbes, u: UnitProbe, layer: int) -> bool:
    assert cp.z is not None
    dv = [d for d in u.distractors if cp.valid(d, layer) and d != u.said]
    return beats_all(cp.z, u.item, layer, u.pos, u.said, dv)


# ---------------------------------------------------------------- cross-method


CrossIdx = Mapping[tuple[str, str], Sequence[tuple[str, int, int]]]  # (item, canon) -> sites


def cross_index(
    units: Iterable[Mapping[str, Any]], canonical: Mapping[tuple[str, str], str]
) -> dict[tuple[str, str], list[tuple[str, int, int]]]:
    """(item, said_canonical) -> [(arm, layer, pos)] over every in-scope arm's units."""
    out: dict[tuple[str, str], list[tuple[str, int, int]]] = defaultdict(list)
    for u in units:
        c = canonical.get((u["said_norm"], u["id"]), u["said_norm"])
        out[(u["id"], c)].append((u["arm"], int(u["layer"]), int(u["pos"])))
    return dict(out)


def cross_arms(
    u: UnitProbe,
    idx: CrossIdx,
    *,
    same_pos: bool = False,
    same_family: bool = False,
    discounted: Iterable[str] = (),
) -> list[str]:
    """Arms of a different lens family (``same_family``: any other arm) naming S for the same
    item at a layer within ±4, at any of the item's positions (``same_pos``: the unit's pos)."""
    fam = U.family_of(u.arm)
    skip = set(discounted)
    out = set()
    for arm, la, pos in idx.get((u.item, U.norm(u.said)), ()):
        if arm == u.arm or arm in skip:
            continue
        if not same_family and U.family_of(arm) == fam:
            continue
        if abs(la - u.layer) > CROSS_LAYER_WINDOW or (same_pos and pos != u.pos):
            continue
        out.add(arm)
    return sorted(out)


def shared_prior_arms(
    units: Iterable[Mapping[str, Any]],
    classes: Mapping[str, Mapping[str, Any]],
    canonical: Mapping[tuple[str, str], str],
) -> dict[tuple[str, str], set[str]]:
    """(class, S) -> arms that name S for >= 50% of the items whose gold is in that class (the
    shared-prior discount sensitivity row)."""
    items_of_class: dict[str, set[str]] = defaultdict(set)
    for k, v in classes.items():
        if v["class"] != NO_CLASS and not v["no_gold"]:
            items_of_class[v["class"]].add(k.split("|", 1)[1])
    names: dict[tuple[str, str], set[str]] = defaultdict(set)  # (arm, S) -> items
    cls_of_s: dict[str, set[str]] = defaultdict(set)
    for u in units:
        c = canonical.get((u["said_norm"], u["id"]), u["said_norm"])
        names[(u["arm"], c)].add(u["id"])
        v = classes.get(f"{u['said_norm']}|{u['id']}")
        if v is not None and v["class"] != NO_CLASS:
            cls_of_s[c].add(v["class"])
    out: dict[tuple[str, str], set[str]] = defaultdict(set)
    for (arm, s), its in names.items():
        for k in cls_of_s.get(s, ()):
            den = items_of_class.get(k)
            if den and len(its & den) / len(den) >= SHARED_PRIOR:
                out[(k, s)].add(arm)
    return dict(out)


# ---------------------------------------------------------------- rung 2 (strict precedence)


def rung2(
    u: UnitProbe,
    cp: ClassProbes | None,
    idx: CrossIdx,
    *,
    l42_tokens_ok: bool = True,
    cross_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One unit's rung-2 outcome and its trace."""
    out: dict[str, Any] = {
        "na_reason": None,
        "fire_L": False,
        "fire_nearby": [],
        "cross_arms": [],
        "no_gold": u.gold is None,
    }
    r = na_reason(cp, u, u.layer)
    if r is not None:
        return {**out, "status": "probe_na", "na_reason": r}
    assert cp is not None
    if fires(cp, u, u.layer):
        return {**out, "status": "probe_faithful", "fire_L": True}
    nearby_off = u.layer == 42 and not l42_tokens_ok
    if not nearby_off:
        nb = [L for L in neighbours(u.layer) if na_reason(cp, u, L) is None and fires(cp, u, L)]
        if nb:
            return {**out, "status": "probe_nearby", "fire_nearby": nb}
    if u.gold is None:  # readability cannot be checked: never cross_method / unconfirmed
        return {**out, "status": "probe_na", "na_reason": "no_gold"}
    out["reached_cross"] = True
    cr = cross_arms(u, idx, **(cross_kwargs or {}))
    if cr:
        return {**out, "status": "cross_method", "cross_arms": cr}
    if nearby_off:
        return {**out, "status": "probe_na", "na_reason": "l42_token_mismatch"}
    return {**out, "status": "probe_unconfirmed"}


def matched_null(
    u: UnitProbe,
    cp: ClassProbes | None,
    items_by_gold: Mapping[str, Iterable[str]],
    named: Mapping[tuple[str, str], set[tuple[int, int]]],
) -> dict[str, int]:
    """Reported, not used for labels: S against the same distractors at the item's other sites
    and at other items with the same gold, where no arm names S (``named``: (item, S) -> the
    (layer, pos) sites some arm names S at)."""
    res = {"n_sites": 0, "n_fire": 0}
    if cp is None or cp.z is None or u.said not in cp.concepts or not cp.valid(u.said, u.layer):
        return res
    dv = [d for d in u.distractors if cp.valid(d, u.layer) and d != u.said]
    if len(dv) < MIN_DISTRACTORS:
        return res
    s_key = U.norm(u.said)
    others = [u.item, *(i for i in items_by_gold.get(U.norm(u.gold or ""), ()) if i != u.item)]
    for it in others:
        bad = named.get((it, s_key), set())
        for p in cp.z.positions(it, u.layer):
            if (it == u.item and p == u.pos) or (u.layer, p) in bad:
                continue
            res["n_sites"] += 1
            res["n_fire"] += beats_all(cp.z, it, u.layer, p, u.said, dv)
    return res


# ---------------------------------------------------------------- final + sensitivity + labels


def final_status(r1: Mapping[str, Any] | None, r2: Mapping[str, Any] | None) -> str | None:
    """unstable / resample_failed come from rung 1; stable / sparse / deterministic take the
    rung-2 outcome (``None`` while rung 2 is unfinished)."""
    st = (r1 or {}).get("rung1")
    if st in ("unstable", "resample_failed"):
        return st
    if st in ("stable", "sparse", "deterministic"):
        return (r2 or {}).get("status")
    return None


def sensitivity(
    r1: Mapping[str, Any] | None,
    r2: Mapping[str, Any] | None,
    primary: str | None,
    variants: Mapping[str, Mapping[str, Any] | None],
) -> dict[str, str | None]:
    """Each row re-labels only. Rung-1 rows: unstable under the row's rule, else the unit's
    rung-2 outcome (``probe_na`` when it has none). Cross rows: the rung-2 outcome recomputed
    with the variant cross-method rule (``variants``), for units whose final came from rung 2."""
    out: dict[str, str | None] = {}
    sens = (r1 or {}).get("sens_unstable")
    for k in ("rung1_rev1", "rung1_branch_a_only", "rung1_said_le1", "rung1_said_le3"):
        if sens is None:
            out[k] = primary
        elif sens[k]:
            out[k] = "unstable"
        else:
            out[k] = (r2 or {}).get("status") or "probe_na"
    from_rung2 = (r1 or {}).get("rung1") in ("stable", "sparse", "deterministic")
    for k in ("cross_same_pos", "cross_same_family", "cross_shared_prior"):
        v = variants.get(k)
        out[k] = (v or {}).get("status", primary) if from_rung2 and v is not None else primary
    return out


def label_row(
    unit: Mapping[str, Any],
    r1: Mapping[str, Any] | None,
    r2: Mapping[str, Any] | None,
    cls: Mapping[str, Any] | None,
    variants: Mapping[str, Mapping[str, Any] | None] | None = None,
    key: str | None = None,
) -> dict[str, Any]:
    """``key``: the token-class name of the unit's eval site (the per-key report)."""
    fin = final_status(r1, r2)
    return {
        "unit": unit["unit"],
        "arm": unit["arm"],
        "id": unit["id"],
        "layer": unit["layer"],
        "pos": unit["pos"],
        "said": unit["said"],
        "actual": unit["actual"],
        "segments": [
            {k: s[k] for k in ("key", "readout", "start", "end")} for s in unit["segments"]
        ],
        "key": key,
        "rung1": (r1 or {}).get("rung1"),
        "counts": (r1 or {}).get("counts"),
        "branch": (r1 or {}).get("branch"),
        "final": fin,
        "rung2": r2,
        "flags": {
            "no_gold": bool((cls or {}).get("no_gold")) if cls else False,
            "no_class": bool((cls or {}).get("no_class")) if cls else False,
            "diverse": bool((r1 or {}).get("diverse")),
        },
        "sensitivity": sensitivity(r1, r2, fin, variants or {}),
    }


def write_labels(rows: Iterable[Mapping[str, Any]], labels_dir: Path) -> dict[str, int]:
    by: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        by[r["arm"]].append(r)
    return {a: U.write_jsonl(Path(labels_dir) / f"{a}.jsonl", rs) for a, rs in sorted(by.items())}


# ---------------------------------------------------------------- loading probe outputs


def load_class_probes(
    probes_dir: Path, concepts: Mapping[str, Mapping[str, Any]]
) -> dict[str, ClassProbes]:
    """``<probes_dir>/fit/<class slug>/fit_L<L>.json`` (``validity`` per concept, the per-key
    ``null_check.keys`` -> ``null_ok[L][key]``, ``merge``),
    ``centroids_L44.json`` and ``<probes_dir>/score/<class slug>/L<L>.npz`` (``item``, ``pos``,
    ``za`` [P, C], ``key`` [P] the merged site key), as written by
    ``scripts/sibling_gate/probes_modal.py``. A score file without ``key`` leaves the site keys
    unresolved, so every per-key state fails (fail-safe)."""
    import numpy as np

    out = {}
    for cls, row in concepts.items():
        cp = ClassProbes(cls, list(row["concepts"]))
        fd = Path(probes_dir) / "fit" / slug(cls)
        for f in sorted(fd.glob("fit_L*.json")):
            res = json.loads(f.read_text())
            la = int(res["layer"])
            for c, v in zip(cp.concepts, res["validity"]["a"], strict=True):
                cp.validity[(c, la)] = float(v) if v is not None else 0.0
            keys = (res.get("null_check") or {}).get("keys") or {}
            cp.null_ok[la] = {int(k): bool(v.get("ok")) for k, v in keys.items()}
            if res.get("merge"):
                cp.merge = {int(k): int(v) for k, v in res["merge"].items()}
        cen = fd / f"centroids_L{DISTRACTOR_LAYER}.json"
        if cen.exists():
            cp.centroids = json.loads(cen.read_text())
        sd = Path(probes_dir) / "score" / slug(cls)
        if sd.exists():
            cp.z = ZTable(cp.concepts)
            for f in sorted(sd.glob("L*.npz")):
                la = int(f.stem[1:])
                z = np.load(f)
                ks = z["key"] if "key" in z.files else [None] * len(z["pos"])
                for it, p, row_z, k in zip(z["item"], z["pos"], z["za"], ks, strict=True):
                    cp.z.add(str(it), la, int(p), row_z.tolist(), None if k is None else int(k))
        out[cls] = cp
    return out


# ---------------------------------------------------------------- per-key report (rev 8.1)

RUNG2_RUNG1 = ("stable", "sparse", "deterministic")


def per_key_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per token-class key of the unit's eval site: units, the rung-1 status mix, rung-2 units
    (rung 1 stable / sparse / deterministic with a final), their share of all rung-2 units, and
    the probe_na / probe_faithful rates among them (shows any selection effect of per-key n/a)."""
    out: dict[str, dict[str, Any]] = {}
    n_r2 = 0
    for r in rows:
        k = r.get("key") or "unknown"
        d = out.setdefault(
            k,
            {
                "n_units": 0,
                "rung1": defaultdict(int),
                "n_rung2": 0,
                "probe_na": 0,
                "probe_faithful": 0,
            },
        )
        d["n_units"] += 1
        d["rung1"][r.get("rung1") or "resample_failed"] += 1
        if r.get("rung1") in RUNG2_RUNG1 and r.get("final"):
            n_r2 += 1
            d["n_rung2"] += 1
            d["probe_na"] += r["final"] == "probe_na"
            d["probe_faithful"] += r["final"] == "probe_faithful"
    for d in out.values():
        d["rung1"] = dict(d["rung1"])
        d["rung2_share"] = d["n_rung2"] / n_r2 if n_r2 else None
        d["probe_na_rate"] = d["probe_na"] / d["n_rung2"] if d["n_rung2"] else None
        d["probe_faithful_rate"] = d["probe_faithful"] / d["n_rung2"] if d["n_rung2"] else None
    order = {n: i for i, n in enumerate(TOKEN_CLASSES)}
    return dict(sorted(out.items(), key=lambda kv: (order.get(kv[0], len(order)), kv[0])))


def per_key_table(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    """Markdown: the per-key report (§3d rev 8.1)."""
    summ = per_key_summary(rows)
    if not summ:
        return []
    r1s = ("stable", "sparse", "deterministic", "unstable", "resample_failed")

    def pct(x: float | None) -> str:
        return "—" if x is None else f"{100 * x:.1f}%"

    md = [
        "### Rung 2 by eval-site key class",
        "",
        "| key | units | " + " | ".join(r1s) + " | rung-2 units | share of rung 2 | probe_na | "
        "probe_faithful |",
        "|" + "---|" * (len(r1s) + 6),
    ]
    for k, d in summ.items():
        md.append(
            f"| {k} | {d['n_units']} | "
            + " | ".join(str(d["rung1"].get(x, 0)) for x in r1s)
            + f" | {d['n_rung2']} | {pct(d['rung2_share'])} | {pct(d['probe_na_rate'])} | "
            f"{pct(d['probe_faithful_rate'])} |"
        )
    return [*md, ""]
