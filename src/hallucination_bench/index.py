"""The hallucination index (plans 0013 + 0019): one row per (method, layer).

Families (each reads another eval's outputs; nothing is re-judged):

* ``main`` (``wsbench_chat``): weighted share of claims hallucinated, ``Σ w(leaf) / n_claims``
  with :data:`MAIN_LEAF_WEIGHTS`;
* ``agentic_misalignment``, ``story``: unweighted share
  ``(false_claim + substitution_category + detail) / n_claims``;
* ``sibling``: the 3-domain user-modelling ladder, unweighted mean over domains of the share of
  committed, probe-testable units labelled hallucination (defined only with units in all three);
* ``overrefusal``, ``strongreject``, ``mask`` (hindex-v4, plan 0019): the probe-validated share of
  disposition claims, ``Σ n_contradicted / Σ n_claims`` over the eval's ``index_counts.jsonl``
  (one row per (arm, layer, item), written by the eval's report); readers restricted to
  :data:`SAFETY_READERS`.

A claim is one claim-map segment whose scoring leaf is a claim (``taxonomy_v2.is_claim``), pooled
over every segment of every mapped cell of the (arm, layer). Claim-family CIs: 2,000-draw item
bootstrap (a fresh ``default_rng(0)`` per row); sibling: 1,000 draws, a fresh ``Random(0)`` per
row, items resampled within domain; safety families: 2,000-draw cluster bootstrap (clusters in
sorted order, a fresh ``default_rng(0)`` per (family, arm, layer)). ``full`` / ``matched`` are
family-weighted means (:data:`FAMILY_WEIGHTS`) with a normal-approximation CI (families
independent).

``version="v3"`` emits exactly hindex-v3 (families :data:`FAMILIES_V3`, frozen notes
:data:`NOTES_V3`); v4 adds the safety families and always emits ``full_v3`` / ``matched_v3`` (the
v3 macros) beside its own ``full`` (all seven) / ``matched`` (all but sibling). The v3 definition
reproduces the exploratory ``hclaims*.py`` / ``um3.py`` / ``macro3.py`` exactly, down to the RNG
consumption order: change a definition only with a new :data:`INDEX_VERSION`.
"""

import json
import math
import os
import random
import subprocess
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

from hallucination_bench import REPO_ROOT
from hallucination_bench import taxonomy_v2 as tx

INDEX_VERSION = "hindex-v4"
INDEX_VERSION_V3 = "hindex-v3"
VERSIONS: dict[str, str] = {"v3": INDEX_VERSION_V3, "v4": INDEX_VERSION}
MAIN, AM, SIBLING, STORY = "main", "agentic_misalignment", "sibling", "story"
OVERREFUSAL, STRONGREJECT, MASK = "overrefusal", "strongreject", "mask"
FAMILIES_V3: tuple[str, ...] = (MAIN, AM, SIBLING, STORY)
MATCHED_V3: tuple[str, ...] = (MAIN, AM, STORY)  # comparable for methods without a sibling score
FAMILY_WEIGHTS_V3: dict[str, float] = {MAIN: 9.0, AM: 3.0, SIBLING: 2.0, STORY: 1.0}
CLAIM_FAMILIES: tuple[str, ...] = (MAIN, AM, STORY)
SAFETY_FAMILIES: tuple[str, ...] = (OVERREFUSAL, STRONGREJECT, MASK)
FAMILIES: tuple[str, ...] = FAMILIES_V3 + SAFETY_FAMILIES
MATCHED: tuple[str, ...] = tuple(f for f in FAMILIES if f != SIBLING)
FAMILY_WEIGHTS: dict[str, float] = {**FAMILY_WEIGHTS_V3, **dict.fromkeys(SAFETY_FAMILIES, 1.0)}
# main: every other leaf weighs 0 (cruxy is ignored)
MAIN_LEAF_WEIGHTS: dict[str, float] = {
    "false_claim": 1.0,
    "substitution_category": 0.5,
    "substitution_sibling": 0.5,
    "detail": 0.2,
}
UNWEIGHTED_LEAVES: tuple[str, ...] = ("false_claim", "substitution_category", "detail")
UM_DOMAINS: tuple[str, ...] = ("country", "capital_city", "holiday")
N_BOOT_CLAIMS = 2000
N_BOOT_SIBLING = 1000
N_BOOT_SAFETY = 2000
CELLS = "cells.jsonl"
UNITS = "units_labeled.jsonl"
SAFETY_COUNTS = "index_counts.jsonl"
SAFETY_META = "index_meta.json"
# plan 0019's reader allowlist (the export's arm names): rows of any other arm are ignored
SAFETY_READERS: dict[str, frozenset[str]] = {
    OVERREFUSAL: frozenset({"s3d-rl600", "mlnla-rl600-all11", "jlens"}),
    STRONGREJECT: frozenset({"s3d-rl600", "mlnla-rl600", "jlens"}),
    MASK: frozenset({"s3d-rl600", "mlnla-rl600-all11", "jlens"}),
}
# opt-in roots: read only when passed explicitly or when the variable is set (never its default)
SAFETY_ENV: dict[str, str] = {
    OVERREFUSAL: "HBENCH_SAFETY_OR_OUT",
    STRONGREJECT: "HBENCH_SAFETY_SR_OUT",
    MASK: "HBENCH_SAFETY_MASK_OUT",
}
SAFETY_INT_COLS: tuple[str, ...] = (
    "n_claims",
    "n_false",
    "n_contradicted",
    "n_unprobed",
    "n_unresolved",
    "n_failed_cells",
    "n_units",
    "n_false_units",
    "n_contradicted_units",
    "n_supported_units",
    "n_unprobed_units",
)
SAFETY_COLS: tuple[str, ...] = ("arm", "layer", "item", "cluster", *SAFETY_INT_COLS)
ABSENT_NO_CLAIMS = "no_claims"
ABSENT_ALL_UNPROBED = "no_probed_false_claims"
# hindex-v3's methods.json notes, frozen in the v3 code path (methods.json may change for v4)
NOTES_V3: dict[str, str] = {
    "jlens.sibling": "no sibling score: J-lens has committed, probe-testable UM units in at most "
    "two of the three domains at every layer, so only `matched` compares it",
    "nla-rl-L42": "single-layer L42 NLA arms (nla-rl-L42, nla-sft-L42) excluded "
    "(Camila 2026-10-20)",
    "s3d-rl600.main": "the s3d-rl600 arm wins at the layers it has, s3d-rl600-all11 fills the rest",
    "v4": "DeepSeek-V4-Flash arms (v4-*, plan 0014) are matched by name; their roots are passed "
    "beside the Qwen ones. V4 AM is read at layers 21/27/33 only, so V4 `full` / `matched` exist "
    "at 21/27/33 only",
    "v4-jlens-std-n1000.sibling": "no sibling score: no committed, probe-testable country units "
    "at any layer (as for jlens)",
}

# per-item count vector columns
C_CLAIMS, C_FC, C_CAT, C_DET, C_SIB, C_WEIGHTED, C_READOUTS, C_SEGMENTS = range(8)
N_COLS = 8
_LEAF_COL = {
    "false_claim": C_FC,
    "substitution_category": C_CAT,
    "detail": C_DET,
    "substitution_sibling": C_SIB,
}

LayerItems = dict[int, np.ndarray]  # layer -> (n_items, N_COLS), items in first-seen order
UmUnits = dict[tuple[str, int, str], list[tuple[str, str]]]  # (method, layer, domain) -> units


# ---------------------------------------------------------------- readers


def _segments(rec: dict[str, Any], family: str) -> list[dict[str, Any]] | None:
    """A mapped cell's segments (chat / story: ``stages.map == "ok"``; AM: ``map_status``)."""
    if family == AM:
        ok = rec.get("map_status") == "ok"
    else:
        ok = (rec.get("stages") or {}).get("map") == "ok"
    return rec["map"]["segments"] if ok and rec.get("map") else None


def claim_rows(cells_path: Path, family: str) -> LayerItems:
    """Per layer, per item count vectors of one arm's ``cells.jsonl`` (streamed: lines without
    segments are skipped unparsed). An item registers with its first cell that has segments."""
    if family not in CLAIM_FAMILIES:
        raise ValueError(f"not a claim family: {family!r}")
    acc: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
    with open(cells_path, encoding="utf-8") as fh:
        for line in fh:
            if '"segments"' not in line:
                continue
            rec = json.loads(line)
            segs = _segments(rec, family)
            if not segs:
                continue
            v = acc[rec["layer"]].setdefault(rec["id"], np.zeros(N_COLS))
            v[C_READOUTS] += len({s["readout"] for s in segs})
            v[C_SEGMENTS] += len(segs)
            for s in segs:
                lf = tx.JUDGE_TO_LEAF.get(s["leaf"], s["leaf"])
                if not tx.is_claim(lf):
                    continue
                v[C_CLAIMS] += 1
                if lf in MAIN_LEAF_WEIGHTS:
                    v[_LEAF_COL[lf]] += 1
                    v[C_WEIGHTED] += MAIN_LEAF_WEIGHTS[lf]
    return {layer: np.array(list(items.values())) for layer, items in sorted(acc.items())}


def um_label(unit: dict[str, Any]) -> str:
    """``um_plot_macro.py``'s rule: the probe fires at L, else a cross-method sibling, else
    hallucination."""
    if unit["probe"]["fire_L"]:
        return "faithful"
    return "cross" if unit["cross"] else "hallucination"


def um_paths(um_root: Path) -> dict[str, Path]:
    """Domain -> ``units_labeled.jsonl`` (country at ``label/``, others at ``<domain>/label/``)."""
    return {d: um_root / ("" if d == "country" else d) / "label" / UNITS for d in UM_DOMAINS}


def um_rows(um_root: Path) -> UmUnits:
    """Committed, probe-testable units of every domain present, as (item, label), file order."""
    units: UmUnits = defaultdict(list)
    for d, p in um_paths(um_root).items():
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["committed"] and r["probe"]["na_reason"] is None:
                units[(r["method"], r["layer"], d)].append((r["item"], um_label(r)))
    return dict(units)


SafetyRows = dict[str, dict[int, list[dict[str, Any]]]]  # arm -> layer -> index_counts rows


def safety_rows(root: Path, family: str) -> tuple[SafetyRows, set[str]]:
    """One safety root's ``index_counts.jsonl`` as arm -> layer -> rows, for the arms of
    :data:`SAFETY_READERS` only; also the ignored arm names. A root without the file is empty."""
    if family not in SAFETY_FAMILIES:
        raise ValueError(f"not a safety family: {family!r}")
    out: SafetyRows = defaultdict(lambda: defaultdict(list))
    ignored: set[str] = set()
    path = Path(root) / SAFETY_COUNTS
    if not path.is_file():
        return {}, ignored
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            if missing := [c for c in SAFETY_COLS if c not in r]:
                raise ValueError(f"{path}:{n}: missing columns {missing}")
            if r["arm"] not in SAFETY_READERS[family]:
                ignored.add(r["arm"])
                continue
            out[r["arm"]][int(r["layer"])].append(r)
    return {a: dict(by) for a, by in out.items()}, ignored


def safety_meta(root: Path) -> dict[str, Any] | None:
    """A safety root's ``index_meta.json`` (prompt versions, probe sha256s, gates, population)."""
    p = Path(root) / SAFETY_META
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def layer_gates(meta: Mapping[str, Any] | None, layer: int) -> Any:
    """The gate results of one layer from ``index_meta.json``'s ``gates``: a mapping keyed by
    layer (``{"20": {site class: ...}}``) or a list of entries carrying ``"layer"``."""
    g = (meta or {}).get("gates")
    if isinstance(g, Mapping):
        return g.get(str(layer), g.get(layer))
    if isinstance(g, list):
        hit = [e for e in g if isinstance(e, Mapping) and e.get("layer") == layer]
        return hit or None
    return None


# ---------------------------------------------------------------- metrics


def claim_metric(mat: np.ndarray, family: str) -> dict[str, Any] | None:
    """Pooled share and item-bootstrap CI of one (arm, layer); None when it has no claims."""
    tot = mat.sum(0)
    if tot[C_CLAIMS] == 0:
        return None
    idx = np.random.default_rng(0).integers(0, len(mat), size=(N_BOOT_CLAIMS, len(mat)))
    bs = mat[idx].sum(1)
    if family == MAIN:
        num, bnum = tot[C_WEIGHTED], bs[:, C_WEIGHTED]
    else:
        num = tot[C_FC] + tot[C_CAT] + tot[C_DET]
        bnum = bs[:, C_FC] + bs[:, C_CAT] + bs[:, C_DET]
    with np.errstate(divide="ignore", invalid="ignore"):
        ci = np.percentile(bnum / bs[:, C_CLAIMS], [2.5, 97.5]).tolist()
    n = tot[C_CLAIMS]
    out: dict[str, Any] = {"v": float(num / n), "ci": ci, "n": int(n), "n_items": len(mat)}
    if family == MAIN:
        out |= {k: float(tot[c] / n) for k, c in (("fc", C_FC), ("cat", C_CAT), ("sib", C_SIB))}
        out["det"] = float(tot[C_DET] / n)
    return out


def sibling_metric(per_domain: Mapping[str, list[tuple[str, str]]]) -> dict[str, Any] | None:
    """Domain-macro hallucination share and within-domain item-bootstrap CI; None unless every
    domain of :data:`UM_DOMAINS` has units."""
    if not all(per_domain.get(d) for d in UM_DOMAINS):
        return None
    doms: dict[str, Any] = {}
    for d in UM_DOMAINS:
        u = per_domain[d]
        doms[d] = {
            "n": len(u),
            "n_items": len({i for i, _ in u}),
            **{k: sum(lb == k for _, lb in u) / len(u) for k in ("faithful", "cross")},
            "hall": sum(lb == "hallucination" for _, lb in u) / len(u),
        }
    rng = random.Random(0)
    by: dict[str, dict[str, list[bool]]] = {d: defaultdict(list) for d in UM_DOMAINS}
    for d in UM_DOMAINS:
        for i, lb in per_domain[d]:
            by[d][i].append(lb == "hallucination")
    sims = []
    for _ in range(N_BOOT_SIBLING):
        vals = []
        for d in UM_DOMAINS:
            keys = list(by[d])
            draw = [h for k in (rng.choice(keys) for _ in keys) for h in by[d][k]]
            vals.append(sum(draw) / len(draw))
        sims.append(sum(vals) / 3)
    sims.sort()
    return {
        "v": sum(doms[d]["hall"] for d in UM_DOMAINS) / 3,
        "ci": [sims[24], sims[974]],  # of N_BOOT_SIBLING = 1000 sorted draws
        "n": sum(doms[d]["n"] for d in UM_DOMAINS),
        "n_items": sum(doms[d]["n_items"] for d in UM_DOMAINS),
        "per_domain": doms,
    }


def safety_metric(rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Probe-validated share ``Σ n_contradicted / Σ n_claims`` of one (family, arm, layer) and its
    cluster-bootstrap CI (clusters in sorted order, a fresh ``default_rng(0)``; percentiles over
    the draws with claims), or ``None`` with the counts and an ``absent_reason``: no claims, or
    false claims none of which is probed."""
    tot = {c: int(sum(int(r[c]) for r in rows)) for c in SAFETY_INT_COLS}
    counts: dict[str, Any] = {"n_items": len({r["item"] for r in rows}), **tot}
    if tot["n_claims"] == 0:
        return None, {**counts, "absent_reason": ABSENT_NO_CLAIMS}
    if tot["n_false"] > 0 and tot["n_false"] == tot["n_unprobed"]:
        return None, {**counts, "absent_reason": ABSENT_ALL_UNPROBED}
    by: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(2))
    for r in rows:
        by[str(r["cluster"])] += (int(r["n_contradicted"]), int(r["n_claims"]))
    mat = np.array([by[k] for k in sorted(by)])
    idx = np.random.default_rng(0).integers(0, len(mat), size=(N_BOOT_SAFETY, len(mat)))
    bs = mat[idx].sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = bs[:, 0] / bs[:, 1]
    # draws of zero-claim clusters only (0/0) carry no share: the CI is over the defined draws
    ci = np.nanpercentile(ratio, [2.5, 97.5]).tolist()
    n = tot["n_claims"]
    out: dict[str, Any] = {
        "v": tot["n_contradicted"] / n,
        "ci": ci,
        "n": n,
        "n_items": counts["n_items"],
        "n_clusters": len(mat),
        "raw": tot["n_false"] / n,
        "n_claims": n,
        "n_false": tot["n_false"],
        "n_contradicted": tot["n_contradicted"],
        "n_unprobed": tot["n_unprobed"],
        "health": {k: tot[k] for k in ("n_unprobed", "n_unresolved", "n_failed_cells")},
        "units": {k: tot[k] for k in SAFETY_INT_COLS if k.endswith("_units")},
    }
    return out, counts


def weighted_macro(per: Mapping[str, Mapping[str, Any]], fams: Iterable[str]) -> dict[str, Any]:
    """Family-weighted mean; CI ``v ± 1.96 se``, ``se`` from each family's CI width / 3.92."""
    fams = list(fams)
    tw = sum(FAMILY_WEIGHTS[f] for f in fams)
    v = sum(FAMILY_WEIGHTS[f] * per[f]["v"] for f in fams) / tw
    se = (
        math.sqrt(
            sum((FAMILY_WEIGHTS[f] * (per[f]["ci"][1] - per[f]["ci"][0]) / 3.92) ** 2 for f in fams)
        )
        / tw
    )
    return {"v": v, "ci": [v - 1.96 * se, v + 1.96 * se]}


# ---------------------------------------------------------------- methods


@dataclass
class Methods:
    """``data/index/methods.json``: configured methods (family -> ordered arm list), exclusions."""

    methods: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    exclude: set[str] = field(default_factory=set)
    notes: dict[str, str] = field(default_factory=dict)


def load_methods(path: Path) -> Methods:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    m = Methods(
        methods={k: {f: list(a) for f, a in v.items()} for k, v in raw.get("methods", {}).items()},
        exclude=set(raw.get("exclude", [])),
        notes=dict(raw.get("notes", {})),
    )
    owner: dict[tuple[str, str], str] = {}
    for name, fams in m.methods.items():
        for f, arms in fams.items():
            for a in arms:
                if (prev := owner.setdefault((f, a), name)) != name:
                    raise ValueError(f"{path}: {f} arm {a!r} listed by {prev!r} and {name!r}")
        if bad := set(fams) - set(FAMILIES):
            raise ValueError(f"{path}: method {name!r}: unknown families {sorted(bad)}")
        if hit := ({name} | {a for arms in fams.values() for a in arms}) & m.exclude:
            raise ValueError(f"{path}: method {name!r} uses excluded names {sorted(hit)}")
    return m


def resolve_methods(
    methods: Methods, arms: Mapping[str, Iterable[str]], families: Iterable[str] = FAMILIES
) -> dict[str, dict]:
    """Method -> family -> ordered arm list over ``families``. A configured method uses its list
    where it names the family, else ``[name]``; every arm neither claimed in its family (listed
    there, or the name of a configured method) nor excluded becomes the identity method of that
    name."""
    families = tuple(families)
    out: dict[str, dict[str, list[str]]] = {
        name: {f: list(fams.get(f, [name])) for f in families}
        for name, fams in methods.methods.items()
    }
    for f in families:
        claimed = set(methods.methods) | {a for m in out.values() for a in m[f]}
        for a in sorted(set(arms.get(f, ())) - claimed - methods.exclude):
            out.setdefault(a, {g: [] for g in families})[f] = [a]
    return out


# ---------------------------------------------------------------- build


def default_sources(out: Path, judge_model: str) -> dict[str, list[Path]]:
    from hallucination_bench.evals.wsbench_chat.run import model_slug

    m = model_slug(judge_model)
    src = {
        MAIN: [out / "wsbench_chat" / m],
        AM: [out / "agentic_misalignment" / m],
        SIBLING: [out / "sibling" / "um"],
        STORY: [out / "story" / m],
    }
    # safety roots are opt-in: only an explicitly set HBENCH_SAFETY_* (never the dev-box default)
    for f, var in SAFETY_ENV.items():
        if v := os.environ.get(var):
            src[f] = [Path(v)]
    return src


def _arm_dirs(family: str, roots: list[Path]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for d in sorted(p for p in root.iterdir() if (p / CELLS).is_file()):
            if d.name in found:
                raise ValueError(f"{family}: arm {d.name!r} under two roots: {found[d.name]}, {d}")
            found[d.name] = d
    return found


def _um_units(roots: list[Path]) -> UmUnits:
    units: UmUnits = {}
    seen: dict[str, Path] = {}
    for root in roots:
        part = um_rows(root)
        for name in sorted({k[0] for k in part}):
            if name in seen:
                raise ValueError(
                    f"{SIBLING}: method {name!r} under two roots: {seen[name]}, {root}"
                )
            seen[name] = root
        units |= part
    return units


def _claim_job(job: tuple[str, str, Path]) -> tuple[str, str, dict[int, dict[str, Any]]]:
    family, arm, d = job
    rows = {L: claim_metric(mat, family) for L, mat in claim_rows(d / CELLS, family).items()}
    return family, arm, {L: r for L, r in rows.items() if r is not None}


def _safety_arms(
    family: str, roots: list[Path]
) -> tuple[SafetyRows, dict[str, dict[str, Any] | None], set[str], list[dict[str, Any]]]:
    rows: SafetyRows = {}
    meta: dict[str, dict[str, Any] | None] = {}
    seen: dict[str, Path] = {}
    ignored: set[str] = set()
    metas = []
    for root in roots:
        part, ign = safety_rows(root, family)
        ignored |= ign
        m = safety_meta(root)
        metas.append({"root": str(Path(root).resolve()), "meta": m})
        for arm in sorted(part):
            if arm in seen:
                raise ValueError(f"{family}: arm {arm!r} under two roots: {seen[arm]}, {root}")
            seen[arm] = Path(root)
            rows[arm] = part[arm]
            meta[arm] = m
    return rows, meta, ignored, metas


def build_index(
    sources: Mapping[str, list[Path]],
    methods: Methods,
    workers: int = 1,
    version: str = "v4",
) -> dict[str, Any]:
    """Every (method, layer) row with the families it has. v4: ``full`` needs all seven families,
    ``matched`` all but sibling, and ``full_v3`` / ``matched_v3`` are the v3 macros; v3
    (``version="v3"``): exactly hindex-v3 (``full`` = four families, ``matched`` =
    :data:`MATCHED_V3`). Missing roots mean an absent family, not an error."""
    if version not in VERSIONS:
        raise ValueError(f"unknown index version {version!r}: one of {sorted(VERSIONS)}")
    v3 = version == "v3"
    families = FAMILIES_V3 if v3 else FAMILIES
    safety_fams = () if v3 else SAFETY_FAMILIES
    arm_dirs = {f: _arm_dirs(f, list(sources.get(f, []))) for f in CLAIM_FAMILIES}
    units = _um_units(list(sources.get(SIBLING, [])))
    safety = {f: _safety_arms(f, list(sources.get(f, []))) for f in safety_fams}
    arms = {**{f: list(arm_dirs[f]) for f in CLAIM_FAMILIES}, SIBLING: [k[0] for k in units]}
    arms |= {f: list(safety[f][0]) for f in safety_fams}
    resolved = resolve_methods(methods, arms, families)

    used = {(f, a) for m in resolved.values() for f in CLAIM_FAMILIES for a in m[f]}
    jobs = [(f, a, d) for f in CLAIM_FAMILIES for a, d in arm_dirs[f].items() if (f, a) in used]
    if workers > 1 and len(jobs) > 1:
        with Pool(min(workers, len(jobs))) as pool:
            done = pool.map(_claim_job, jobs, chunksize=1)
    else:
        done = [_claim_job(j) for j in jobs]
    by_arm: dict[tuple[str, str], dict[int, dict[str, Any]]] = {(f, a): r for f, a, r in done}

    grouped: dict[tuple[str, int], dict[str, list[tuple[str, str]]]] = defaultdict(dict)
    for (name, layer, d), u in units.items():
        grouped[(name, layer)][d] = u
    for (name, layer), per in grouped.items():
        if (r := sibling_metric(per)) is not None:
            by_arm.setdefault((SIBLING, name), {})[layer] = r

    absent: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    for f in safety_fams:
        rows_f, meta_f, _, _ = safety[f]
        for a in sorted(rows_f):
            for layer, rs in sorted(rows_f[a].items()):
                r, counts = safety_metric(rs)
                if r is None:
                    absent.setdefault((f, a), {})[layer] = counts
                    continue
                gates = layer_gates(meta_f[a], layer)
                by_arm.setdefault((f, a), {})[layer] = {**r, "gates": gates}

    rows = []
    for method in sorted(resolved):
        fam_arms = resolved[method]
        layers = sorted({L for f in families for a in fam_arms[f] for L in by_arm.get((f, a), {})})
        for layer in layers:
            per: dict[str, dict[str, Any]] = {}
            gone: dict[str, dict[str, Any]] = {}
            for f in families:
                for a in fam_arms[f]:
                    if (r := by_arm.get((f, a), {}).get(layer)) is not None:
                        per[f] = {"arm": a, **r}
                        break
                if f in safety_fams and f not in per:
                    for a in fam_arms[f]:
                        if (c := absent.get((f, a), {}).get(layer)) is not None:
                            gone[f] = {"arm": a, "absent_reason": c["absent_reason"]}
                            break
            row: dict[str, Any] = {"method": method, "layer": layer, "families": per}
            if v3:
                if all(f in per for f in FAMILIES_V3):
                    row["full"] = weighted_macro(per, FAMILIES_V3)
                if all(f in per for f in MATCHED_V3):
                    row["matched"] = weighted_macro(per, MATCHED_V3)
            else:
                for key, fams in (
                    ("full", FAMILIES),
                    ("matched", MATCHED),
                    ("full_v3", FAMILIES_V3),
                    ("matched_v3", MATCHED_V3),
                ):
                    if all(f in per for f in fams):
                        row[key] = weighted_macro(per, fams)
                if gone:
                    row["absent"] = gone
            rows.append(row)
    weights: dict[str, Any] = {
        "family": FAMILY_WEIGHTS_V3 if v3 else FAMILY_WEIGHTS,
        "main_leaf": MAIN_LEAF_WEIGHTS,
        "unweighted_leaves": list(UNWEIGHTED_LEAVES),
    }
    out: dict[str, Any] = {
        "version": VERSIONS[version],
        "weights": weights,
        "sources": {f: [str(Path(p).resolve()) for p in sources.get(f, [])] for f in families},
        "methods": resolved,
        "exclude": sorted(methods.exclude),
        "notes": NOTES_V3 if v3 else methods.notes,
        "rows": rows,
    }
    if not v3:
        out["safety"] = {
            f: {
                "share": "n_contradicted / n_claims (probe-validated); raw = n_false / n_claims",
                "readers": sorted(SAFETY_READERS[f]),
                "ignored_arms": sorted(safety[f][2]),
                "meta": safety[f][3],
                "absent": [
                    {"arm": a, "layer": layer, **c}
                    for (g, a), by in sorted(absent.items())
                    if g == f
                    for layer, c in sorted(by.items())
                ],
            }
            for f in SAFETY_FAMILIES
        }
    return out


# ---------------------------------------------------------------- output


def _pct(r: Mapping[str, Any] | None) -> str:
    if r is None:
        return "—"
    return f"{100 * r['v']:.1f} [{100 * r['ci'][0]:.1f}, {100 * r['ci'][1]:.1f}]"


def _render_md_v3(index: Mapping[str, Any]) -> str:
    """hindex-v3's table, verbatim."""
    fw = index["weights"]["family"]
    lw = index["weights"]["main_leaf"]
    L = [
        f"# Hallucination index ({index['version']})",
        "",
        "% of claims hallucinated per (method, layer), 95% CIs. Family weights "
        + ", ".join(f"{f} {w:g}" for f, w in fw.items())
        + "; `matched` = main + agentic_misalignment + story. main leaf weights "
        + ", ".join(f"{k} {w:g}" for k, w in lw.items())
        + "; agentic_misalignment / story: unweighted "
        + " + ".join(index["weights"]["unweighted_leaves"])
        + "; sibling: 3-domain UM ladder.",
        "",
        "| method | layer | full | matched | main | agentic_misalignment | sibling | story "
        "| main fc / cat / sib / det |",
        "|---|---:|---|---|---|---|---|---|---|",
    ]
    for r in index["rows"]:
        p = r["families"]
        main = p.get(MAIN)
        comp = (
            " / ".join(f"{100 * main[k]:.1f}" for k in ("fc", "cat", "sib", "det")) if main else "—"
        )
        cells = [r["method"], str(r["layer"]), _pct(r.get("full")), _pct(r.get("matched"))]
        cells += [_pct(p.get(f)) for f in FAMILIES_V3] + [comp]
        L.append("| " + " | ".join(cells) + " |")
    L += _md_footer(index)
    return "\n".join(L) + "\n"


def _md_footer(index: Mapping[str, Any]) -> list[str]:
    L: list[str] = []
    arms = sorted(
        {
            f"{r['method']}.{f} = {a['arm']}"
            for r in index["rows"]
            for f, a in r["families"].items()
            if a["arm"] != r["method"]
        }
    )
    if arms:
        L += ["", "Arms used under another name (at one or more layers): " + "; ".join(arms) + "."]
    if index["exclude"]:
        L += ["", "Excluded arms: " + ", ".join(index["exclude"]) + "."]
    if index["notes"]:
        L += ["", *(f"- {k}: {v}" for k, v in sorted(index["notes"].items()))]
    return L


def render_md(index: Mapping[str, Any]) -> str:
    if index["version"] == INDEX_VERSION_V3:
        return _render_md_v3(index)
    fw = index["weights"]["family"]
    lw = index["weights"]["main_leaf"]
    L = [
        f"# Hallucination index ({index['version']})",
        "",
        "% of claims hallucinated per (method, layer), 95% CIs. Family weights "
        + ", ".join(f"{f} {w:g}" for f, w in fw.items())
        + "; `full` = all seven families, `matched` = all but sibling, `full_v3` = hindex-v3's "
        "`full` (main + agentic_misalignment + sibling + story). main leaf weights "
        + ", ".join(f"{k} {w:g}" for k, w in lw.items())
        + "; agentic_misalignment / story: unweighted "
        + " + ".join(index["weights"]["unweighted_leaves"])
        + "; sibling: 3-domain UM ladder; overrefusal / strongreject / mask: probe-validated "
        "share of disposition claims (false claims the eval's probe contradicts / claims).",
        "",
        "| method | layer | full | matched | full_v3 | " + " | ".join(FAMILIES) + " "
        "| main fc / cat / sib / det |",
        "|---|---:|" + "---|" * (3 + len(FAMILIES) + 1),
    ]
    for r in index["rows"]:
        p = r["families"]
        main = p.get(MAIN)
        comp = (
            " / ".join(f"{100 * main[k]:.1f}" for k in ("fc", "cat", "sib", "det")) if main else "—"
        )
        cells = [r["method"], str(r["layer"])]
        cells += [_pct(r.get(k)) for k in ("full", "matched", "full_v3")]
        cells += [_pct(p.get(f)) for f in FAMILIES] + [comp]
        L.append("| " + " | ".join(cells) + " |")
    gone = sorted(
        f"{r['method']}.{f}@{r['layer']} ({a['absent_reason']})"
        for r in index["rows"]
        for f, a in r.get("absent", {}).items()
    )
    if gone:
        L += ["", "Safety families absent (not 0): " + "; ".join(gone) + "."]
    L += _md_footer(index)
    return "\n".join(L) + "\n"


def git_sha() -> str | None:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return res.stdout.strip()


def write_index(index: dict[str, Any], dest: Path) -> tuple[Path, Path]:
    dest.mkdir(parents=True, exist_ok=True)
    js, md = dest / "index.json", dest / "index.md"
    js.write_text(json.dumps({**index, "git_sha": git_sha()}, indent=1) + "\n", encoding="utf-8")
    md.write_text(render_md(index), encoding="utf-8")
    return js, md
