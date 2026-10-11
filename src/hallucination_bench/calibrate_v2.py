"""Calibrate the claim-map judge (taxonomy v2) on unit-level gold and apply the held-out gate.

Plan 0002, §Calibration. Cases (``data/calibration/{dev,heldout}_v2.jsonl``, schema in
``data/calibration/README.md``) carry what the judge sees in production (prompt, marked
response, readouts, the established v5c spans) plus gold on the **production unit split**:
a leaf per unit, and sub-spans with leaves (plus a ``rest_leaf``) for units that need
splitting. :func:`validate_case` re-runs the splitter and refuses a case whose gold unit ids or
texts no longer match it. Units are shared, so prediction and gold match one-to-one by unit.

Scoring (one repeat at a time, then the mean over repeats with a Wilson 95% CI):

* **segment level** (recall rows, per-leaf recall, sibling-vs-category): each gold segment's
  predicted leaf is the majority-character predicted leaf over that segment's own characters
  gold segments with no letter or digit (a bare ".") are left out here;
* **unit level** (accuracy, flip rate, real-part accuracy): each side's majority-character leaf
  over the unit (a unit split on neither side is just its leaf);
* **character level** (accuracy, claim/non-claim F1): every character of a gold segment.

**Single-call mode** (plan rev 13, the default): the cases keep their production unit split
(established spans are still their own units, so gold ids are stable), but the established
flags are ignored: those units are shown and judged as ordinary text. ``--anchor v5c``
restores rev 12 (flags shown, forced / disputed rules). There, a predicted
``unknown_incorrect`` is scored by the judge's ``alt_leaf`` if disputed, or its raw leaf if
forced (:func:`claimmap_v2.resolved_leaf`); forcing and dispute counts are reported.

Gold written before rev 13 is re-mapped mechanically on load (``taxonomy_v2.LEAF_RENAMES``:
``world_true`` -> ``supplementary_true``, ``possible_belief`` -> ``supplementary_false``).
Optional gold ``probe_target`` strings on probe-candidate units are reported next to the
judge's (informational, not gated).
Each repeat is a separate call (the repeat index is in the cache key and the fingerprint).
On ``heldout`` only aggregates are reported.
"""

import asyncio
import functools
import json
import warnings
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hallucination_bench import DATA
from hallucination_bench import taxonomy_v2 as tx
from hallucination_bench.cache import Cache, fingerprint
from hallucination_bench.calibrate import wilson
from hallucination_bench.evals.wsbench_chat.score_v2 import majority_leaf
from hallucination_bench.judges import claimmap_v2 as cm
from hallucination_bench.judges.v5c import READ_MARK
from hallucination_bench.llm import LLM, Usage
from hallucination_bench.units import Unit, split_cell

CAL_DIR = DATA / "calibration"
SPLITS = ("dev", "heldout")
RECALL_ROWS: dict[str, frozenset[str]] = {
    "false_claim": frozenset({"false_claim"}),
    "detail": frozenset({"detail"}),
    "substitution": frozenset(tx.SUBSTITUTION_LEAVES),
    "correct": frozenset(tx.CORRECT_LEAVES),
}
WILSON_LO = 0.58
# (measure, key, threshold, higher_is_better)
GATE: tuple[tuple[str, str, float, bool], ...] = (
    ("group accuracy (8 groups), unit level", "group_acc_unit", 0.85, True),
    ("group accuracy, character level", "group_acc_char", 0.85, True),
    (f"verdict accuracy ({len(tx.VERDICTS)} verdicts), unit level", "verdict_acc_unit", 0.85, True),
    (f"leaf accuracy ({len(tx.LEAF_NAMES)} leaves), unit level", "leaf_acc_unit", 0.70, True),
    *(
        (f"recall of {k} (segment level)", f"recall_{k}", 0.75, True)
        for k in ("false_claim", "detail", "substitution", "correct")
    ),
    ("sibling-vs-category accuracy (segment level)", "sibling_vs_category", 0.75, True),
    ("claim/non-claim character F1", "claim_f1_char", 0.85, True),
    ("group flip rate between runs", "group_flip_rate", 0.07, False),
    ("real-part group accuracy", "real_group_acc_unit", 0.80, True),
    ("n_unlocated_subspans / splits", "unlocated_subspan_rate", 0.02, False),
)
INVALID_RETRIES = 2
DEV_MIN_UNITS = 100
DEV_MIN_PER_LEAF = 3


def cases_path(split: str) -> Path:
    return CAL_DIR / f"{split}_v2.jsonl"


def _rename(x: Any, fallbacks: Counter | None = None) -> Any:
    if not isinstance(x, str):
        return x
    if x in tx.FALLBACK_RENAMES:
        if fallbacks is not None:
            fallbacks[x] += 1
        return tx.FALLBACK_RENAMES[x]
    return tx.LEAF_RENAMES.get(x, x)


def _merge_tags(d: dict[str, Any], old: Any, key: str) -> None:
    t = tx.MERGE_TAGS.get(old) if isinstance(old, str) else None
    if t:
        d[key] = {**t, **(d.get(key) or {})}


def remap_case(c: dict[str, Any], fallbacks: Counter | None = None) -> dict[str, Any]:
    """The case with old gold leaf names mapped (a new dict; the input is not changed): every
    earlier leaf goes mechanically through the rev-16 merges (``taxonomy_v2.LEAF_RENAMES``), and
    the tags the merge carries (``MERGE_TAGS``: ``reason`` / ``source`` / the unverifiable flags)
    are recorded under ``tags`` (unit leaf and splits) and ``rest_tags`` (rest leaf).
    ``FALLBACK_RENAMES`` (none since rev 15) are counted in ``fallbacks`` and
    :func:`load_cases` warns. Judgement calls (contradicted old ``unrealised_*``) are not made
    here."""
    rn = functools.partial(_rename, fallbacks=fallbacks)
    gold = c.get("gold")
    if not isinstance(gold, dict) or not isinstance(gold.get("units"), list):
        return c
    units = []
    for g in gold["units"]:
        if not isinstance(g, dict):
            units.append(g)
            continue
        g = {**g}
        _merge_tags(g, g.get("leaf"), "tags")
        _merge_tags(g, g.get("rest_leaf"), "rest_tags")
        for k in ("leaf", "rest_leaf", "alt_leaf"):
            if k in g:
                g[k] = rn(g[k])
        if isinstance(g.get("splits"), list):
            splits = []
            for sp in g["splits"]:
                if isinstance(sp, dict):
                    sp = {**sp}
                    _merge_tags(sp, sp.get("leaf"), "tags")
                    sp["leaf"] = rn(sp.get("leaf"))
                splits.append(sp)
            g["splits"] = splits
        units.append(g)
    return {**c, "gold": {**gold, "units": units}}


# ---------------------------------------------------------------- cases


def case_units(c: dict[str, Any]) -> tuple[list[Unit], list[int]]:
    est = c.get("established") or [[] for _ in c["readouts"]]
    return split_cell(c["readouts"], est)


def validate_case(c: dict[str, Any], *, require_gold: bool = True) -> list[str]:
    """Schema errors for one case (empty when valid); old leaf names are accepted."""
    c = remap_case(c)
    cid = c.get("case_id", "?")
    errs: list[str] = []
    for k in ("case_id", "prompt", "response", "readouts", "gold"):
        if k not in c:
            errs.append(f"{cid}: missing {k}")
    if errs:
        return errs
    if c["response"].count(READ_MARK) != 1:
        errs.append(f"{cid}: response must contain {READ_MARK} exactly once")
    rs = c["readouts"]
    if not isinstance(rs, list) or not 1 <= len(rs) <= 3 or not all(isinstance(r, str) for r in rs):
        return [*errs, f"{cid}: readouts must be a list of 1-3 strings"]
    if c.get("part", "synthetic") not in ("synthetic", "real"):
        errs.append(f"{cid}: part must be synthetic|real")
    if c.get("kind", "prose") not in ("prose", "tokens"):
        errs.append(f"{cid}: kind must be prose|tokens")
    est = c.get("established")
    if est is not None and (
        not isinstance(est, list)
        or len(est) != len(rs)
        or not all(isinstance(x, list) and all(isinstance(s, str) for s in x) for x in est)
    ):
        return [*errs, f"{cid}: established must be one list of strings per readout"]
    units, unloc = case_units(c)
    if any(unloc):
        errs.append(f"{cid}: established span(s) not found in their readout: {unloc}")
    gold = c["gold"].get("units") if isinstance(c["gold"], dict) else None
    if not isinstance(gold, list) or len(gold) != len(units):
        n = len(gold) if isinstance(gold, list) else "?"
        return [*errs, f"{cid}: gold.units must list all {len(units)} production units (got {n})"]
    n_labelled = 0
    for u, g in zip(units, gold, strict=True):
        tag = f"{cid}/{u.id}"
        if g.get("id") != u.id or g.get("text") != u.text:
            errs.append(f"{tag}: gold unit does not match the splitter ({g.get('id')!r})")
            continue
        leaf = g.get("leaf")
        splits = g.get("splits") or []
        if leaf is None and not splits:
            if require_gold and c.get("part", "synthetic") == "synthetic":
                errs.append(f"{tag}: synthetic units need gold")
            continue
        n_labelled += 1
        pt = g.get("probe_target")
        if pt is not None and not (isinstance(pt, str) and pt.strip()):
            errs.append(f"{tag}: gold probe_target must be a non-empty string or absent")
        if not splits and leaf not in tx.LEAVES:
            errs.append(f"{tag}: bad gold leaf {leaf!r}")
            continue
        if splits:
            cnt = dict.fromkeys(cm.PARSE_COUNTS, 0)
            pieces = cm._pieces(u, {"leaf": leaf, **g}, cnt)
            if cnt["n_unlocated_subspans"]:
                errs.append(f"{tag}: a gold split is not a verbatim, in-order sub-span")
            if cnt["n_missing_rest"]:
                errs.append(f"{tag}: gold rest_leaf is required (the splits leave a remainder)")
            if any(p.leaf is None for p in pieces):
                errs.append(f"{tag}: bad gold split leaf")
        if u.established:
            gl = [p.leaf for p in _gold_pieces(u, g)]
            if not any(x in tx.INCORRECT_LEAVES for x in gl) and not g.get("dispute"):
                errs.append(f"{tag}: established unit needs an incorrect gold leaf or dispute")
    if require_gold and n_labelled == 0:
        errs.append(f"{cid}: no gold-labelled unit")
    return errs


def load_cases(split: str, path: Path | None = None) -> list[dict[str, Any]]:
    p = path or cases_path(split)
    fallbacks: Counter = Counter()
    cases = [
        remap_case(json.loads(x), fallbacks)
        for x in p.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    if fallbacks:
        warnings.warn(
            f"{p}: {sum(fallbacks.values())} gold label(s) use the pre-rev-14 leaf "
            f"`unrealised_prediction`; mapped to `unrealised_option` as a fallback. Re-label "
            "them (rev 14: most unrealised predictions are incorrect).",
            stacklevel=2,
        )
    errs = [e for c in cases for e in validate_case(c)]
    if errs:
        raise ValueError(f"{p}: invalid cases:\n  " + "\n  ".join(errs[:30]))
    ids = Counter(c["case_id"] for c in cases)
    dup = [k for k, n in ids.items() if n > 1]
    if dup:
        raise ValueError(f"{p}: duplicate case_id {dup}")
    return cases


def skeleton(c: dict[str, Any]) -> dict[str, Any]:
    """The case with ``gold.units`` filled with the production units (``leaf: null``) for a
    labeller to complete; existing gold for matching units is kept."""
    units, _ = case_units(c)
    old = {
        (g.get("id"), g.get("text")): g
        for g in ((c.get("gold") or {}).get("units") or [])
        if isinstance(g, dict)
    }
    gold = []
    for u in units:
        g = old.get((u.id, u.text))
        gold.append(
            g
            if g is not None
            else {
                "id": u.id,
                "text": u.text,
                "established": u.established,
                "leaf": None,
                "splits": [],
                "rest_leaf": None,
            }
        )
    return {**c, "gold": {"units": gold}}


def case_prompt(c: dict[str, Any], units: Sequence[Unit], anchor: str = "none") -> tuple[str, str]:
    return cm.render_prompt(
        c["prompt"], c["response"], units, kind=c.get("kind", "prose"), anchor=anchor
    )


def mode_units(c: dict[str, Any], anchor: str = "none") -> list[Unit]:
    """The case's production units; in single-call mode with the established flags dropped."""
    units, _ = case_units(c)
    return units if cm.check_anchor(anchor) else cm.plain_units(units)


# ---------------------------------------------------------------- gold / pred segments

Span = tuple[int, int, str]  # raw readout offsets + leaf


def _gold_pieces(u: Unit, g: dict[str, Any]) -> list[cm._Piece]:
    cnt = dict.fromkeys(cm.PARSE_COUNTS, 0)
    return cm._pieces(u, {"leaf": g.get("leaf"), **g}, cnt)


def gold_segments(u: Unit, g: dict[str, Any]) -> list[Span]:
    return [(u.start + p.start, u.start + p.end, p.leaf or tx.FAILED) for p in _gold_pieces(u, g)]


def pred_segments(segs: Sequence[cm.Segment]) -> dict[str, list[Span]]:
    out: dict[str, list[Span]] = {}
    for s in segs:
        out.setdefault(s.unit, []).append((s.start, s.end, cm.resolved_leaf(s)))
    return out


def has_alnum(text: str) -> bool:
    """Segment-level measures skip gold segments with no letter or digit (".", "—")."""
    return any(ch.isalnum() for ch in text)


def _why(segs: Sequence[cm.Segment], unit: str, a: int, b: int) -> str:
    return next((s.why for s in segs if s.unit == unit and s.start < b and a < s.end), "")


def _probe_at(segs: Sequence[cm.Segment], unit: str, a: int, b: int) -> str | None:
    """The judge's probe_target on the segment covering most of ``[a, b)`` of ``unit``."""
    best: tuple[int, str | None] = (0, None)
    for s in segs:
        ov = min(b, s.end) - max(a, s.start)
        if s.unit == unit and ov > best[0]:
            best = (ov, s.probe_target)
    return best[1]


def _leaf_at(spans: Sequence[Span], i: int) -> str | None:
    return next((lf for a, b, lf in spans if a <= i < b), None)


# ---------------------------------------------------------------- one repeat


def _ratio(a: float, b: float) -> float | None:
    return a / b if b else None


def run_metrics(
    cases: Sequence[dict[str, Any]], preds: Sequence[tuple[list[cm.Segment], dict[str, int]]]
) -> dict[str, Any]:
    """Metrics of one repeat; ``preds[i]`` = (segments, parse counts) of case ``i``."""
    c: Counter = Counter()
    gold_n: Counter = Counter()
    gold_hit: Counter = Counter()
    unit_groups: dict[tuple[str, str], str] = {}
    disagree: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    for case, (segs, counts) in zip(cases, preds, strict=True):
        units, _ = case_units(case)
        pred = pred_segments(segs)
        real = case.get("part") == "real"
        for k in ("n_splits", "n_unlocated_subspans", "n_forced_established"):
            c[k] += counts.get(k, 0)
        c["n_disputed_established"] += counts.get("n_disputed_established", 0)
        c["n_missing_units"] += counts.get("n_missing_units", 0)
        c["n_probe_candidates"] += counts.get("n_probe_candidates", 0)
        c["n_missing_probe_targets"] += counts.get("n_missing_probe_targets", 0)
        for u, g in zip(units, case["gold"]["units"], strict=True):
            if g.get("leaf") is None and not g.get("splits"):
                continue  # unlabelled context unit
            gs = gold_segments(u, g)
            ps = pred.get(u.id, [])
            gu = majority_leaf(gs, u.start, u.end) or tx.FAILED
            pu = majority_leaf(ps, u.start, u.end) or tx.FAILED
            c["n_units"] += 1
            c["leaf_ok_unit"] += gu == pu
            c["verdict_ok_unit"] += tx.verdict_of(gu) == tx.verdict_of(pu)
            if g.get("probe_target"):
                probe_rows.append(
                    {
                        "case_id": case["case_id"],
                        "unit": u.id,
                        "gold_leaf": gu,
                        "pred_leaf": pu,
                        "gold": g["probe_target"],
                        "pred": _probe_at(segs, u.id, u.start, u.end),
                    }
                )
            c["group_ok_unit"] += tx.group_of(gu) == tx.group_of(pu)
            unit_groups[(case["case_id"], u.id)] = tx.group_of(pu)
            if real:
                c["n_real_units"] += 1
                c["real_group_ok_unit"] += tx.group_of(gu) == tx.group_of(pu)
            for a, b, gl in gs:
                # segment level (pure punctuation fragments are not segments here)
                if has_alnum(u.text[a - u.start : b - u.start]):
                    pl = majority_leaf(ps, a, b) or tx.FAILED
                    c["n_gold_segments"] += 1
                    gold_n[gl] += 1
                    gold_hit[gl] += pl == gl
                    for row, leaves in RECALL_ROWS.items():
                        if gl in leaves:
                            gold_n[f"row_{row}"] += 1
                            gold_hit[f"row_{row}"] += pl in leaves
                    if gl in tx.SUBSTITUTION_LEAVES and pl in tx.SUBSTITUTION_LEAVES:
                        c["n_sub_both"] += 1
                        c["sub_same"] += pl == gl
                    if pl != gl:
                        disagree.append(
                            {
                                "case_id": case["case_id"],
                                "rules": case.get("rules") or case.get("rule"),
                                "unit": u.id,
                                "text": u.text[a - u.start : b - u.start],
                                "gold": gl,
                                "pred": pl,
                                "why": _why(segs, u.id, a, b),
                            }
                        )
                # character level
                for i in range(a, b):
                    if u.text[i - u.start].isspace():
                        continue
                    pc = _leaf_at(ps, i) or pu
                    c["n_chars"] += 1
                    c["leaf_ok_char"] += pc == gl
                    c["group_ok_char"] += tx.group_of(pc) == tx.group_of(gl)
                    gcl, pcl = tx.is_claim(gl), tx.is_claim(pc)
                    c["tp"] += gcl and pcl
                    c["fp"] += pcl and not gcl
                    c["fn"] += gcl and not pcl
    prec = _ratio(c["tp"], c["tp"] + c["fp"])
    rec = _ratio(c["tp"], c["tp"] + c["fn"])
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if c["n_chars"] else None)
    out: dict[str, Any] = {
        "n_units": c["n_units"],
        "n_gold_segments": c["n_gold_segments"],
        "leaf_acc_unit": _ratio(c["leaf_ok_unit"], c["n_units"]),
        "group_acc_unit": _ratio(c["group_ok_unit"], c["n_units"]),
        "verdict_acc_unit": _ratio(c["verdict_ok_unit"], c["n_units"]),
        "leaf_acc_char": _ratio(c["leaf_ok_char"], c["n_chars"]),
        "group_acc_char": _ratio(c["group_ok_char"], c["n_chars"]),
        "claim_f1_char": f1,
        "real_group_acc_unit": _ratio(c["real_group_ok_unit"], c["n_real_units"]),
        "sibling_vs_category": _ratio(c["sub_same"], c["n_sub_both"]),
        "unlocated_subspan_rate": _ratio(c["n_unlocated_subspans"], c["n_splits"]) or 0.0,
        "n_splits": c["n_splits"],
        "n_unlocated_subspans": c["n_unlocated_subspans"],
        "n_forced_established": c["n_forced_established"],
        "n_disputed_established": c["n_disputed_established"],
        "n_missing_units": c["n_missing_units"],
        "n_probe_candidates": c["n_probe_candidates"],
        "n_missing_probe_targets": c["n_missing_probe_targets"],
        "probe_targets": probe_rows,
        "leaf_recall": {
            lf: {"recall": _ratio(gold_hit[lf], gold_n[lf]), "n": gold_n[lf]}
            for lf in tx.LEAF_NAMES
        },
        "_unit_groups": unit_groups,
        "_disagreements": disagree,
        "_n": {
            "leaf_acc_unit": c["n_units"],
            "group_acc_unit": c["n_units"],
            "verdict_acc_unit": c["n_units"],
            "leaf_acc_char": c["n_chars"],
            "group_acc_char": c["n_chars"],
            "real_group_acc_unit": c["n_real_units"],
            "sibling_vs_category": c["n_sub_both"],
        },
    }
    for row in RECALL_ROWS:
        out[f"recall_{row}"] = _ratio(gold_hit[f"row_{row}"], gold_n[f"row_{row}"])
        out["_n"][f"recall_{row}"] = gold_n[f"row_{row}"]
    return out


def _mean(xs: Sequence[float | None]) -> float | None:
    v = [x for x in xs if x is not None]
    return sum(v) / len(v) if v else None


SCALARS: tuple[str, ...] = (
    "leaf_acc_unit",
    "group_acc_unit",
    "verdict_acc_unit",
    "leaf_acc_char",
    "group_acc_char",
    "claim_f1_char",
    "real_group_acc_unit",
    "sibling_vs_category",
    "unlocated_subspan_rate",
    *(f"recall_{r}" for r in RECALL_ROWS),
)


def score(
    cases: Sequence[dict[str, Any]],
    preds: Sequence[Sequence[tuple[list[cm.Segment], dict[str, int]]]],
    split: str,
    anchor: str = "none",
) -> dict[str, Any]:
    runs = [run_metrics(cases, p) for p in preds]
    mean: dict[str, Any] = {k: _mean([r[k] for r in runs]) for k in SCALARS}
    if len(runs) >= 2:
        a, b = runs[0]["_unit_groups"], runs[1]["_unit_groups"]
        n = len(a)
        mean["group_flip_rate"] = _ratio(sum(a[k] != b.get(k) for k in a), n)
    ns = dict(runs[0]["_n"])
    ns["group_flip_rate"] = runs[0]["n_units"]
    ci = {k: wilson(mean.get(k), ns[k]) for k in ns if mean.get(k) is not None}
    leaf_recall = {
        lf: {
            "recall": _mean([r["leaf_recall"][lf]["recall"] for r in runs]),
            "n": runs[0]["leaf_recall"][lf]["n"],
        }
        for lf in tx.LEAF_NAMES
    }
    out: dict[str, Any] = {
        "split": split,
        "prompt_version": cm.PROMPT_VERSION,
        "anchor": anchor,
        "taxonomy_version": tx.TAXONOMY_VERSION,
        "chosen_decisions": dict(tx.CHOSEN),
        "n_cases": len(cases),
        "n_units": runs[0]["n_units"],
        "n_gold_segments": runs[0]["n_gold_segments"],
        "n": ns,
        "mean": mean,
        "wilson95": ci,
        "leaf_recall": leaf_recall,
        "runs": [
            {k: v for k, v in r.items() if not k.startswith("_") and k != "probe_targets"}
            for r in runs
        ],
        "gate": gate(mean, ci, split),
    }
    out["gate_passed"] = all(g["passed"] is not False for g in out["gate"])
    if split != "heldout":
        out["probe_targets"] = [
            {"run": i, **d} for i, r in enumerate(runs) for d in r["probe_targets"]
        ]
    if split == "dev":
        out["disagreements"] = [
            {"run": i, **d} for i, r in enumerate(runs) for d in r["_disagreements"]
        ]
    return out


def gate(
    mean: dict[str, Any], ci: dict[str, list[float] | None], split: str = "heldout"
) -> list[dict[str, Any]]:
    """The gate rows; recall rows also need a Wilson lower bound >= 0.58. On dev a metric with
    no data is ``n/a`` (``passed = None``)."""
    rows = []
    for name, key, thr, hib in GATE:
        v = mean.get(key)
        ok: bool | None = v is not None and (v >= thr if hib else v <= thr)
        lo = None
        if key.startswith("recall_"):
            lo = (ci.get(key) or [None])[0]
            ok = bool(ok) and lo is not None and lo >= WILSON_LO
        if v is None and split != "heldout":
            ok = None
        rows.append(
            {
                "measure": name,
                "key": key,
                "value": v,
                "threshold": thr,
                "wilson_lo": lo,
                "passed": ok,
            }
        )
    return rows


def coverage(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Gold units and gold segments per leaf (the dev / held-out minimums)."""
    n_units = 0
    seg: Counter = Counter()
    n_est = n_disp = 0
    for c in cases:
        units, _ = case_units(c)
        for u, g in zip(units, c["gold"]["units"], strict=True):
            if g.get("leaf") is None and not g.get("splits"):
                continue
            n_units += 1
            n_est += u.established
            n_disp += bool(u.established and g.get("dispute"))
            for a, b, lf in gold_segments(u, g):
                if has_alnum(u.text[a - u.start : b - u.start]):
                    seg[lf] += 1
    return {
        "n_units": n_units,
        "segments_per_leaf": {lf: seg[lf] for lf in tx.LEAF_NAMES},
        "n_established_units": n_est,
        "n_disputed_established": n_disp,
    }


# ---------------------------------------------------------------- calls


def cal_key(split: str, case_id: str, r: int) -> str:
    return f"v2:{split}:{case_id}:r{r}"


async def map_cases(
    cases: Sequence[dict[str, Any]],
    *,
    split: str,
    model: str,
    effort: str | None,
    cache: Cache,
    llm: Any,
    repeats: int = 2,
    usage: Usage | None = None,
    anchor: str = "none",
) -> list[list[tuple[list[cm.Segment], dict[str, int]]]]:
    """preds[r][case_i] = (segments, parse counts) of repeat ``r`` (fail closed). Single-call
    mode (default) ignores the established flags; ``anchor="v5c"`` shows them."""
    schema = cm.schema_for(anchor)
    usage = usage if usage is not None else Usage()
    out: list[list[Any]] = [[None] * len(cases) for _ in range(repeats)]

    async def one(r: int, i: int, c: dict[str, Any]) -> None:
        units = mode_units(c, anchor)
        system, user = case_prompt(c, units, anchor)
        key = cal_key(split, c["case_id"], r)
        fp = fingerprint(cm.PROMPT_VERSION, model, effort, f"r{r}", system, user)
        row = cache.get(key, fp)
        raw: dict[str, Any] | None
        if row is not None:
            raw = row["result"]
        elif llm is None:
            raw = None
        else:
            call_usage = Usage()
            n_invalid = 0
            while True:  # plan rev 14: an invalid answer is retried up to 2 times in-run
                res = await llm.call(model=model, system=system, user=user, schema=schema)
                usage.add(res.usage)
                call_usage.add(res.usage)
                raw = res.result
                if raw is None or cm.validate(raw, units):
                    break
                n_invalid += 1
                if n_invalid > INVALID_RETRIES:
                    break
            meta = {"stage": "calibrate_v2", "usage": call_usage.to_dict()}
            if n_invalid:
                meta["n_invalid_attempts"] = n_invalid
            if raw is not None and not cm.validate(raw, units):
                cache.put(key, fp, {"result": None, "meta": {**meta, "raw": raw}})
                raw = None
            else:
                cache.put(key, fp, {"result": raw, "meta": meta})
        out[r][i] = cm.parse(raw, units)

    if llm is not None:
        await llm.preflight(model)
    await asyncio.gather(*(one(r, i, c) for r in range(repeats) for i, c in enumerate(cases)))
    return out


# ---------------------------------------------------------------- render + run


def render_markdown(res: dict[str, Any]) -> str:
    def f(v: Any) -> str:
        return "—" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))

    lines = [
        f"## Calibration v2: {res['split']} ({res['prompt_version']}, "
        f"{'single-call' if res.get('anchor', 'none') == 'none' else 'anchor ' + res['anchor']}"
        f"), {res['n_cases']} cases, "
        f"{res['n_units']} gold units, {res['n_gold_segments']} gold segments, "
        f"{len(res['runs'])} runs",
        "",
        "| measure | mean | Wilson 95% | threshold | pass |",
        "|---|---|---|---|---|",
    ]
    for g in res["gate"]:
        ci = res["wilson95"].get(g["key"])
        cis = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "—"
        mark = {True: "yes", False: "NO", None: "n/a"}[g["passed"]]
        lines.append(f"| {g['measure']} | {f(g['value'])} | {cis} | {g['threshold']} | {mark} |")
    m = res["mean"]
    lines += [
        "",
        f"leaf accuracy, character level: {f(m.get('leaf_acc_char'))}; forced / disputed "
        f"established per run: {[r['n_forced_established'] for r in res['runs']]} / "
        f"{[r['n_disputed_established'] for r in res['runs']]}; missing units per run: "
        f"{[r['n_missing_units'] for r in res['runs']]}; probe candidates / missing probe_target "
        f"per run: {[r['n_probe_candidates'] for r in res['runs']]} / "
        f"{[r['n_missing_probe_targets'] for r in res['runs']]}",
        "",
        "| leaf | gold segments | recall (mean) |",
        "|---|---|---|",
    ]
    for lf, v in res["leaf_recall"].items():
        lines.append(f"| {lf} | {v['n']} | {f(v['recall'])} |")
    if res.get("probe_targets"):
        lines += ["", "### Probe-target anchors (gold vs judge; informational)", ""]
        for d in res["probe_targets"]:
            lines.append(
                f"- r{d['run']} {d['case_id']} {d['unit']}: gold {d['gold_leaf']} "
                f"{json.dumps(d['gold'], ensure_ascii=False)} / pred {d['pred_leaf']} "
                f"{json.dumps(d['pred'], ensure_ascii=False)}"
            )
    if "disagreements" in res:
        lines += ["", f"### Disagreements, segment level ({len(res['disagreements'])})", ""]
        for d in res["disagreements"]:
            lines.append(
                f"- r{d['run']} {d['case_id']} {d['unit']} "
                f"{json.dumps(d['text'][:160], ensure_ascii=False)}: gold {d['gold']} / pred "
                f"{d['pred']} — why: {d['why']}"
            )
    return "\n".join(lines)


def run(
    split: str,
    *,
    model: str,
    effort: str | None,
    out_root: Path,
    repeats: int = 2,
    rpm: float = 1500.0,
    concurrency: int = 64,
    cases_file: Path | None = None,
    anchor: str = "none",
) -> dict[str, Any]:
    from hallucination_bench.evals.wsbench_chat.run import model_slug

    cm.check_anchor(anchor)
    cases = load_cases(split, cases_file)
    d = Path(out_root) / "calibration" / model_slug(model)
    usage = Usage()

    async def go() -> list[list[tuple[list[cm.Segment], dict[str, int]]]]:
        llm = LLM(rpm=rpm, concurrency=concurrency, effort=effort)
        try:
            with Cache(d / "cache.jsonl") as cache:
                return await map_cases(
                    cases,
                    split=split,
                    model=model,
                    effort=effort,
                    cache=cache,
                    llm=llm,
                    repeats=repeats,
                    usage=usage,
                    anchor=anchor,
                )
        finally:
            await llm.aclose()

    preds = asyncio.run(go())
    res = score(cases, preds, split, anchor)
    res["coverage"] = coverage(cases)
    res["spend_this_run"] = usage.to_dict()
    res["model"] = model
    res["effort"] = effort
    stem = result_stem(split, anchor)
    (d / f"{stem}.json").write_text(json.dumps(res, indent=1) + "\n")
    (d / f"{stem}.md").write_text(render_markdown(res) + "\n")
    return res


def result_stem(split: str, anchor: str = "none") -> str:
    return f"{split}_{cm.PROMPT_VERSION}" + ("" if anchor == "none" else f"_{anchor}")


def gate_passed(out_root: Path, model: str, anchor: str = "none") -> bool:
    """Whether a held-out run of the current prompt version (and mode) passed the gate. Every
    claim-map output made before that is labelled pre-gate."""
    from hallucination_bench.evals.wsbench_chat.run import model_slug

    p = (
        Path(out_root)
        / "calibration"
        / model_slug(model)
        / f"{result_stem('heldout', anchor)}.json"
    )
    try:
        return bool(json.loads(p.read_text()).get("gate_passed"))
    except (OSError, ValueError):
        return False
