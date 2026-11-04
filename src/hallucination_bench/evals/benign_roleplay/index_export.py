"""Plan 0027: the eval's ``summary.json`` -> a hindex-v7 ``benign_roleplay`` root.

One ``index_counts.jsonl`` row per (arm, layer, scored item): ``n_claims = 1``, ``n_false =
n_contradicted`` = 1 if flagged (a verified recognition, or unjudged; the eval's rule) else 0,
``n_failed_cells`` = 1 if unjudged, ``cluster = str(template_id or id)`` (the eval's cluster
rule). ``index.safety_metric`` on these rows is the same arithmetic as ``metrics.rate_ci`` (sorted
clusters, a fresh ``default_rng(0)``, 2,000 draws), so the index value and CI equal the eval's
per-layer ``false_recognition_rate`` / ``ci95``. Only single-layer units (``L<k>``) go to
``index_counts.jsonl``; plan 0029's ``--joint`` writes the all-layer unit (``all``) to
``index_counts_joint.jsonl`` (layer ``"all"``; hindex-v8's joint row). Nothing is judged here."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .wsb import WsbError

UNIT = "prompts"  # == hallucination_bench.index.UNIT_PROMPTS
COUNTS, META = "index_counts.jsonl", "index_meta.json"
COUNTS_JOINT = "index_counts_joint.jsonl"  # plan 0029
JOINT_UNIT = JOINT_LAYER = "all"
INT_COLS = (
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


def _readers() -> frozenset[str]:
    from hallucination_bench import index

    return index.SAFETY_READERS_V7_ADDED[index.BENIGN_ROLEPLAY]


def unit_rows(
    arm: str,
    layer: int | str,
    u: Mapping[str, Any],
    ids_all: Sequence[str],
    clusters: Mapping[str, str],
) -> list[dict[str, Any]]:
    """The rows of one (arm, ``L<k>``) unit of a summary (plan 0029: or the ``all`` unit)."""
    where = f"{arm} L{layer}" if layer != JOINT_LAYER else f"{arm} {JOINT_UNIT}"
    if "scored_ids" in u:
        scored = list(u["scored_ids"])
    elif u["n"] == len(ids_all):  # a summary from before plan 0027 (no scored_ids)
        scored = list(ids_all)
    else:
        raise WsbError(
            f"{where}: n = {u['n']} of {len(ids_all)} bank items and no scored_ids (re-run "
            "`hbench benign-roleplay report` to record them)"
        )
    if len(scored) != u["n"] or len(set(scored)) != len(scored):
        raise WsbError(f"{where}: scored_ids ({len(scored)}) disagree with n = {u['n']}")
    if unknown := set(scored) - set(ids_all):
        raise WsbError(f"{where}: ids not in the bank: {sorted(unknown)[:5]}")
    flagged = set(u["flagged_ids"])
    if not flagged <= set(scored):
        raise WsbError(f"{where}: flagged_ids not within scored_ids")
    if len(flagged) != u["n_flagged"]:
        raise WsbError(f"{where}: {len(flagged)} flagged_ids but n_flagged = {u['n_flagged']}")
    unjudged = set(u.get("unjudged_ids", ()))
    rows = []
    for i in sorted(scored):
        f = int(i in flagged)
        r = dict.fromkeys(INT_COLS, 0)
        r |= {
            "n_claims": 1,
            "n_false": f,
            "n_contradicted": f,
            "n_failed_cells": int(i in unjudged),
        }
        rows.append({"arm": arm, "layer": layer, "item": i, "cluster": clusters[i], **r})
    return rows


def export(
    summaries: Sequence[Path],
    dest: Path,
    items_doc: Mapping[str, Any],
    *,
    arm_map: Mapping[str, str] | None = None,
    joint: bool = False,
) -> dict[str, Any]:
    """Write ``dest/index_counts.jsonl`` + ``dest/index_meta.json``; exit 2 (``WsbError``) on a
    bank mismatch, mixed instruments, an incomplete unit (unless the summary admitted refused
    cells as flagged), an ``arm_map`` target outside the allowlist or key in no summary, or an arm
    in two summaries. Arms that are not index readers (e.g. plan 0023's ``mlnla-rl2-2000``) are
    skipped and listed under ``skipped_arms``, as the index lists ``ignored_arms``."""
    arm_map = dict(arm_map or {})
    readers = _readers()
    if bad := {t for t in arm_map.values() if t not in readers}:
        raise WsbError(f"--arm-map targets {sorted(bad)} are not index readers {sorted(readers)}")
    items = items_doc["items"]
    ids_all = [it["id"] for it in items]
    clusters = {it["id"]: str(it.get("template_id") or it["id"]) for it in items}
    n_shared = sum(not it["hb_only"] for it in items)
    rows: list[dict[str, Any]] = []
    joint_rows: list[dict[str, Any]] = []
    instruments: dict[str, Any] = {}
    sources = []
    skipped: list[str] = []
    loaded = [(sp, Path(sp).read_bytes()) for sp in summaries]
    seen_arms = {arm for _, raw in loaded for arm in json.loads(raw)["rates"]}
    if unknown := set(arm_map) - seen_arms:
        raise WsbError(f"--arm-map keys {sorted(unknown)} are not arms of any summary")
    for sp, raw in loaded:
        s = json.loads(raw)
        b = s["bank"]
        if (b["n_items"], b["n_shared"], b["n_hb_only"]) != (
            len(ids_all),
            n_shared,
            len(ids_all) - n_shared,
        ):
            raise WsbError(f"{sp}: bank counts {b} differ from the HB bank of record")
        sources.append(
            {"summary": str(Path(sp).resolve()), "sha256": hashlib.sha256(raw).hexdigest()}
        )
        for arm, a in sorted(s["rates"].items()):
            name = arm_map.get(arm, arm)
            if name not in readers:
                skipped.append(arm)
                continue
            if name in instruments:
                raise WsbError(f"arm {name!r} appears in two summaries")
            instruments[name] = a["instrument"]
            for unit, u in sorted(a["units"].items()):
                if unit == JOINT_UNIT and not joint:
                    continue
                if not u["complete"] and not s.get("accept_refused_as_flagged"):
                    raise WsbError(f"{sp}: {arm} {unit} is incomplete ({u['n_unjudged']} unjudged)")
                if unit == JOINT_UNIT:
                    joint_rows += unit_rows(name, JOINT_LAYER, u, ids_all, clusters)
                    continue
                rows += unit_rows(name, int(unit[1:]), u, ids_all, clusters)
    distinct = {json.dumps(v, sort_keys=True) for v in instruments.values()}
    if len(distinct) > 1:
        raise WsbError(f"arms judged on different instruments: {instruments}")
    meta = {
        "unit": UNIT,
        "eval": "benign_roleplay",
        "plan": "0027",
        "summaries": sources,
        "instrument": next(iter(instruments.values()), None),
        "instrument_by_arm": instruments,
        "arm_map": arm_map,
        "arms": sorted(instruments),
        "skipped_arms": sorted(skipped),
        "layers": sorted({r["layer"] for r in rows}),
        "n_rows": len(rows),
        "gates": {},
    }
    if joint:  # plan 0029: only a --joint export records it (v7 roots stay byte-identical)
        meta["joint"] = {"file": COUNTS_JOINT, "n_rows": len(joint_rows)}
    dest.mkdir(parents=True, exist_ok=True)
    with open(dest / COUNTS, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    if joint:
        with open(dest / COUNTS_JOINT, "w", encoding="utf-8") as fh:
            for r in joint_rows:
                fh.write(json.dumps(r) + "\n")
    (dest / META).write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    return meta
