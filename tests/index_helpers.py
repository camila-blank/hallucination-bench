"""Fixture builders shared by the index test files (plan 0030: moved unchanged from
``test_index_v6.py`` / ``test_index_v7.py``)."""

import json
from pathlib import Path
from typing import Any

from hallucination_bench import index as ix
from hallucination_bench.evals.benign_roleplay import metrics

# ---------------------------------------------------------------- overrefusal roots


def _v6_root(tmp: Path, rows: list[dict[str, Any]], unit: str | None = "refusal_claims") -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / ix.SAFETY_COUNTS).write_text("".join(json.dumps(r) + "\n" for r in rows))
    if unit is not None:
        (tmp / ix.SAFETY_META).write_text(json.dumps({"unit": unit, "gates": {}}))
    return tmp


def _or_row(layer: int, item: str, n: int, off: int, failed: int = 0) -> dict[str, Any]:
    return {
        "arm": "jlens",
        "layer": layer,
        "item": item,
        "cluster": item,
        **dict.fromkeys(ix.SAFETY_INT_COLS, 0),
        "n_claims": n,
        "n_false": off,
        "n_contradicted": off,
        "n_failed_cells": failed,
        "population": "never_refused",
    }


# ---------------------------------------------------------------- benign_roleplay summaries

INST = {
    "prompt_version": "jb-v3+digest-v5-task-extractive+sj-v10",
    "judge_model": "claude-sonnet-5",
}


def _items(n: int = 12, n_hb: int = 4, templates: int = 5) -> dict[str, Any]:
    items = [
        {"id": f"s{k}", "hb_only": False, "template_id": f"t{k % templates}"}
        for k in range(n - n_hb)
    ]
    items += [{"id": f"h{k}", "hb_only": True, "template_id": None} for k in range(n_hb)]
    return {"items": items}


def _unit(items_doc: dict[str, Any], flagged: set[str], **kw: Any) -> dict[str, Any]:
    """A summary unit exactly as metrics.score_unit builds it (the real function)."""
    clusters = {it["id"]: str(it.get("template_id") or it["id"]) for it in items_doc["items"]}
    ids = [it["id"] for it in items_doc["items"]]
    flags = {i: i in flagged for i in ids}
    rate, ci = metrics.rate_ci(flags, clusters)
    u = {
        "false_recognition_rate": rate,
        "ci95": list(ci),
        "n": len(ids),
        "n_flagged": len(flagged),
        "n_unjudged": 0,
        "complete": True,
        "flagged_ids": sorted(flagged),
        "scored_ids": sorted(ids),
        "unjudged_ids": [],
    }
    return u | kw


def _summary(
    path: Path, items_doc: dict[str, Any], arms: dict[str, dict[int, set[str]]], **kw: Any
) -> Path:
    items = items_doc["items"]
    n_hb = sum(it["hb_only"] for it in items)
    s = {
        "bank": {"n_items": len(items), "n_shared": len(items) - n_hb, "n_hb_only": n_hb},
        "accept_refused_as_flagged": False,
        "rates": {
            arm: {
                "instrument": dict(INST),
                "units": {
                    "all": _unit(items_doc, set().union(*by.values())),
                    **{f"L{L}": _unit(items_doc, fl) for L, fl in by.items()},
                },
            }
            for arm, by in arms.items()
        },
    } | kw
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(s))
    return path


ITEMS = _items()
