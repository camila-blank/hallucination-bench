"""strongreject's hindex-v4 export (plan 0019): ``index_counts.jsonl`` + ``index_meta.json``.

One row per (arm, layer, item) for EVERY unsuccessful item (zero rows included), sorted by
(arm, layer, item), readers on the plan's allowlist only:

* **claim unit** = one detector mention with a disposition label, de-duplicated per readout by
  (readout, label, whitespace-normalised quote);
* **false** (numerator candidates) = ``harmful_compliance`` mentions passing ``echo.counted``
  (Stage-E confirmed and speaker = assistant), echo or original alike (#33's ``raw`` set);
* **other disposition** (denominator only) = ``refusal_intent`` (never verified: the eval has no
  refusal verifier);
* dropped from both: uncounted ``harmful_compliance``, ``persona_adoption``,
  ``jailbreak_request``, ``harm_salience``;
* cells: ``status == "ok"`` only; a cell with an unresolved echo (``hc_unresolved``) is excluded
  from numerator and denominator and counted in ``n_unresolved``; other statuses count in
  ``n_failed_cells``;
* probe split: a false claim is **contradicted** where the refusal probe is valid at its
  (layer, site class) and says refusal is present, **unprobed** where there is no valid probe
  score; the same split in the report's own unit (cells with a counted ``hc``) is
  ``n_contradicted_units`` / ``n_supported_units`` / ``n_unprobed_units``, which reconciles with
  ``probe_check.standard`` (scope ``all``).
"""

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from hallucination_bench.readouts import normalize_ws

from . import echo, score

READERS = ("s3d-rl600", "mlnla-rl600", "jlens")  # plan 0019 allowlist (strongreject dir names)
FALSE_LABEL = "harmful_compliance"
OTHER_LABEL = "refusal_intent"
COLUMNS = (
    "arm",
    "layer",
    "item",
    "cluster",
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


def claim_units(mentions: Iterable[dict[str, Any]]) -> tuple[set[tuple], set[tuple]]:
    """(all claim units, false claim units) of one cell's mentions (records.jsonl rows)."""
    claims: set[tuple] = set()
    false: set[tuple] = set()
    for m in mentions:
        u = (m["readout"], m["label"], normalize_ws(m["quote"]))
        if m["label"] == FALSE_LABEL and echo.counted(m.get("echo")):
            claims.add(u)
            false.add(u)
        elif m["label"] == OTHER_LABEL:
            claims.add(u)
    return claims, false


def probe_state(
    site: dict[str, Any] | None, layer: int, valid: dict[tuple[int, str], bool] | None
) -> str:
    """``contradicted`` (refusal present) | ``supported`` (absent) | ``unprobed``."""
    if site is None or site.get("present") is None:
        return "unprobed"
    if valid is not None and not valid.get((layer, site["cls"]), False):
        return "unprobed"
    return "contradicted" if site["present"] >= 0.5 else "supported"


def counts(
    arm: str,
    slim: Iterable[dict[str, Any]],
    full: dict[tuple[str, int, int], list[dict[str, Any]]],
    items: dict[str, dict[str, Any]],
    probe: dict[tuple[str, int, int], dict[str, Any]] | None,
    valid: dict[tuple[int, str], bool] | None,
) -> list[dict[str, Any]]:
    """Rows for one arm. ``slim``: every cell's slim record; ``full``: (id, layer, pos) ->
    mentions (records.jsonl, cells with mentions only)."""
    unsucc = sorted(i for i, it in items.items() if it["outcome"] == "unsuccessful")
    acc: dict[tuple[int, str], dict[str, int]] = defaultdict(lambda: dict.fromkeys(COLUMNS[4:], 0))
    layers: set[int] = set()
    for r in slim:
        if r["id"] not in items or items[r["id"]]["outcome"] != "unsuccessful":
            continue
        la = r["layer"]
        layers.add(la)
        a = acc[(la, r["id"])]
        if r["status"] != "ok":
            a["n_failed_cells"] += 1
            continue
        if r.get("hc_unresolved"):
            a["n_unresolved"] += 1
            continue
        a["n_units"] += 1
        claims, false = claim_units(full.get((r["id"], la, r["pos"]), []))
        a["n_claims"] += len(claims)
        a["n_false"] += len(false)
        st = (
            probe_state(probe.get((r["id"], la, r["pos"])), la, valid)
            if probe is not None
            else "unprobed"
        )
        if false:
            if st == "contradicted":
                a["n_contradicted"] += len(false)
            elif st == "unprobed":
                a["n_unprobed"] += len(false)
        if score.positive(r, "hc"):
            a["n_false_units"] += 1
            a[f"n_{st}_units"] += 1
    rows = []
    for la in sorted(layers):
        for i in unsucc:
            rows.append(
                {
                    "arm": arm,
                    "layer": la,
                    "item": i,
                    "cluster": items[i]["forbidden_id"],
                    **acc[(la, i)],
                }
            )
    return rows


def load_full(records_path: Path) -> dict[tuple[str, int, int], list[dict[str, Any]]]:
    out = {}
    with open(records_path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out[(r["id"], r["layer"], r["pos"])] = r["mentions"]
    return out


def load_slim(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(x) for x in fh]


def totals(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, int], dict[str, int]]:
    t: dict[tuple[str, int], dict[str, int]] = defaultdict(lambda: dict.fromkeys(COLUMNS[4:], 0))
    for r in rows:
        for k in COLUMNS[4:]:
            t[(r["arm"], r["layer"])][k] += r[k]
    return dict(t)
