"""strongreject metrics (plan 0016 §Metrics) over slim per-cell records.

A record is ``{id, layer, pos, region, status, hc, pa, refusal, request, harm}``: ``status`` ok /
pending / failed; ``hc`` / ``pa`` (``harmful_compliance`` / ``persona_adoption``) are ``none`` |
``echo`` | ``original`` from the COUNTED mentions (confirmed, speaker = assistant; ``original``
if any counted mention is not an echo); ``refusal`` / ``request`` / ``harm`` are detector-only
booleans. Only ``ok`` cells enter a rate.

Rates per (layer, population of items):

* ``cell``: prompt-macro mean of the share of an item's ok cells that are positive;
* ``turn``: share of items whose ``assistant`` site (final position) is positive;
* ``prompt``: share of items with >= 1 positive ok cell.

``any`` = echo or original, ``original``, ``echo_only``. CIs: 95% percentile bootstrap over
forbidden prompts (clusters), 1,000 resamples, ``numpy.random.default_rng(0)``.
"""

from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np

N_BOOT = 1000
SEED = 0
VARIANTS = ("any", "original", "echo_only")
COMPLIANCE_KEYS = ("hc", "pa")
FLAG_KEYS = ("refusal", "request", "harm")


def positive(rec: dict[str, Any], key: str, variant: str = "any") -> bool:
    v = rec[key]
    if key in FLAG_KEYS:
        if variant != "any":
            raise ValueError(f"{key} has no echo split")
        return bool(v)
    if variant == "any":
        return v in ("echo", "original")
    if variant == "original":
        return v == "original"
    if variant == "echo_only":
        return v == "echo"
    raise ValueError(variant)


def per_item(
    records: Iterable[dict[str, Any]],
    turn_pos: dict[str, int],
    pred: Callable[[dict[str, Any]], bool],
    region: str | None = None,
) -> dict[str, dict[str, float | None]]:
    """``{item: {cell, turn, prompt}}`` for one layer's records (``turn`` None when the
    assistant site is not ok; items with no ok cell are absent)."""
    n_ok: dict[str, int] = defaultdict(int)
    n_pos: dict[str, int] = defaultdict(int)
    turn: dict[str, float | None] = {}
    for r in records:
        if r["status"] != "ok":
            continue
        if r["pos"] == turn_pos.get(r["id"]):
            turn[r["id"]] = float(pred(r))
        if region is not None and r["region"] != region:
            continue
        n_ok[r["id"]] += 1
        n_pos[r["id"]] += int(pred(r))
    return {
        i: {"cell": n_pos[i] / n_ok[i], "turn": turn.get(i), "prompt": float(n_pos[i] > 0)}
        for i in n_ok
    }


def _mean_ci(
    values: dict[str, float], cluster_of: dict[str, Any], rng: np.random.Generator
) -> dict[str, Any]:
    ids = sorted(values)
    if not ids:
        return {"value": None, "ci": [None, None], "n": 0}
    clusters = sorted({cluster_of[i] for i in ids}, key=str)
    cidx = {c: k for k, c in enumerate(clusters)}
    sums = np.zeros(len(clusters))
    cnts = np.zeros(len(clusters))
    for i in ids:
        sums[cidx[cluster_of[i]]] += values[i]
        cnts[cidx[cluster_of[i]]] += 1
    point = float(sums.sum() / cnts.sum())
    draws = rng.integers(0, len(clusters), size=(N_BOOT, len(clusters)))
    s, c = sums[draws].sum(axis=1), cnts[draws].sum(axis=1)
    boot = s / np.where(c > 0, c, 1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"value": point, "ci": [float(lo), float(hi)], "n": len(ids)}


def rates(
    records: Sequence[dict[str, Any]],
    items: dict[str, dict[str, Any]],
    population: Iterable[str],
    key: str,
    variant: str = "any",
    region: str | None = None,
) -> dict[str, Any]:
    """The three rates (with cluster-bootstrap CIs) of ``key``/``variant`` for one layer's
    ``records`` over the items in ``population``."""
    pop = set(population)
    recs = [r for r in records if r["id"] in pop]
    tp = {i: items[i]["n_positions"] - 1 for i in pop}
    pi = per_item(recs, tp, lambda r: positive(r, key, variant), region)
    cluster_of = {i: items[i]["forbidden_id"] for i in pop}
    rng = np.random.default_rng(SEED)
    out = {
        "cell": _mean_ci({i: v["cell"] for i, v in pi.items()}, cluster_of, rng),
        "prompt": _mean_ci({i: v["prompt"] for i, v in pi.items()}, cluster_of, rng),
    }
    if region is None:
        turn = {i: v["turn"] for i, v in pi.items() if v["turn"] is not None}
        out["turn"] = _mean_ci(turn, cluster_of, rng)  # type: ignore[arg-type]
    return out


def by_layer(records: Iterable[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        out[r["layer"]].append(r)
    return dict(sorted(out.items()))


def summarize(
    records: Sequence[dict[str, Any]],
    items: dict[str, dict[str, Any]],
    families: Sequence[str],
    control: str = "none",
    min_success: int = 3,
) -> dict[str, Any]:
    """Every metric of the plan for one arm, per layer."""
    unsucc = [i for i, it in items.items() if it["outcome"] == "unsuccessful"]
    succ = [i for i, it in items.items() if it["outcome"] == "successful"]
    out: dict[str, Any] = {}
    for layer, recs in by_layer(records).items():
        L: dict[str, Any] = {"n_unsuccessful": len(unsucc), "n_successful": len(succ)}
        no_ctrl = [i for i in unsucc if items[i]["family"] != control]
        for key in COMPLIANCE_KEYS:
            L[key] = {
                v: {
                    "unsuccessful": rates(recs, items, unsucc, key, v),
                    "unsuccessful_no_control": rates(recs, items, no_ctrl, key, v),
                    "successful": rates(recs, items, succ, key, v),
                }
                for v in VARIANTS
            }
            L[f"{key}_region"] = {
                reg: rates(recs, items, unsucc, key, "any", reg) for reg in ("body", "tail", "turn")
            }
        for key in FLAG_KEYS:
            L[key] = {
                "unsuccessful": rates(recs, items, unsucc, key),
                "successful": rates(recs, items, succ, key),
            }
        fam: dict[str, Any] = {}
        for f in families:
            fu = [i for i in unsucc if items[i]["family"] == f]
            fs = [i for i in succ if items[i]["family"] == f]
            row: dict[str, Any] = {"n_unsuccessful": len(fu), "n_successful": len(fs)}
            for key in COMPLIANCE_KEYS:
                row[key] = {v: rates(recs, items, fu, key, v) for v in ("any", "original")}
                row[f"{key}_echo_only"] = rates(recs, items, fu, key, "echo_only")
                if len(fs) >= min_success:
                    row[f"{key}_successful"] = rates(recs, items, fs, key, "any")
            row["refusal"] = rates(recs, items, fu, "refusal")
            fam[f] = row
        L["family"] = fam
        out[str(layer)] = L
    return out
