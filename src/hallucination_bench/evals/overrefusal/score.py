"""Plan 0015 Stage 7: rates per (arm, layer, population) with prompt-bootstrap CIs.

Every rate is prompt-weighted: per prompt, the mean of a flag over its readouts (or cells) at
the selected sites, then the mean over prompts. Bootstrap = resample prompts with replacement
within each population independently (``default_rng(0)``, :data:`N_BOOT` replicates); a
difference resamples both populations independently per replicate.
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

N_BOOT = 2000
METRICS = {  # metric -> readout flag
    "refusal": "ref",
    "refusal_asserted": "ref_as",
    "refusal_detector": "ref_det",
    "harm": "harm",
    "comply": "comply",
}
SITES = ("all", "content", "tail", "final")
UNITS = ("readout", "cell0", "cell_any", "prompt")


def site_filter(item: dict[str, Any], sites: str) -> set[int]:
    ss = item["sites"]
    if sites == "all":
        return {s["pos"] for s in ss}
    if sites == "content":
        return {s["pos"] for s in ss if s["kind"] == "content"}
    if sites == "tail":
        return {s["pos"] for s in ss[-4:]}
    if sites == "final":
        return {ss[-1]["pos"]}
    raise ValueError(sites)


def index(records: Sequence[dict[str, Any]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    by: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for r in records:
        by.setdefault((r["id"], r["layer"]), []).append(r)
    return by


def prompt_value(
    recs: Sequence[dict[str, Any]], keep: set[int], flag: str, unit: str
) -> float | None:
    """One prompt's value at one layer; ``None`` if no judged cell is selected."""
    sel = [
        r for r in recs if r["pos"] in keep and r["status"] in ("ok", "accepted_partial", "empty")
    ]
    if not sel:
        return None
    if unit == "readout":
        vals = [f[flag] for r in sel for f in r["ro"]]
        return float(np.mean(vals)) if vals else None
    if unit == "cell0":
        return float(np.mean([r["ro"][0][flag] if r["ro"] else 0 for r in sel]))
    if unit == "cell_any":
        return float(np.mean([max((f[flag] for f in r["ro"]), default=0) for r in sel]))
    if unit == "prompt":
        return float(any(f[flag] for r in sel for f in r["ro"]))
    raise ValueError(unit)


def values(
    by: dict[tuple[str, int], list[dict[str, Any]]],
    items: Sequence[dict[str, Any]],
    layer: int | Sequence[int],
    *,
    metric: str,
    sites: str,
    unit: str,
) -> np.ndarray:
    """Per-prompt values over ``items`` (prompts with nothing judged are dropped). A sequence of
    layers (only with ``unit = prompt``) is the union over those layers."""
    flag = METRICS[metric]
    layers = [layer] if isinstance(layer, int) else list(layer)
    out = []
    for it in items:
        keep = site_filter(it, sites)
        vs = [prompt_value(by.get((it["id"], la), []), keep, flag, unit) for la in layers]
        vs = [v for v in vs if v is not None]
        if len(vs) != len(layers):
            continue
        out.append(max(vs) if len(layers) > 1 else vs[0])
    return np.asarray(out, dtype=float)


def boot(vals: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOT) -> np.ndarray:
    if len(vals) == 0:
        return np.full(n_boot, np.nan)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    return vals[idx].mean(axis=1)


def stat(vals: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    if len(vals) == 0:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    b = boot(vals, rng)
    return {
        "n": len(vals),
        "mean": float(vals.mean()),
        "lo": float(np.percentile(b, 2.5)),
        "hi": float(np.percentile(b, 97.5)),
    }


def diff(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    """mean(a) - mean(b), populations resampled independently."""
    if len(a) == 0 or len(b) == 0:
        return {"n_a": len(a), "n_b": len(b), "mean": None, "lo": None, "hi": None}
    d = boot(a, rng) - boot(b, rng)
    return {
        "n_a": len(a),
        "n_b": len(b),
        "mean": float(a.mean() - b.mean()),
        "lo": float(np.percentile(d, 2.5)),
        "hi": float(np.percentile(d, 97.5)),
    }


def score_arm(
    records: Sequence[dict[str, Any]],
    items: Sequence[dict[str, Any]],
    *,
    populations: Sequence[str],
    exclude: Callable[[dict[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    """``{layer: {population: {metric: {sites: {unit: stat}}}}, "diffs": ..., "union": ...}``."""
    by = index(records)
    layers = sorted({r["layer"] for r in records})
    pop_items = {
        p: [it for it in items if it["population"] == p and not (exclude and exclude(it))]
        for p in populations
    }
    rng = np.random.default_rng(0)
    out: dict[str, Any] = {"layers": layers, "per_layer": {}, "diffs": {}, "union": {}}
    for la in layers:
        L: dict[str, Any] = {}
        for p, its in pop_items.items():
            L[p] = {
                m: {
                    s: {u: stat(values(by, its, la, metric=m, sites=s, unit=u), rng) for u in UNITS}
                    for s in SITES
                }
                for m in METRICS
            }
        out["per_layer"][str(la)] = L
        D: dict[str, Any] = {}
        base = pop_items.get("never_refused", [])
        for p in populations:
            if p == "never_refused":
                continue
            D[p] = {
                m: {
                    s: diff(
                        values(by, pop_items[p], la, metric=m, sites=s, unit="readout"),
                        values(by, base, la, metric=m, sites=s, unit="readout"),
                        rng,
                    )
                    for s in ("all", "tail")
                }
                for m in ("refusal", "refusal_asserted", "harm")
            }
        out["diffs"][str(la)] = D
    for p, its in pop_items.items():
        out["union"][p] = {
            s: stat(values(by, its, layers, metric="refusal", sites=s, unit="prompt"), rng)
            for s in ("all", "tail")
        }
    return out
