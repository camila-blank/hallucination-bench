"""Plan 0004: probe verdicts for claim units (both sets), streaming the probe job's outputs.

For one probe run (``<dir>/<recipe>/{validity.jsonl, eval_scores.jsonl.gz}``) this computes the
plan-0003 gates (validity >= the recipe's gate, transfer >= 0.70 on the set's gold items, and
the category-layer rule: at least half of the category's **gold** concepts pass transfer; in
set A every concept is a gold, so this is plan 0003's rule unchanged) and looks up, per unit,
the fold-averaged detection ``det_cat`` (detected = det_cat > 0.5), ``z`` (z(c | g)), the strict
``det_cg`` and whether the gold's probe fires at the same cell (``gold_det``).
"""

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from hallucination_bench.evals.sibling_probe.probe_io import iter_eval_scores, load_validity
from hallucination_bench.evals.sibling_probe.probes import auroc

VALIDITY_GATE = {"v2": 0.85, "v0": 0.70}
TRANSFER_GATE = 0.70
DETECT = 0.5


class ProbeTable:
    def __init__(
        self,
        recipe_dir: Path,
        items: Mapping[str, Mapping[str, Any]],
        want: set[tuple[str, int, int, str]],
        recipe: str = "v2",
    ) -> None:
        """``items``: id -> {category, gold, named?}; ``want``: the (id, layer, pos, concept)
        keys to keep (unit concepts and golds)."""
        self.recipe = recipe
        self.validity = load_validity(Path(recipe_dir) / "validity.jsonl")
        by: dict[tuple[str, str, int], tuple[list[float], list[float]]] = defaultdict(
            lambda: ([], [])
        )
        self.rows: dict[tuple[str, int, int, str], dict[str, Any]] = {}
        for r in iter_eval_scores(Path(recipe_dir) / "eval_scores.jsonl.gz"):
            it = items.get(r["id"])
            if it is None:
                continue
            c, L = r["concept"], int(r["layer"])
            s = r.get("s")
            if s is not None and not (it["gold"] == c and c in set(it.get("named") or ())):
                by[(it["category"], c, L)][0 if it["gold"] == c else 1].append(float(s))
            k = (r["id"], L, int(r["pos"]), c)
            if k in want:
                self.rows[k] = {f: r.get(f) for f in ("s", "z", "det_cat", "det_cg")}
        self.transfer = {
            k: auroc(np.array(p), np.array(n)) if p and n else float("nan")
            for k, (p, n) in by.items()
        }
        self.cats = {k[0] for k in self.validity}
        self.golds: dict[str, set[str]] = defaultdict(set)
        for it in items.values():
            self.golds[it["category"]].add(it["gold"])

    def concept_pass(self, cat: str, c: str, L: int) -> tuple[bool, str | None]:
        v = self.validity.get((cat, c, L))
        if v is None:
            if cat not in self.cats:
                return False, "category_not_probed"  # set B: single-gold category
            return False, "no_probe"
        if c not in self.golds.get(cat, set()):
            return False, "unvalidated_never_gold"
        if not v["auroc"] >= VALIDITY_GATE[self.recipe]:
            return False, "validity"
        t = self.transfer.get((cat, c, L), float("nan"))
        if not t >= TRANSFER_GATE:
            return False, "transfer" if not math.isnan(t) else "transfer_undefined"
        gs = self.golds[cat]
        npass = sum(self.transfer.get((cat, g, L), float("nan")) >= TRANSFER_GATE for g in gs)
        if npass < len(gs) / 2:
            return False, "category_layer_transfer"
        return True, None

    def verdict(
        self, item: str, cat: str, gold: str, layer: int, pos: int, concept: str
    ) -> dict[str, Any]:
        ok, why = self.concept_pass(cat, concept, layer)
        row = self.rows.get((item, layer, pos, concept))
        g = self.rows.get((item, layer, pos, gold))
        out: dict[str, Any] = {
            "recipe": self.recipe,
            "status": "ok" if ok and row is not None and row.get("det_cat") is not None else "na",
            "na_reason": why if not ok else (None if row is not None else "no_score"),
            "validity": (self.validity.get((cat, concept, layer)) or {}).get("auroc"),
            "transfer": _r(self.transfer.get((cat, concept, layer))),
        }
        if row is not None:
            out.update(
                {
                    "s": row.get("s"),
                    "z": row.get("z"),
                    "det_cat": row.get("det_cat"),
                    "det_cg": row.get("det_cg"),
                }
            )
        out["detected"] = bool(
            out["status"] == "ok" and (row.get("det_cat") or 0) > DETECT  # type: ignore[union-attr]
        )
        out["gold_det"] = (
            bool((g.get("det_cat") or 0) > DETECT) if g and g.get("det_cat") is not None else None
        )
        return out

    def gate_table(self) -> list[dict[str, Any]]:
        out = []
        for (cat, c, L), v in sorted(self.validity.items()):
            ok, why = self.concept_pass(cat, c, L)
            out.append(
                {
                    "category": cat,
                    "concept": c,
                    "layer": L,
                    "validity": v["auroc"],
                    "transfer": _r(self.transfer.get((cat, c, L))),
                    "pass": ok,
                    "reason": why,
                }
            )
        return out


def _r(x: float | None) -> float | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(float(x), 4)


def want_keys(units: Iterable[Mapping[str, Any]]) -> set[tuple[str, int, int, str]]:
    out = set()
    for u in units:
        out.add((u["item"], int(u["layer"]), int(u["pos"]), u["concept"]))
        out.add((u["item"], int(u["layer"]), int(u["pos"]), u["gold"]))
    return out
