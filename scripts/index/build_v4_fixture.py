"""Writes the synthetic hindex-v4 golden fixture (plan 0021, Tests): a synthetic overrefusal root
and a v4 ``methods.json`` under ``tests/fixtures/index_v4/``, then the index the imported code
builds (``version="v4"``) from them plus the hindex-v3 synthetic claim cells / UM units
(``golden/index.json`` without ``git_sha`` / ``sources``, ``golden/index.md``).

Run once with the PRE-change code on the path (plan 0021: ``index.py`` at the merge base, before
hindex-v5), e.g.::

    git archive 77e2306 src | tar -x -C /tmp/hb-pre
    PYTHONPATH=/tmp/hb-pre/src python scripts/index/build_v4_fixture.py --repo .

Re-running it after that rewrites the golden with whatever the code then does, so don't."""

import argparse
import json
import random
from pathlib import Path
from typing import Any

from hallucination_bench import index as ix

# the hindex-v4 run's (#42) methods.json notes
NOTES = json.loads(
    (Path(__file__).resolve().parents[2] / "data" / "index" / "methods.json").read_text()
)["notes"]


def _row(rng: random.Random, arm: str, layer: int, item: str, **kw: int) -> dict[str, Any]:
    r: dict[str, Any] = {"arm": arm, "layer": layer, "item": item, "cluster": f"k{item[-1]}"}
    r |= dict.fromkeys(ix.SAFETY_INT_COLS, 0)
    if not kw:
        judged = rng.random() > 0.1
        false = judged and rng.random() < 0.4
        unprobed = false and rng.random() < 0.25
        kw = {
            "n_claims": int(judged),
            "n_false": int(false),
            "n_contradicted": int(false and not unprobed),
            "n_unprobed": int(unprobed),
            "n_unresolved": int(rng.random() < 0.1),
            "n_failed_cells": int(not judged),
        }
    return r | kw


def safety_rows() -> list[dict[str, Any]]:
    rng = random.Random(21)
    rows = []
    for arm, layers in (("jlens", (20, 24, 28)), ("s3d-rl600", (20, 24))):
        for L in layers:
            rows += [_row(rng, arm, L, f"o{i}") for i in range(12)]
    # s3d-rl600 @28: no claims; mlnla-rl600-all11 @20: every false claim unprobed; @24 one row
    rows += [_row(rng, "s3d-rl600", 28, f"o{i}", n_failed_cells=1) for i in range(3)]
    rows += [_row(rng, "mlnla-rl600-all11", 20, "o0", n_claims=1, n_false=1, n_unprobed=1)]
    rows += [_row(rng, "mlnla-rl600-all11", 24, "o0", n_claims=2, n_false=1, n_contradicted=1)]
    rows += [_row(rng, "textreader-x", 20, "o0", n_claims=1, n_false=1, n_contradicted=1)]
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--repo", type=Path, required=True, help="the checkout to write fixtures into")
    a = p.parse_args()
    fix = a.repo.resolve() / "tests" / "fixtures" / "index_v4"
    v3 = a.repo.resolve() / "tests" / "fixtures" / "index_v3"
    root = fix / "or"
    root.mkdir(parents=True, exist_ok=True)
    (root / ix.SAFETY_COUNTS).write_text("".join(json.dumps(r) + "\n" for r in safety_rows()))
    meta = {
        "gates": {"20": {"tail0": True, "content": False}, "24": {"tail0": True}},
        "population": "never_refused",
        "unit": "items",
    }
    (root / ix.SAFETY_META).write_text(json.dumps(meta, indent=1) + "\n")
    cfg = json.loads((v3 / "methods.json").read_text())
    cfg["methods"]["M"]["overrefusal"] = ["jlens"]
    cfg["methods"]["c"] = {"overrefusal": ["s3d-rl600"]}
    cfg["notes"] = NOTES
    (fix / "methods.json").write_text(json.dumps(cfg, indent=1) + "\n")
    sources = {
        ix.MAIN: [v3 / "main"],
        ix.AM: [v3 / "am"],
        ix.STORY: [v3 / "story"],
        ix.SIBLING: [v3 / "um"],
        ix.OVERREFUSAL: [root],
    }
    built = ix.build_index(sources, ix.load_methods(fix / "methods.json"), version="v4")
    assert built["version"] == "hindex-v4"
    g = fix / "golden"
    g.mkdir(exist_ok=True)
    keep = {k: v for k, v in built.items() if k not in ("git_sha", "sources")}
    (g / "index.json").write_text(json.dumps(keep, indent=1) + "\n")
    (g / "index.md").write_text(ix.render_md(built))
    print(f"wrote {fix} ({len(built['rows'])} rows)")


if __name__ == "__main__":
    main()
