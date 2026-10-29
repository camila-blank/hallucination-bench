"""Writes the synthetic hindex-v3 golden fixture (plan 0019, Tests): claim cells + UM units under
``tests/fixtures/index_v3/``, then the index the code of the day builds from them
(``golden/index.json`` without ``git_sha`` / ``sources``, ``golden/index.md``).

Run once, before ``index.py`` became hindex-v4 (commit of record in the plan's results doc);
re-running it after that rewrites the golden with whatever the code then does, so don't."""

import json
import random
from typing import Any

from hallucination_bench import REPO_ROOT
from hallucination_bench import index as ix

FIX = REPO_ROOT / "tests" / "fixtures" / "index_v3"
LEAVES = [
    "false_claim",
    "substitution_category",
    "substitution_sibling",
    "detail",
    "correct",
    "model_state",
    "junk",
    "non_claim",
    "off_topic",
]


def _cells(rng: random.Random, n_items: int, layers: list[int], am: bool) -> list[dict[str, Any]]:
    out = []
    for L in layers:
        for i in range(n_items):
            for _ in range(rng.randint(1, 3)):
                segs = [
                    {"readout": rng.randint(0, 3), "leaf": rng.choice(LEAVES)}
                    for _ in range(rng.randint(0, 6))
                ]
                c: dict[str, Any] = {"id": f"i{i}", "layer": L, "map": {"segments": segs}}
                ok = rng.random() > 0.1
                if am:
                    c["map_status"] = "ok" if ok else "failed"
                else:
                    c["stages"] = {"summary": "ok", "map": "ok" if ok else "pending"}
                out.append(c)
    return out


def _units(rng: random.Random, methods: list[str], layers: list[int], dom: str) -> list[dict]:
    out = []
    for m in methods:
        for L in layers:
            for i in range(rng.randint(3, 8)):
                out.append(
                    {
                        "method": m,
                        "item": f"{dom}-{i % 4}",
                        "layer": L,
                        "committed": rng.random() > 0.15,
                        "probe": {
                            "na_reason": None if rng.random() > 0.1 else "invalid probe at L",
                            "fire_L": rng.random() > 0.5,
                        },
                        "cross": ["x"] if rng.random() > 0.6 else [],
                    }
                )
    return out


def main() -> None:
    rng = random.Random(19)
    layers = [20, 24, 28]
    arms = {
        "main": {"a": 6, "b": 5, "c": 7, "d": 4, "x": 3},
        "am": {"c": 5, "d": 6, "e": 4},
        "story": {"c": 4, "e": 5, "a2": 3},
    }
    for fam, spec in arms.items():
        for arm, n in spec.items():
            d = FIX / fam / arm
            d.mkdir(parents=True, exist_ok=True)
            ls = layers if arm != "d" else layers[:2]
            cells = _cells(rng, n, ls, am=fam == "am")
            (d / ix.CELLS).write_text("".join(json.dumps(c) + "\n" for c in cells))
    um_methods = {"country": ["c", "s"], "capital_city": ["c", "s", "d"], "holiday": ["c", "s"]}
    for dom, ms in um_methods.items():
        p = ix.um_paths(FIX / "um")[dom]
        p.parent.mkdir(parents=True, exist_ok=True)
        us = _units(rng, ms, layers, dom)
        p.write_text("".join(json.dumps(u) + "\n" for u in us))
    real = json.loads((REPO_ROOT / "data" / "index" / "methods.json").read_text())
    cfg = {
        "methods": {
            "M": {"main": ["a", "b"], "story": ["a2"], "sibling": ["s"]},
        },
        "exclude": ["x"],
        "notes": real["notes"],  # the v3 notes (frozen in the v3 code path)
    }
    (FIX / "methods.json").write_text(json.dumps(cfg, indent=1) + "\n")
    sources = {
        ix.MAIN: [FIX / "main"],
        ix.AM: [FIX / "am"],
        ix.STORY: [FIX / "story"],
        ix.SIBLING: [FIX / "um"],
    }
    built = ix.build_index(sources, ix.load_methods(FIX / "methods.json"))
    g = FIX / "golden"
    g.mkdir(exist_ok=True)
    keep = {k: v for k, v in built.items() if k != "sources"}
    (g / "index.json").write_text(json.dumps(keep, indent=1) + "\n")
    (g / "index.md").write_text(ix.render_md(built))
    print(f"{len(built['rows'])} rows")


if __name__ == "__main__":
    main()
