"""Capture rows for the read positions some arm still lacks (plan 0016 rev 5).

Rev 5 reads every prompt position. The rev-1..4 readouts (thinned sites, same items, same
render, same layers) are reused as they are; this writes
``outputs/strongreject/capture_rows_rest.json``: the wsbench capture rows of ``items.json`` with
``eval_positions`` = the positions where at least one arm has no readout at some layer (the new
positions, plus the 650 cells the rev-4 NLA read left unread). ``merge_readouts.py`` then joins
the old and the new readouts per arm (old rows win on overlap).

    uv run --no-sync python scripts/strongreject/rest_rows.py [--arms s3d-rl600,mlnla-rl600,jlens]
"""

import argparse
import json
from pathlib import Path

from hallucination_bench import REPO_ROOT
from hallucination_bench.evals.strongreject import bank

LAYERS = (28, 36, 44, 52)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="s3d-rl600,mlnla-rl600,jlens")
    ap.add_argument("--readouts-root", default="/workspace/hb-readouts-strongreject")
    a = ap.parse_args()
    items = bank.load_items()
    have: dict[str, set[tuple[str, int, int]]] = {}
    for arm in a.arms.split(","):
        s = set()
        for line in open(Path(a.readouts_root) / arm / "strongreject.jsonl", encoding="utf-8"):
            r = json.loads(line)
            s.add((r["id"], r["layer"], r["pos"]))
        have[arm] = s
    rows, n = [], 0
    for it in items:
        need = sorted(
            p["pos"]
            for p in it["sites"]
            if any((it["id"], la, p["pos"]) not in h for h in have.values() for la in LAYERS)
        )
        if need:
            n += len(need)
            rows.append(
                {
                    "label": it["id"],
                    "family": "strongreject",
                    "render": "chat",
                    "user": it["jailbreak_prompt"],
                    "look_for": "",
                    "targets": [],
                    "eval_positions": need,
                }
            )
    dst = REPO_ROOT / "outputs/strongreject/capture_rows_rest.json"
    dst.write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {dst}: {len(rows)} items, {n} positions")


if __name__ == "__main__":
    main()
