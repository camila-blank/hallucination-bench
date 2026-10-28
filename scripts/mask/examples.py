"""Print verified / rejected lie_claim examples from an arm's cells.jsonl with their readout
context (plan 0017 hand-reading). Streams cells.jsonl and the readouts file.

    uv run --no-sync python scripts/mask/examples.py <arm> [--label honest|lie] [--real 1|0]
"""

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from hallucination_bench.evals.mask import bank  # noqa: E402
from hallucination_bench.evals.mask.detect import readouts_of  # noqa: E402
from hallucination_bench.readouts import _parse_row  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--label", default="honest")
    ap.add_argument("--real", type=int, default=1)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/mask/claude-opus-5-5")
    a = ap.parse_args()
    picks = []
    with (REPO / a.out / a.arm / "cells.jsonl").open() as fh:
        for line in fh:
            c = json.loads(line)
            if not c["label"].startswith(a.label):
                continue
            for m in c["mentions"]:
                v = m.get("verify")
                if m["category"] == "lie_claim" and v and bool(v["real"]) == bool(a.real):
                    picks.append((c, m))
    random.Random(a.seed).shuffle(picks)
    picks = picks[: a.n]
    want = {(c["task_id"], c["layer"], c["pos"]) for c, _ in picks}
    ro: dict = {}
    with (bank.READOUTS_ROOT / a.arm / bank.READOUTS_FILE).open() as fh:
        for line in fh:
            cell = _parse_row(json.loads(line))
            if cell and (cell.id, cell.layer, cell.pos) in want:
                ro[(cell.id, cell.layer, cell.pos)] = readouts_of(cell)
    items = {it["task_id"]: it for it in bank.load_items()}
    print(f"{len(picks)} shown")
    for c, m in picks:
        it = items[c["task_id"]]
        text = ro[(c["task_id"], c["layer"], c["pos"])][m["readout"]]
        s, e = m["start"], m["end"]
        ctx = text[max(0, s - 160) : e + 80].replace("\n", " / ")
        print(
            f"--- {c['task_id']} {it['archetype']} {c['label']}/{c['stratum']} L{c['layer']} "
            f"{c['site_type']} p{c['pos']} actor={m['actor']} blind={m.get('verify_blind')}"
        )
        print(f"  prop: {it['option_a']} | {it['option_b']}")
        print(f"  quote: {m['quote']!r}")
        print(f"  ctx: ...{ctx}...")
        print(f"  verifier: {m['verify']['reason']}")


if __name__ == "__main__":
    main()
