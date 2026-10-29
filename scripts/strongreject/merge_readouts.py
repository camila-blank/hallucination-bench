"""Merge an arm's rev-5 rest readouts into its readouts file (plan 0016 rev 5).

Old rows (the rev-1..4 reads) are kept byte for byte; a rest row is added only for an (id,
layer, pos) the old file lacks. The old file is kept as ``strongreject.rev4.jsonl`` the first
time. Then the full grid check: every items.json site x layer exactly once.

    uv run --no-sync python scripts/strongreject/merge_readouts.py <arm> [--rest-root DIR]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from hallucination_bench.evals.strongreject import bank

LAYERS = (28, 36, 44, 52)
L11 = (20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60)  # plan 0019


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--root", default="/workspace/hb-readouts-strongreject")
    ap.add_argument("--rest-root", default="/workspace/hb-readouts-strongreject-rest")
    ap.add_argument(
        "--layers",
        default=",".join(map(str, LAYERS)),
        help="the grid checked after the merge (plan 0019: all 11)",
    )
    ap.add_argument(
        "--backup",
        default="strongreject.rev4.jsonl",
        help="the pre-merge copy of the file, kept beside it (created once; plan 0019 uses "
        "strongreject.rev5.jsonl, the merged 4-layer file)",
    )
    a = ap.parse_args()
    dst = Path(a.root) / a.arm / "strongreject.jsonl"
    old = dst.with_name(a.backup)
    if not old.exists():
        shutil.copyfile(dst, old)
    seen: set[tuple[str, int, int]] = set()
    tmp = dst.with_suffix(".tmp")
    n_old = n_new = n_dup = n_escaped = 0
    with tmp.open("w", encoding="utf-8") as out:
        for line in old.open(encoding="utf-8"):
            r = json.loads(line)
            seen.add((r["id"], r["layer"], r["pos"]))
            out.write(line if line.endswith("\n") else line + "\n")
            n_old += 1
        rest = Path(a.rest_root) / a.arm / "strongreject.jsonl"
        for line in rest.open(encoding="utf-8", newline="\n"):
            r = json.loads(line)
            k = (r["id"], r["layer"], r["pos"])
            if k in seen:
                n_dup += 1
                continue
            seen.add(k)
            body = line.rstrip("\n")
            if len(body.splitlines()) > 1:
                # a raw U+2028 / \x1c-style separator inside a string: str.splitlines (which
                # the readouts loader uses) would cut the row, so re-encode this row ASCII-escaped
                body = json.dumps(r, ensure_ascii=True)
                n_escaped += 1
            out.write(body + "\n")
            n_new += 1
    tmp.replace(dst)
    layers = [int(x) for x in a.layers.split(",")]
    want = {
        (it["id"], la, s["pos"]) for it in bank.load_items() for la in layers for s in it["sites"]
    }
    miss, off = want - seen, seen - want
    print(
        f"{a.arm}: old {n_old} + new {n_new} (rest rows already present: {n_dup}; "
        f"re-escaped {n_escaped}); "
        f"grid {len(seen & want)}/{len(want)}, missing {len(miss)}, off-grid {len(off)}"
    )
    sys.exit(1 if miss or off else 0)


if __name__ == "__main__":
    main()
