"""Plan 0028: copy the official J-lens interp-v1 ``summ:`` rows of the WSB cells into the x3
output root, so ``hbench run --stages summary`` calls the API for new cells only.

Rows are copied verbatim (key, fingerprint, result, meta); the run re-checks each fingerprint,
so a stale row is re-asked, never trusted. Only successful rows are copied.

    uv run --no-sync python scripts/main_x3/seed_summaries.py --bank <x3>/items.json \\
        --dest <out>/wsbench_chat/claude-opus-5-5/jlens/cache.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from hallucination_bench import paths  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--dest", type=Path, required=True)
    a = ap.parse_args()
    if a.dest.exists():
        raise SystemExit(f"{a.dest} exists: seed only a fresh cache")
    items = json.loads(a.bank.read_text())["items"]
    wsb = {(i["id"], int(s["pos"])) for i in items for s in i["sites"] if s["wsb"]}
    src = paths.MAIN_OUT / "jlens" / "cache.jsonl"
    keep: dict[str, str] = {}
    with src.open() as fh:
        for line in fh:
            if '"summ:' not in line[:40]:
                continue
            r = json.loads(line)
            k = r["key"]
            if not k.startswith("summ:") or r.get("result") is None:
                continue
            id_, _lay, pos = k[5:].split("__")
            if (id_, int(pos[1:])) in wsb:
                keep[k] = line if line.endswith("\n") else line + "\n"  # last ok row wins
    a.dest.parent.mkdir(parents=True, exist_ok=True)
    a.dest.write_text("".join(keep.values()))
    print(f"copied {len(keep)} summ rows from {src} -> {a.dest}")


if __name__ == "__main__":
    main()
