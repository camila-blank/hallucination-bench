"""Merge per-slice caches written by parallel `hbench run --out outputs_split/<tag>` processes
back into outputs/. Cache.get returns the newest successful row per (key, fingerprint), so a
plain union of rows is correct; exact duplicate lines are dropped.

    uv run python scripts/merge_split_caches.py [--split-root outputs_split] [--out outputs]
"""

import argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--split-root", default="outputs_split")
ap.add_argument("--out", default="outputs")
ap.add_argument("--model", default="claude-opus-5-5")
a = ap.parse_args()
dest_root = Path(a.out) / "wsbench_chat" / a.model
srcs: dict[str, list[Path]] = {}
for p in sorted(Path(a.split_root).glob(f"*/wsbench_chat/{a.model}/*/cache.jsonl")):
    srcs.setdefault(p.parent.name, []).append(p)
for arm, paths in sorted(srcs.items()):
    dest = dest_root / arm / "cache.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    seen, out = set(), []
    for p in ([dest] if dest.exists() else []) + paths:
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip() and line not in seen:
                seen.add(line)
                out.append(line)
    tmp = dest.with_suffix(".merge.tmp")
    tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
    tmp.replace(dest)
    print(f"{arm}: {len(paths)} slices -> {len(out)} rows")
