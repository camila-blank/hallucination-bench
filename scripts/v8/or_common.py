"""Overrefusal on the common item set (plan 0029 Amendment 4 reporting).

Per unit, the off-topic share over the items failed under no method, beside the headline (all
judged items). usage: python scripts/v8/or_common.py <offtopic_index dir>
"""

import json
import sys
from collections import defaultdict

R = sys.argv[1]  # offtopic_index dir
for f in ("index_counts.jsonl", "index_counts_joint.jsonl"):
    rows = [json.loads(line) for line in open(f"{R}/{f}", encoding="utf-8")]
    bad = defaultdict(set)
    for r in rows:
        if r["n_failed_cells"]:
            bad[r["layer"]].add(r["item"])
    agg = defaultdict(lambda: [0, 0, 0, 0, 0])  # off, decl (all) off, decl (common), n_failed
    for r in rows:
        a = agg[(r["arm"], r["layer"])]
        a[0] += r["n_contradicted"]
        a[1] += r["n_claims"]
        a[4] += r["n_failed_cells"]
        if r["item"] not in bad[r["layer"]]:
            a[2] += r["n_contradicted"]
            a[3] += r["n_claims"]
    print(f"## {f}")
    print("| arm | layer | failed items | headline % | common-set % | common items excluded |")
    print("|---|---|---:|---:|---:|---:|")
    for (arm, la), a in sorted(agg.items(), key=lambda x: (x[0][0], str(x[0][1]).zfill(3))):
        h = 100 * a[0] / a[1] if a[1] else float("nan")
        c = 100 * a[2] / a[3] if a[3] else float("nan")
        print(f"| {arm} | {la} | {a[4]} | {h:.2f} | {c:.2f} | {len(bad[la])} |")
