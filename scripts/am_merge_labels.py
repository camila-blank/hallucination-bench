"""Merge agent label files (audit/labels/*.jsonl) into the audit sheets' `label` fields (plan 0008).

Each label row is matched on (sheet, arm, key, start, end). Unmatched label rows, and sheet rows
(other than camila_subset) left without a label, are reported; nothing else is changed.

    uv run --no-sync python scripts/am_merge_labels.py [--audit <audit dir>]
"""

import argparse
import json
from pathlib import Path


def _k(r: dict) -> tuple:
    return (r["sheet"], r["arm"], r["key"], r.get("start"), r.get("end"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="outputs/agentic_misalignment/claude-opus-5-5/audit")
    a = ap.parse_args()
    ad = Path(a.audit)
    labels: dict[tuple, object] = {}
    for f in sorted((ad / "labels").glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                labels[_k(r)] = r["label"]
    used: set[tuple] = set()
    for sheet in sorted(ad.glob("*.jsonl")):
        if sheet.stem == "camila_subset":
            continue
        rows = [json.loads(x) for x in sheet.read_text().splitlines() if x.strip()]
        n_set = n_missing = 0
        for r in rows:
            k = _k(r)
            if k in labels:
                r["label"] = labels[k]
                used.add(k)
                n_set += 1
            elif r.get("label") is None:
                n_missing += 1
        sheet.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"{sheet.name}: labelled {n_set}, still unlabelled {n_missing}")
    print(f"label rows unmatched: {len(set(labels) - used)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
