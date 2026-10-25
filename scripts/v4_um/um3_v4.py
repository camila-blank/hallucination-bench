"""um3 for the V4 port: the official index aggregation
(/workspace/hb-halluc-index/outputs/hallucination_index/um3.py), verbatim rule, on
outputs/v4/um/{,capital_city/,holiday/}label/units_labeled.jsonl.

Per (method, layer, domain): share of committed, probe-testable (``probe.na_reason is None``)
sibling mentions labelled hallucination (faithful = probe fires at L, else cross-method allowed,
else hallucination). Domain macro (all 3 domains present) + item bootstrap within domain (1000,
seed 0). Method ``jlens`` is written as ``v4-jlens-std-n1000``.

    .venv/bin/python scripts/v4_um/um3_v4.py [--label-dir label] [--out um3_v4.json]
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

UM = Path(__file__).resolve().parents[2] / "outputs" / "v4" / "um"
DOMS = ["country", "capital_city", "holiday"]
NAME = {"jlens": "v4-jlens-std-n1000"}


def label(r):
    if r["probe"]["fire_L"]:
        return "faithful"
    return "cross" if r["cross"] else "hall"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-dir", default="label")
    ap.add_argument("--out", default=str(UM / "um3_v4.json"))
    a = ap.parse_args()
    units = defaultdict(list)  # (method, layer, dom) -> [(item, is_hall, label)]
    for d in DOMS:
        p = UM / ("" if d == "country" else d) / a.label_dir / "units_labeled.jsonl"
        for x in p.read_text().splitlines():
            r = json.loads(x)
            if r["committed"] and r["probe"]["na_reason"] is None:
                m = NAME.get(r["method"], r["method"])
                units[(m, r["layer"], d)].append((r["item"], label(r) == "hall", label(r)))

    rows = []
    for m, L in sorted({(k[0], k[1]) for k in units}):
        per = {}
        for d in DOMS:
            u = units.get((m, L, d), [])
            if u:
                per[d] = {
                    "n": len(u),
                    "hall": sum(h for _, h, _ in u) / len(u),
                    "faithful": sum(lb == "faithful" for *_, lb in u) / len(u),
                    "cross": sum(lb == "cross" for *_, lb in u) / len(u),
                    "n_items": len({i for i, *_ in u}),
                }
        rec = {"method": m, "layer": L, "per_domain": per}
        if len(per) == 3:
            rng = random.Random(0)
            by = {d: defaultdict(list) for d in DOMS}
            for d in DOMS:
                for i, h, _ in units[(m, L, d)]:
                    by[d][i].append(h)
            sims = []
            for _ in range(1000):
                vals = []
                for d in DOMS:
                    keys = list(by[d])
                    draw = [v for k in (rng.choice(keys) for _ in keys) for v in by[d][k]]
                    vals.append(sum(draw) / len(draw))
                sims.append(sum(vals) / 3)
            sims.sort()
            rec["macro"] = sum(per[d]["hall"] for d in DOMS) / 3
            rec["ci"] = [sims[24], sims[974]]
            rec["n"] = sum(per[d]["n"] for d in DOMS)
        rows.append(rec)
    Path(a.out).write_text(json.dumps(rows, indent=1))
    print(f"-> {a.out}")
    for r in rows:
        if "macro" in r:
            print(
                f"{r['method']:20s} L{r['layer']} n={r['n']:4d} macro={r['macro'] * 100:5.1f} "
                f"[{r['ci'][0] * 100:.1f},{r['ci'][1] * 100:.1f}] "
                + " ".join(
                    f"{d[:4]}={r['per_domain'][d]['hall'] * 100:.0f}%(n{r['per_domain'][d]['n']})"
                    for d in DOMS
                )
            )
        else:
            print(
                f"{r['method']:20s} L{r['layer']} missing domains",
                {d: v["n"] for d, v in r["per_domain"].items()},
            )


if __name__ == "__main__":
    main()
