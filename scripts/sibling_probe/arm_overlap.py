"""Do s3d O-lens, NLA and J-lens name the same sibling concepts? (Camila 2026-10-11 follow-up)

Sibling = a category concept that is not the item's gold and not named in its prompt.
Mention = s3d / NLA: any alias hit in the readout; J-lens: a raw top-10 token alias hit
(the lens's own output, not the LLM summary). Units: (item, sibling) over all cells, and
(item, layer, pos, sibling) cells for s3d vs J-lens (same layers). Null: arm B's per-item
sibling sets permuted across items of the same category (keeps concept popularity) and across
items with the same gold (keeps gold-driven siblings).
"""

import json
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
J = ROOT / "outputs/sibling/judge"
ARMS = ("s3d-rl600", "nla-rl-L42", "jlens")
OUT = ROOT / "outputs/sibling/overlap"

items = {
    it["id"]: it for it in json.loads((ROOT / "data/sibling_probe/items.json").read_text())["items"]
}
concepts = json.loads((ROOT / "data/sibling_probe/concepts.json").read_text())["categories"]
cat_concepts = {c: [x["id"] for x in v["concepts"]] for c, v in concepts.items()}


def mentions(arm, rec):
    if arm == "jlens":
        return set(rec.get("token_hits") or {})
    return {
        m["concept"] for m in rec.get("mentions") or [] if m.get("source", "readout") == "readout"
    }


item_sets = {a: defaultdict(set) for a in ARMS}  # arm -> item -> siblings
cell_sets = {a: {} for a in ARMS}  # arm -> (item, layer, pos) -> siblings
meta = {}
for a in ARMS:
    for line in open(J / a / "cells.jsonl"):
        r = json.loads(line)
        meta[r["id"]] = (r["category"], r["gold"], set(r.get("named") or []))
        sib = mentions(a, r) - {r["gold"]} - set(r.get("named") or [])
        item_sets[a][r["id"]] |= sib
        cell_sets[a][(r["id"], r["layer"], r["pos"])] = sib

ids = sorted(meta)


def pair_units(a, b, sets_a, sets_b, keys):
    """indicator vectors over (key, sibling) for every candidate sibling of the key's item."""
    xa, xb, cats, golds, grp = [], [], [], [], []
    for k in keys:
        iid = k if isinstance(k, str) else k[0]
        cat, gold, named = meta[iid]
        for c in cat_concepts[cat]:
            if c == gold or c in named:
                continue
            xa.append(c in sets_a.get(k, ())), xb.append(c in sets_b.get(k, ()))
            cats.append(cat), golds.append(gold), grp.append((k, c))
    return np.array(xa), np.array(xb), cats, golds, grp


def stats(xa, xb):
    na, nb, nab = xa.sum(), xb.sum(), (xa & xb).sum()
    union = (xa | xb).sum()
    return {
        "n_units": len(xa),
        "n_A": int(na),
        "n_B": int(nb),
        "n_both": int(nab),
        "jaccard": round(nab / union, 3) if union else None,
        "P(B|A)": round(nab / na, 3) if na else None,
        "P(A|B)": round(nab / nb, 3) if nb else None,
        "P(B)": round(nb / len(xb), 4),
        "lift": round((nab / na) / (nb / len(xb)), 2) if na and nb else None,
    }


def permuted_jaccard(a, b, sets_a, sets_b, by, n=200, seed=0):
    """null Jaccard: B's item sets reassigned among items sharing `by` (category or gold)."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for i in ids:
        groups[meta[i][0] if by == "category" else (meta[i][0], meta[i][1])].append(i)
    out = []
    for _ in range(n):
        perm = {}
        for g in groups.values():
            sh = g[:]
            rng.shuffle(sh)
            perm.update({i: sets_b.get(j, set()) for i, j in zip(g, sh, strict=True)})
        xa, xb, *_ = pair_units(a, b, sets_a, perm, ids)
        u = (xa | xb).sum()
        out.append((xa & xb).sum() / u if u else 0.0)
    return round(float(np.mean(out)), 3), round(float(np.percentile(out, 95)), 3)


res = {"item_level": {}, "cell_level": {}, "triple": {}}
for a, b in combinations(ARMS, 2):
    xa, xb, cats, *_ = pair_units(a, b, item_sets[a], item_sets[b], ids)
    s = stats(xa, xb)
    s["null_jaccard_category_perm(mean,p95)"] = permuted_jaccard(
        a, b, item_sets[a], item_sets[b], "category"
    )
    s["null_jaccard_gold_perm(mean,p95)"] = permuted_jaccard(
        a, b, item_sets[a], item_sets[b], "gold"
    )
    s["by_category"] = {}
    cats = np.array(cats)
    for c in sorted(set(cats)):
        m = cats == c
        s["by_category"][c] = stats(xa[m], xb[m])
    res["item_level"][f"{a} vs {b}"] = s

# cell level: s3d vs jlens at identical (item, layer, pos)
keys = sorted(set(cell_sets["s3d-rl600"]) & set(cell_sets["jlens"]))
xa, xb, *_ = pair_units("s3d", "jlens", cell_sets["s3d-rl600"], cell_sets["jlens"], keys)
res["cell_level"]["s3d-rl600 vs jlens (same layer, pos)"] = stats(xa, xb)
# NLA L42 vs s3d L44 at the same (item, pos)
nla = {(k[0], k[2]): v for k, v in cell_sets["nla-rl-L42"].items()}
s3 = {(k[0], k[2]): v for k, v in cell_sets["s3d-rl600"].items() if k[1] == 44}
jl = {(k[0], k[2]): v for k, v in cell_sets["jlens"].items() if k[1] == 44}
keys = sorted(set(nla) & set(s3))
for name, other in (
    ("s3d-rl600@L44 vs nla-rl-L42 (same pos)", s3),
    ("jlens@L44 vs nla-rl-L42 (same pos)", jl),
):
    xa, xb, *_ = pair_units("x", "nla", other, nla, keys)
    res["cell_level"][name] = stats(xa, xb)

# triple: how many arms name each (item, sibling); Venn counts
venn = defaultdict(int)
per = defaultdict(lambda: defaultdict(int))
for i in ids:
    cat, gold, named = meta[i]
    for c in cat_concepts[cat]:
        if c == gold or c in named:
            continue
        key = tuple(a for a in ARMS if c in item_sets[a][i])
        venn[key] += 1
        per[cat][len(key)] += 1
res["triple"]["venn_item_sibling"] = {
    "+".join(k) or "none": v for k, v in sorted(venn.items(), key=lambda kv: -kv[1])
}
res["triple"]["n_arms_by_category"] = {c: dict(v) for c, v in per.items()}

# top siblings named by all three
all3 = defaultdict(int)
for i in ids:
    cat, gold, named = meta[i]
    for c in set.intersection(*(item_sets[a][i] for a in ARMS)):
        all3[(cat, gold, c)] += 1
res["triple"]["named_by_all_three(top)"] = [
    {"category": k[0], "gold": k[1], "sibling": k[2], "items": v}
    for k, v in sorted(all3.items(), key=lambda kv: -kv[1])[:20]
]

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "overlap.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != "triple"}, indent=1)[:6000])
print(json.dumps(res["triple"], indent=1)[:3000])
