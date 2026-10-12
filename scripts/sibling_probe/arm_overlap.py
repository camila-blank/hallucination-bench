"""Do s3d O-lens, NLA and J-lens name the same sibling concepts? (plan 0003 rev 11, item 5;
exploratory, outside the verdict)

    uv run python scripts/sibling_probe/arm_overlap.py [--n-perm 500] [--n-boot 2000]
    -> outputs/sibling/overlap/{overlap.json,overlap.md}

* **Sibling mention**: a category concept that is not the item's gold and not named in the
  prompt or in the response, that a lens names. s3d / NLA: any alias hit in the readout text.
  J-lens (primary): a raw top-10 token-alias hit AND a summary alias hit (the rule the score uses);
  rows with raw tokens only (``jlens-tok``) are reported separately.
* **Layer rule (primary)**: NLA L42 vs s3d / J-lens L44, compared at the same (item, position)
  (offset from the end); unit = (item, position, sibling). **Secondary**: the all-layer union per
  item, unit = (item, sibling).
* **Statistics**: P(B|A), lift = P(B|A)/P(B), Jaccard, each against TWO co-primary nulls with all
  three statistics permuted: B's per-item cell sets reassigned (a) among items of the same
  category and (b) among items with the same gold (conserves gold-driven siblings; most gold
  groups have 2 items, so this null is near-degenerate and conservative: the number of swappable
  items is reported). Observed statistics carry item-clustered bootstrap CIs. Jaccard is
  confounded (s3d reads 4 layers, NLA 1).
* **Agreement vs probe support**: unit = (item, sibling); lens count / strata (0 / 1 / 2 / 3 lenses
  naming it under the layer rule; plus s3d AND NLA, not J-lens, where the J-lens/probe link is
  mechanical). Probe support = mean over the item's L44 cells (every unit, whichever lens named
  it; L42 is a sensitivity row) of z(c | g) and of detection at tau_cat, under V2 and V0. z is
  also reported adjusted for concept fixed effects (popular siblings have stronger probes); CIs
  cluster on items.
"""

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from hallucination_bench import REPO_ROOT
from hallucination_bench.evals.sibling_probe import bank as B
from hallucination_bench.evals.sibling_probe import probe_io as PIO
from hallucination_bench.evals.sibling_probe.concepts import load_inventory

J = REPO_ROOT / "outputs/sibling/judge"
OUT = REPO_ROOT / "outputs/sibling/overlap"
S3D, NLA, JL, JT = "s3d-rl600", "nla-rl-L42", "jlens", "jlens-tok"
LENS = (S3D, NLA, JL)  # the three lenses (J-lens = token + summary rule)
SHORT = {S3D: "s3d", NLA: "NLA", JL: "J-lens", JT: "J-lens (tokens only)"}
LAYER_RULE = {S3D: 44, NLA: 42, JL: 44, JT: 44}  # primary layer per lens
PROBE_LAYER, SENS_LAYER = 44, 42


def r4(x: float | None, nd: int = 4) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


# ------------------------------------------------------------------ data


def load():
    inv = load_inventory()
    cats = {n: c.ids for n, c in inv.items()}
    te = {(n, c.id): c.token_eligible for n, cat in inv.items() for c in cat.concepts}
    items = {it["id"]: it for it in B.load_items()}
    ids = sorted(items)
    iix = {k: i for i, k in enumerate(ids)}
    cmax = max(len(v) for v in cats.values())
    cix = {n: {c: j for j, c in enumerate(v)} for n, v in cats.items()}
    meta = {}
    cells = {a: [] for a in (S3D, NLA, JL, JT)}  # (item idx, layer, offset, concept idxs)
    for arm in (S3D, NLA, JL):
        for line in open(J / arm / "cells.jsonl"):
            r = json.loads(line)
            it = r["id"]
            cat = r["category"]
            meta[it] = (
                cat,
                r["gold"],
                set(r.get("named") or ()),
                set(r.get("response_named") or ()),
            )
            ci = cix[cat]

            def idx(cs, ci=ci):
                return frozenset(ci[c] for c in cs if c in ci)

            if arm == JL:
                tok = set(r.get("token_hits") or ())
                summ = {
                    m["concept"] for m in r.get("mentions") or () if m.get("source") == "summary"
                }
                cells[JL].append((iix[it], r["layer"], r["offset"], idx(tok & summ)))
                cells[JT].append((iix[it], r["layer"], r["offset"], idx(tok)))
            else:
                ms = {
                    m["concept"]
                    for m in r.get("mentions") or ()
                    if m.get("source", "readout") == "readout"
                }
                cells[arm].append((iix[it], r["layer"], r["offset"], idx(ms)))
    N = len(ids)
    # candidate siblings per item: not gold, not named in the prompt or the response
    M = np.zeros((N, cmax), bool)
    for it, (cat, gold, named, resp) in meta.items():
        for c, j in cix[cat].items():
            M[iix[it], j] = c != gold and c not in named and c not in resp
    TE = np.zeros((N, cmax), bool)
    for it, (cat, _g, _n, _r) in meta.items():
        for c, j in cix[cat].items():
            TE[iix[it], j] = te[(cat, c)]
    return ids, items, meta, cats, cix, cells, M, TE


def build_arrays(cells, N, cmax):
    """[N, O, C] mention indicators and [N, O] validity (offset from the end = position) at each
    lens's primary layer (``LAYER_RULE``)."""
    omax = max(c[2] for rows in cells.values() for c in rows) + 1
    X = {a: np.zeros((N, omax, cmax), bool) for a in cells}
    V = {a: np.zeros((N, omax), bool) for a in cells}
    for a, rows in cells.items():
        for i, lay, off, cs in rows:
            if lay == LAYER_RULE[a]:
                V[a][i, off] = True
                for c in cs:
                    X[a][i, off, c] = True
    return X, V


def build_union(cells, N, cmax):
    X = {a: np.zeros((N, 1, cmax), bool) for a in cells}
    V = {a: np.ones((N, 1), bool) for a in cells}
    for a, rows in cells.items():
        for i, _lay, _off, cs in rows:
            for c in cs:
                X[a][i, 0, c] = True
    return X, V


# ------------------------------------------------------------------ overlap statistics


def counts(A, B_, VA, VB, M, pi=None):
    """Per-item counts (n_units, n_A, n_B, n_both) over units (item, position, sibling) present in
    both lenses; ``pi`` reassigns B's item rows (permutation null)."""
    if pi is not None:
        B_, VB = B_[pi], VB[pi]
    U = (VA & VB)[:, :, None] & M[:, None, :]
    xa, xb = A & U, B_ & U
    return np.stack([U.sum((1, 2)), xa.sum((1, 2)), xb.sum((1, 2)), (xa & xb).sum((1, 2))], axis=1)


def stats_from(tot):
    n, na, nb, nab = (float(x) for x in tot)
    pba = nab / na if na else np.nan
    pb = nb / n if n else np.nan
    return {
        "P(B|A)": pba,
        "P(A|B)": nab / nb if nb else np.nan,
        "lift": pba / pb if na and nb else np.nan,
        "jaccard": nab / (na + nb - nab) if na + nb - nab else np.nan,
    }


STAT_KEYS = ("P(B|A)", "P(A|B)", "lift", "jaccard")


def boot_ci(cnt, W):
    """Item-clustered bootstrap CIs of the statistics from per-item counts [N, 4]."""
    tot = W @ cnt  # [B, 4]
    out = {}
    for k in STAT_KEYS:
        v = np.array([stats_from(t)[k] for t in tot])
        v = v[np.isfinite(v)]
        out[k] = [r4(np.quantile(v, 0.025)), r4(np.quantile(v, 0.975))] if v.size else [None, None]
    return out


def groups_of(ids, meta, by):
    g = defaultdict(list)
    for i, k in enumerate(ids):
        cat, gold = meta[k][0], meta[k][1]
        g[cat if by == "category" else (cat, gold)].append(i)
    return list(g.values())


def perm_from(groups, rng, n):
    pi = np.arange(n)
    for g in groups:
        pi[g] = rng.permutation(g)
    return pi


def overlap(A, B_, VA, VB, M, ids, meta, n_perm, W, rng):
    cnt = counts(A, B_, VA, VB, M)
    obs = stats_from(cnt.sum(0))
    res = {
        "n_units": int(cnt[:, 0].sum()),
        "n_A": int(cnt[:, 1].sum()),
        "n_B": int(cnt[:, 2].sum()),
        "n_both": int(cnt[:, 3].sum()),
        "observed": {k: r4(obs[k]) for k in STAT_KEYS},
        "observed_ci95_item_bootstrap": boot_ci(cnt, W),
        "nulls": {},
    }
    for by in ("category", "gold"):
        groups = groups_of(ids, meta, by)
        sw = [g for g in groups if len(g) > 1]
        draws = {k: [] for k in STAT_KEYS}
        for _ in range(n_perm):
            s = stats_from(counts(A, B_, VA, VB, M, perm_from(groups, rng, len(ids))).sum(0))
            for k in STAT_KEYS:
                draws[k].append(s[k])
        nul = {}
        for k in STAT_KEYS:
            d = np.array(draws[k], float)
            d = d[np.isfinite(d)]
            nul[k] = {
                "null_mean": r4(d.mean()),
                "null_p95": r4(np.quantile(d, 0.95)),
                "excess": r4(obs[k] - d.mean()),
                "p_null_ge_obs": r4((np.sum(d >= obs[k]) + 1) / (d.size + 1)),
            }
        nul["swappable"] = {
            "n_groups": len(groups),
            "n_groups_swappable": len(sw),
            "n_items_swappable": int(sum(len(g) for g in sw)),
            "effective_swaps": int(sum(len(g) - 1 for g in sw)),
            "mean_group_size": r4(np.mean([len(g) for g in groups])),
        }
        res["nulls"][f"within_{by}"] = nul
    return res


# ------------------------------------------------------------------ probe support


def probe_support(recipe_dir: Path, layers=(PROBE_LAYER, SENS_LAYER)):
    """(layer, item, concept) -> (mean z, mean det_cat) over the item's cells at that layer."""
    acc = defaultdict(lambda: [0.0, 0, 0.0, 0])
    for r in PIO.iter_eval_scores(recipe_dir / "eval_scores.jsonl.gz"):
        if int(r["layer"]) not in layers:
            continue
        a = acc[(int(r["layer"]), r["id"], r["concept"])]
        if r.get("z") is not None:
            a[0] += r["z"]
            a[1] += 1
        if r.get("det_cat") is not None:
            a[2] += r["det_cat"]
            a[3] += 1
    return {
        k: (a[0] / a[1] if a[1] else np.nan, a[2] / a[3] if a[3] else np.nan)
        for k, a in acc.items()
    }


def fe_wls(y, X, g, w):
    """Coefficients of y ~ X + group FE under weights w (within transform)."""
    keep = w > 0
    y, X, g, w = y[keep], X[keep], g[keep], w[keep]
    gc = np.unique(g, return_inverse=True)[1]
    sw = np.bincount(gc, weights=w)

    def dm(v):
        if v.ndim == 1:
            return v - (np.bincount(gc, weights=w * v) / sw)[gc]
        return np.stack([dm(v[:, j]) for j in range(v.shape[1])], axis=1)

    yd, Xd = dm(y), dm(X)
    sq = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(Xd * sq[:, None], yd * sq, rcond=None)
    return beta


def ratio_ci(item, x, W):
    ok = np.isfinite(x)
    n = W.shape[1]
    num = np.bincount(item[ok], weights=x[ok], minlength=n)
    den = np.bincount(item[ok], minlength=n).astype(float)
    est = num.sum() / den.sum() if den.sum() else np.nan
    with np.errstate(all="ignore"):
        b = (W @ num) / (W @ den)
    b = b[np.isfinite(b)]
    return {
        "est": r4(est),
        "lo": r4(np.quantile(b, 0.025)) if b.size else None,
        "hi": r4(np.quantile(b, 0.975)) if b.size else None,
        "n": int(ok.sum()),
    }


def support_tables(units, support, layer, W, n_items):
    """Per stratum mean z / detection and the concept-FE-adjusted stratum effects."""
    item, concept, nl, combo = units["item"], units["concept"], units["n_lenses"], units["combo"]
    z = np.array(
        [
            support.get((layer, units["ids"][i], c), (np.nan, np.nan))[0]
            for i, c in zip(item, units["cname"], strict=True)
        ]
    )
    d = np.array(
        [
            support.get((layer, units["ids"][i], c), (np.nan, np.nan))[1]
            for i, c in zip(item, units["cname"], strict=True)
        ]
    )
    out = {"layer": layer, "strata": {}, "by_combination": {}, "fixed_effects": {}}
    strata = {
        "0 lenses": nl == 0,
        "1 lens": nl == 1,
        "2 lenses": nl == 2,
        "3 lenses": nl == 3,
        "s3d and NLA, not J-lens": combo == "s3d+NLA",
    }
    for name, m in strata.items():
        out["strata"][name] = {
            "n_units": int(m.sum()),
            "n_items": int(np.unique(item[m]).size),
            "z": ratio_ci(item[m], z[m], W),
            "det_tau_cat": ratio_ci(item[m], d[m], W),
        }
    for cb in sorted(set(combo.tolist())):
        m = combo == cb
        out["by_combination"][cb] = {
            "n_units": int(m.sum()),
            "z": ratio_ci(item[m], z[m], W),
            "det_tau_cat": ratio_ci(item[m], d[m], W),
        }
    # concept fixed effects: y ~ stratum dummies + (category, concept) FE; item bootstrap
    models = {
        "by_n_lenses": (
            ["1 lens", "2 lenses", "3 lenses"],
            np.stack([(nl == k).astype(float) for k in (1, 2, 3)], axis=1),
        ),
        "with_s3d_nla_split": (
            ["1 lens", "2 lenses: s3d+NLA (no J-lens)", "2 lenses incl. J-lens", "3 lenses"],
            np.stack(
                [
                    (nl == 1).astype(float),
                    (combo == "s3d+NLA").astype(float),
                    ((nl == 2) & (combo != "s3d+NLA")).astype(float),
                    (nl == 3).astype(float),
                ],
                axis=1,
            ),
        ),
    }
    gcode = np.unique(concept, return_inverse=True)[1]
    for mname, (labels, X) in models.items():
        for yname, y in (("z", z), ("det_tau_cat", d)):
            ok = np.isfinite(y)
            beta = fe_wls(y[ok], X[ok], gcode[ok], np.ones(int(ok.sum())))
            boots = []
            for b in range(W.shape[0]):
                w = W[b][item[ok]]
                try:
                    boots.append(fe_wls(y[ok], X[ok], gcode[ok], w))
                except np.linalg.LinAlgError:
                    continue
            boots = np.array(boots)
            out["fixed_effects"].setdefault(mname, {})[yname] = {
                lab: {
                    "coef_vs_0_lenses": r4(beta[j]),
                    "lo": r4(np.quantile(boots[:, j], 0.025)),
                    "hi": r4(np.quantile(boots[:, j], 0.975)),
                }
                for j, lab in enumerate(labels)
            }
    return out


# ------------------------------------------------------------------ main


def main(n_perm: int, n_boot: int, seed: int) -> dict:
    ids, _items, meta, cats, _cix, cells, M, TE = load()
    N, cmax = M.shape
    rng = np.random.default_rng(seed)
    W = rng.multinomial(N, np.full(N, 1.0 / N), size=n_boot).astype(float)
    Xp, Vp = build_arrays(cells, N, cmax)  # primary layer rule, (item, offset) cells
    Xu, Vu = build_union(cells, N, cmax)  # all-layer union per item
    res: dict = {
        "config": {
            "n_perm": n_perm,
            "n_boot": n_boot,
            "seed": seed,
            "n_items": N,
            "layer_rule": {SHORT[a]: LAYER_RULE[a] for a in LAYER_RULE},
            "mention_rule": "s3d / NLA: any alias hit in the readout; J-lens: raw top-10 token "
            "alias hit AND summary alias hit; siblings exclude the gold and concepts named in the "
            "prompt or the response",
            "n_candidate_siblings": int(M.sum()),
        },
        "primary_cell_level": {},
        "secondary_all_layer_union": {},
        "jlens_tokens_only": {},
        "triple": {},
        "probe_support": {},
    }
    pairs = list(combinations(LENS, 2))
    for scope, Mm in (("all siblings", M), ("token-eligible siblings", M & TE)):
        for a, b in pairs:
            if scope != "all siblings" and JL not in (a, b):
                continue
            key = f"{SHORT[a]} vs {SHORT[b]} [{scope}]"
            res["primary_cell_level"][key] = overlap(
                Xp[a], Xp[b], Vp[a], Vp[b], Mm, ids, meta, n_perm, W, rng
            )
            res["secondary_all_layer_union"][key] = overlap(
                Xu[a], Xu[b], Vu[a], Vu[b], Mm, ids, meta, n_perm, W, rng
            )
    for a in (S3D, NLA):
        key = f"{SHORT[a]} vs {SHORT[JT]} [all siblings]"
        res["jlens_tokens_only"][key] = {
            "primary_cell_level": overlap(
                Xp[a], Xp[JT], Vp[a], Vp[JT], M, ids, meta, n_perm, W, rng
            ),
            "all_layer_union": overlap(Xu[a], Xu[JT], Vu[a], Vu[JT], M, ids, meta, n_perm, W, rng),
        }
    # who names each (item, sibling): lens sets under the layer rule (item level)
    named = {a: Xp[a].any(axis=1) for a in LENS}  # [N, C]
    union_named = {a: Xu[a][:, 0, :] for a in LENS}
    for label, nm in (("layer_rule", named), ("all_layer_union", union_named)):
        venn = Counter()
        for i in range(N):
            for c in np.flatnonzero(M[i]):
                venn["+".join(SHORT[a] for a in LENS if nm[a][i, c]) or "none"] += 1
        res["triple"][label] = dict(venn.most_common())
    # probe support
    ii, cc = np.nonzero(M)
    cname = np.array([cats[meta[ids[i]][0]][c] for i, c in zip(ii, cc, strict=True)], dtype=object)
    flags = np.stack([named[a][ii, cc] for a in LENS], axis=1)
    combo = np.array(
        [
            "+".join(SHORT[a] for a, f in zip(LENS, row, strict=True) if f) or "none"
            for row in flags
        ],
        dtype=object,
    )
    catname = np.array([meta[ids[i]][0] for i in ii], dtype=object)
    units = {
        "item": ii,
        "ids": ids,
        "cname": cname,
        "concept": np.array(
            [f"{a}/{b}" for a, b in zip(catname, cname, strict=True)], dtype=object
        ),
        "n_lenses": flags.sum(1),
        "combo": combo,
        "te": TE[ii, cc],
    }
    res["probe_support"]["n_units"] = len(ii)
    for recipe, d in PIO.recipe_dirs(PIO.PROBES_DIR).items():
        sup = probe_support(d)
        rr = {}
        for layer in (PROBE_LAYER, SENS_LAYER):
            rr[f"L{layer}" + ("" if layer == PROBE_LAYER else " (sensitivity)")] = support_tables(
                units, sup, layer, W[: min(n_boot, 1000)], N
            )
        te = units["te"]
        sub = {k: (v[te] if isinstance(v, np.ndarray) else v) for k, v in units.items()}
        rr["L44 token-eligible siblings only"] = support_tables(
            sub, sup, PROBE_LAYER, W[: min(n_boot, 1000)], N
        )
        res["probe_support"][recipe] = rr
    return res


def fmt(ci):
    if not ci or ci.get("est") is None:
        return "–"
    return f"{ci['est']:.3f} [{ci['lo']:.3f}, {ci['hi']:.3f}]"


def markdown(res: dict) -> str:
    cfg = res["config"]
    L = [
        "# Cross-lens overlap on sibling concepts (plan 0003 rev 11, exploratory)",
        "",
        f"{cfg['n_items']} items, {cfg['n_candidate_siblings']} candidate (item, sibling) units; "
        f"layer rule {json.dumps(cfg['layer_rule'])}. {cfg['mention_rule']}. "
        f"{cfg['n_perm']} permutations per null, {cfg['n_boot']} item-bootstrap draws.",
    ]

    def table(title, d):
        L.extend(
            [
                "",
                f"## {title}",
                "",
                "| pair | units | n A | n B | n both | stat | observed [95% CI] | within-category "
                "null (mean, p95, excess) | within-same-gold null (mean, p95, excess) |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
        )
        for k, v in d.items():
            for s in ("P(B|A)", "lift", "jaccard"):
                ci = v["observed_ci95_item_bootstrap"][s]
                nc, ng = v["nulls"]["within_category"][s], v["nulls"]["within_gold"][s]
                L.append(
                    f"| {k if s == 'P(B|A)' else ''} | {v['n_units'] if s == 'P(B|A)' else ''} | "
                    f"{v['n_A'] if s == 'P(B|A)' else ''} | {v['n_B'] if s == 'P(B|A)' else ''} | "
                    f"{v['n_both'] if s == 'P(B|A)' else ''} | {s} | {v['observed'][s]} "
                    f"[{ci[0]}, {ci[1]}] | {nc['null_mean']}, {nc['null_p95']}, {nc['excess']} | "
                    f"{ng['null_mean']}, {ng['null_p95']}, {ng['excess']} |"
                )
        v0 = next(iter(d.values()))
        sw = v0["nulls"]["within_gold"]["swappable"]
        L.extend(
            [
                "",
                f"Same-gold null: {sw['n_groups']} (category, gold) groups, "
                f"{sw['n_groups_swappable']} with at least 2 items; {sw['n_items_swappable']} "
                f"items swappable, effective number of swaps {sw['effective_swaps']} "
                f"(mean group size {sw['mean_group_size']}). "
                "Near-degenerate and conservative.",
            ]
        )

    table(
        "Primary: NLA L42 vs s3d / J-lens L44 at the same (item, position)",
        res["primary_cell_level"],
    )
    table("Secondary: all-layer union per item", res["secondary_all_layer_union"])
    L += ["", "## J-lens raw tokens only (no summary requirement), reported separately", ""]
    flat = {}
    for k, v in res["jlens_tokens_only"].items():
        flat[k + " (primary layer rule)"] = v["primary_cell_level"]
        flat[k + " (all-layer union)"] = v["all_layer_union"]
    table("J-lens tokens only", flat)
    L += ["", "## Which lenses name each (item, sibling)", ""]
    for label, v in res["triple"].items():
        L.append(f"- {label}: `{json.dumps(v)}`")
    ps = res["probe_support"]
    L += ["", f"## Agreement vs probe support ({ps['n_units']} (item, sibling) units)", ""]
    for recipe, blocks in ps.items():
        if recipe == "n_units":
            continue
        for bname, b in blocks.items():
            L += [
                "",
                f"### {recipe.upper()}, {bname}",
                "",
                "| stratum | units | items | mean z(c|g) [CI] | detection tau_cat [CI] |",
                "|---|---|---|---|---|",
            ]
            for s, v in b["strata"].items():
                L.append(
                    f"| {s} | {v['n_units']} | {v['n_items']} | {fmt(v['z'])} | "
                    f"{fmt(v['det_tau_cat'])} |"
                )
            L += [
                "",
                "By lens combination:",
                "",
                "| combination | units | z | detection |",
                "|---|---|---|---|",
            ]
            for s, v in b["by_combination"].items():
                L.append(f"| {s} | {v['n_units']} | {fmt(v['z'])} | {fmt(v['det_tau_cat'])} |")
            L += [
                "",
                "Concept fixed effects (coefficient vs 0 lenses, within (category, concept); "
                "item-bootstrap 95% CI):",
                "",
                "| model | outcome | stratum | coef [CI] |",
                "|---|---|---|---|",
            ]
            for mname, ys in b["fixed_effects"].items():
                for yname, labs in ys.items():
                    for lab, v in labs.items():
                        L.append(
                            f"| {mname} | {yname} | {lab} | {v['coef_vs_0_lenses']} "
                            f"[{v['lo']}, {v['hi']}] |"
                        )
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    result = main(a.n_perm, a.n_boot, a.seed)
    (out / "overlap.json").write_text(json.dumps(result, indent=1))
    (out / "overlap.md").write_text(markdown(result))
    print((out / "overlap.md").read_text())
