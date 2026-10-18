# ruff: noqa: E501
"""Plan 0004 summary / report / calibration samples (used by ``verdicts.py``)."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from hallucination_bench import DATA
from hallucination_bench.evals.sibling_probe import claims as CL
from hallucination_bench.evals.sibling_probe import verdict as V

N_BOOT = 1000
LAYERS = (20, 24, 28, 32, 36, 40, 42, 44, 48, 52, 56, 60)


def boot_share(
    items: list[str], x: list[float], n_boot: int = N_BOOT, seed: int = 0
) -> dict[str, Any]:
    """Share mean(x) with an item-clustered bootstrap 95% CI."""
    if not x:
        return {"est": None, "lo": None, "hi": None, "n": 0}
    ids = sorted(set(items))
    idx = {k: i for i, k in enumerate(ids)}
    g = np.array([idx[i] for i in items])
    xs = np.array(x, dtype=float)
    s = np.bincount(g, weights=xs, minlength=len(ids))
    c = np.bincount(g, minlength=len(ids)).astype(float)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=n_boot)
    num, den = W @ s, W @ c
    b = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    return {
        "est": round(float(xs.mean()), 4),
        "lo": round(float(np.nanpercentile(b, 2.5)), 4),
        "hi": round(float(np.nanpercentile(b, 97.5)), 4),
        "n": len(xs),
        "n_items": len(ids),
    }


def _grp(rows, key):
    out = defaultdict(list)
    for r in rows:
        out[key(r)].append(r)
    return out


def summarise(rows: list[dict], api_usd: float = 0.0) -> dict[str, Any]:
    S: dict[str, Any] = {"n_units": len(rows), "verdict_prompt_version": V.PROMPT_VERSION}
    # counts per method x set x category (headline denominator = all units)
    cnt = defaultdict(Counter)
    for r in rows:
        cnt[f"{r['set']}|{r['method']}"][r["category"]] += 1
    S["counts"] = {k: dict(v) | {"total": sum(v.values())} for k, v in sorted(cnt.items())}
    S["counts_v0"] = {
        k: dict(Counter(r["category_v0"] for r in rs))
        for k, rs in sorted(_grp(rows, lambda r: f"{r['set']}|{r['method']}").items())
    }
    S["readout_samples"] = {
        k: sum(len(r["readouts"]) for r in rs)
        for k, rs in sorted(_grp(rows, lambda r: f"{r['set']}|{r['method']}").items())
    }
    S["kinds"] = {
        k: dict(Counter(r["kind"] for r in rs))
        for k, rs in sorted(_grp(rows, lambda r: f"{r['set']}|{r['method']}").items())
    }
    # figure 1: probe agreement (detected / units with a probe verdict) + n/a share
    agree, by_layer = {}, {}
    for k, rs in sorted(_grp(rows, lambda r: (r["set"], r["method"])).items()):
        ok = [r for r in rs if r["probe"]["status"] == "ok"]
        agree[f"{k[0]}|{k[1]}"] = boot_share(
            [r["item"] for r in ok], [float(r["probe"]["detected"]) for r in ok]
        ) | {
            "na_share": round(1 - len(ok) / len(rs), 4) if rs else None,
            "n_units": len(rs),
            "na_reasons": dict(
                Counter(r["probe"].get("na_reason") for r in rs if r["probe"]["status"] != "ok")
            ),
        }
        for L, rl in sorted(_grp(rs, lambda r: r["layer"]).items()):
            okl = [r for r in rl if r["probe"]["status"] == "ok"]
            by_layer[f"{k[0]}|{k[1]}|{L}"] = boot_share(
                [r["item"] for r in okl], [float(r["probe"]["detected"]) for r in okl]
            ) | {"na_share": round(1 - len(okl) / len(rl), 4), "n_units": len(rl)}
    S["probe_agreement"], S["probe_agreement_by_layer"] = agree, by_layer
    # hallucination rate: over all units (headline) and over units with a probe verdict; with and without consensus
    hr = {}
    for k, rs in sorted(_grp(rows, lambda r: f"{r['set']}|{r['method']}").items()):
        h = [float(r["hallucination"]) for r in rs]
        nocons = [
            float(
                r["hallucination"]
                or ((r["category"] == "consensus") and r["form_label"] == "assertion")
            )
            for r in rs
        ]
        ok = [r for r in rs if r["probe"]["status"] == "ok"]
        hr[k] = {
            "all_units": boot_share([r["item"] for r in rs], h),
            "all_units_without_consensus": boot_share([r["item"] for r in rs], nocons),
            "with_probe_verdict": boot_share(
                [r["item"] for r in ok], [float(r["hallucination"]) for r in ok]
            ),
            "probe_na_asserted": sum(
                r["category"] == "probe_na" and r["form_label"] == "assertion" for r in rs
            ),
        }
    S["hallucination_rate"] = hr
    # depth panel
    S["by_depth"] = {
        f"{k[0]}|{k[1]}|{k[2]}": dict(Counter(r["category"] for r in rs))
        for k, rs in sorted(_grp(rows, lambda r: (r["set"], r["method"], r["depth"])).items())
    }
    # consensus
    cons = [
        r for r in rows if (r["consensus_mlnla"] if r["method"] == CL.MLNLA else r["consensus"])
    ]
    S["consensus"] = {
        "triples": len(
            {
                (r["set"], r["item"], r["pos"], r["concept"])
                for r in rows
                if r["consensus"] and r["method"] == CL.NLA42
            }
        ),
        "units_primary": dict(Counter(f"{r['set']}|{r['method']}" for r in rows if r["consensus"])),
        "units_mlnla_variant": dict(
            Counter(f"{r['set']}|{r['method']}" for r in rows if r["consensus_mlnla"])
        ),
        "early_layer_units": sum(1 for r in cons if r["depth"] == "early"),
        "probe_rejects": sum(
            1 for r in cons if r["probe"]["status"] == "ok" and not r["probe"]["detected"]
        ),
        "hallucinated_consensus": sum(
            1 for r in rows if r["category"] == "consensus" and r["form_label"] == "assertion"
        ),
        "consensus_category_units": sum(1 for r in rows if r["category"] == "consensus"),
    }
    S["validated_form"] = dict(
        Counter(r["form_label"] for r in rows if r["category"] == "validated")
    )
    # judge vs rule (layer-blind run), on the judge pool units with a blind verdict
    jr = {}
    blind = [
        r
        for r in rows
        if r.get("judge_blind")
        and r["kind"] == "sibling"
        and r["probe"]["status"] == "ok"
        and not r["probe"]["detected"]
    ]

    def jcat(r, fld):
        return V.judge_category(r[fld]["form_label"])

    for cut in [None, *V.CUTS]:
        name = "default" if cut is None else f"cut{cut}"
        per = {}
        for L, rl in sorted(_grp(blind, lambda r: r["layer"]).items()):
            rl2 = [r for r in rl if V.rule_label(L, cut) != "unclear"]
            per[str(L)] = {
                "n": len(rl2),
                "agree": round(
                    sum(jcat(r, "judge_blind") == V.rule_label(L, cut) for r in rl2) / len(rl2), 4
                )
                if rl2
                else None,
                "judge_dist": dict(Counter(jcat(r, "judge_blind") for r in rl)),
            }
        allr = [r for r in blind if V.rule_label(r["layer"], cut) != "unclear"]
        jr[name] = {
            "overall": round(
                sum(jcat(r, "judge_blind") == V.rule_label(r["layer"], cut) for r in allr)
                / len(allr),
                4,
            )
            if allr
            else None,
            "n": len(allr),
            "by_layer": per,
        }
    S["judge_vs_rule_blind"] = jr
    both = [r for r in rows if r.get("judge_blind") and r.get("judge")]
    S["visible_vs_blind_agreement"] = {
        "n": len(both),
        "agree": round(
            sum(
                V.judge_category(r["judge"]["form_label"])
                == V.judge_category(r["judge_blind"]["form_label"])
                for r in both
            )
            / len(both),
            4,
        )
        if both
        else None,
    }
    S["judge_quote_failures"] = sum(r["judge"]["n_unquoted"] for r in rows if r.get("judge"))
    S["judge_missing"] = sum(1 for r in rows if not r.get("judge") or not r["judge"]["answered"])
    # set-B coverage (rev 3 item 7)
    b = [r for r in rows if r["set"] == "B"]
    bsib = [r for r in b if r["kind"] == "sibling"]
    val = [
        r
        for r in bsib
        if r["probe"].get("na_reason")
        not in (
            "unvalidated_never_gold",
            "no_probe",
            "transfer_undefined",
            "category_not_probed",
            "out_of_inventory",
            "entity_substitution",
        )
    ]
    S["set_b_coverage"] = {
        "units": len(b),
        "sibling_units": len(bsib),
        "validatable_units": len(val),
        "validatable_share": round(len(val) / len(b), 4) if b else None,
        "limitation": bool(b) and len(val) / max(len(b), 1) < 0.30,
        "claimed_concepts": len({(r["category"], r["concept"]) for r in b}),
        "claimed_concepts_validatable": len({(r["category"], r["concept"]) for r in val}),
    }
    S["worked_examples_ali"] = [
        {k: r[k] for k in ("method", "layer", "concept", "category", "form_label", "rule")}
        | {
            "segments": [s["text"] for s in r["segments"]],
            "z": r["probe"].get("z"),
            "detected": r["probe"]["detected"],
            "na_reason": r["probe"].get("na_reason"),
        }
        for r in sorted(
            (r for r in rows if r["item"] == "br-ent-ath-ali"),
            key=lambda r: (r["method"], r["layer"]),
        )
    ]
    S["api_usd_verdict_judge"] = round(api_usd, 2)
    return S


def _fmt(ci):
    if not ci or ci.get("est") is None:
        return "—"
    return f"{ci['est']:.2f} [{ci['lo']:.2f}, {ci['hi']:.2f}] (n={ci['n']})"


def markdown(S: dict) -> str:
    L = [
        "# Plan 0004 results scaffold: sibling claims, probes, verdicts",
        "",
        f"Units: {S['n_units']}. Verdict judge {S['verdict_prompt_version']}. Spend (verdict judge cache): ${S['api_usd_verdict_judge']}.",
        "",
        "## Claim units per category (denominator = all units)",
        "",
        "| set/method | " + " | ".join(V.CATEGORIES) + " | total |",
        "|---|" + "---|" * (len(V.CATEGORIES) + 1),
    ]
    for k, c in S["counts"].items():
        L.append(
            f"| {k} | " + " | ".join(str(c.get(x, 0)) for x in V.CATEGORIES) + f" | {c['total']} |"
        )
    L += [
        "",
        "## Probe agreement (detected / units with a probe verdict; item-clustered 95% CI)",
        "",
        "| set/method | agreement | probe n/a share | n/a reasons |",
        "|---|---|---|---|",
    ]
    for k, a in S["probe_agreement"].items():
        L.append(f"| {k} | {_fmt(a)} | {a['na_share']} | {json.dumps(a['na_reasons'])} |")
    L += [
        "",
        "## Hallucination rate",
        "",
        "| set/method | all units | all units, consensus ignored | with probe verdict | probe-n/a asserted |",
        "|---|---|---|---|---|",
    ]
    for k, h in S["hallucination_rate"].items():
        L.append(
            f"| {k} | {_fmt(h['all_units'])} | {_fmt(h['all_units_without_consensus'])} | {_fmt(h['with_probe_verdict'])} | {h['probe_na_asserted']} |"
        )
    L += [
        "",
        "## Consensus (post hoc)",
        "",
        "```",
        json.dumps(S["consensus"], indent=1),
        "```",
        "",
        "## Judge vs rule (layer-blind run)",
        "",
    ]
    for name, j in S["judge_vs_rule_blind"].items():
        L.append(f"- {name}: overall agreement {j['overall']} (n={j['n']})")
    L += [
        "",
        f"Visible vs blind judge agreement: {S['visible_vs_blind_agreement']}",
        "",
        f"Judge vs blind gold (100 units): {json.dumps(S.get('gold'))}",
        "",
        "## Set-B coverage",
        "",
        "```",
        json.dumps(S["set_b_coverage"], indent=1),
        "```",
        "",
        "## Ali (br-ent-ath-ali) claim units",
        "",
    ]
    for e in S["worked_examples_ali"]:
        L.append(
            f"- {e['method']} L{e['layer']} {e['concept']}: **{e['category']}** (form {e['form_label']}, rule {e['rule']}, probe z={e['z']} det={e['detected']} {e['na_reason'] or ''}): "
            + " / ".join(s[:120] for s in e["segments"])
        )
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ calibration samples


def _rank(uid: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{uid}".encode()).hexdigest()


def _strat(rows, n, salt, exclude=frozenset(), fixed=()):
    fixed = [r for r in fixed if r["uid"] not in exclude]
    pool = [
        r for r in rows if r["uid"] not in exclude and r["uid"] not in {f["uid"] for f in fixed}
    ]
    strata = defaultdict(list)
    for r in pool:
        strata[(r["method"], r["depth"], r["set"])].append(r)
    need = max(0, n - len(fixed))
    out = list(fixed)
    keys = sorted(strata)
    # round-robin over strata (balanced), seeded order within each
    for k in keys:
        strata[k].sort(key=lambda r: _rank(r["uid"], salt))
    i = 0
    while len(out) < len(fixed) + need and any(strata.values()):
        k = keys[i % len(keys)]
        if strata[k]:
            out.append(strata[k].pop(0))
        i += 1
    return out[: max(n, len(fixed))]


def calib_record(r: dict, ctx_render) -> dict:
    """Blind record: the prompt, response, marked readout, concept and layer; NO judge output,
    NO probe result, NO category."""
    return {"uid": r["uid"], **ctx_render(r)}


def write_calibration(rows: list[dict], ctx_render=None) -> dict[str, Path]:
    import verdicts as VV

    ctx = VV.Context()
    keys_by_arm = defaultdict(set)
    for r in rows:
        keys_by_arm[r["arm"]].add(r["cell_key"])

    def rend(r):
        cell = ctx.samples(r["arm"], r["set"], keys_by_arm[r["arm"]]).get(r["cell_key"])
        rec = ctx.cells(r["arm"], keys_by_arm[r["arm"]]).get(r["cell_key"]) or {}
        if cell is None:
            samples = []
        elif cell.kind == "tokens":
            samples = [rec.get("summary") or ""]
        else:
            samples = [s for s in cell.samples or () if s.strip()][: VV.HAL_K]
        if r["set"] == "A":
            it = ctx.ia[r["item"]]
            prompt, resp = VV.SB.user_text(it["messages"]), it.get("response") or ""
        else:
            it = ctx.ib[r["item"]]
            prompt, resp = it["prompt"], VV.BB.response_text(it)
        return {
            "set": r["set"],
            "method": r["method"],
            "item": r["item"],
            "layer": r["layer"],
            "concept": ctx.concept_label(r),
            "prompt": prompt,
            "response": resp,
            "readout_marked": V.mark_readouts(samples, r["segments"]),
            "segments": [f"S{k}" for k in range(1, len(r["segments"]) + 1)],
        }

    pool = [r for r in rows if r["kind"] in ("sibling", "entity_substitution")]
    ali = [r for r in pool if r["item"] == "br-ent-ath-ali"]
    gold = _strat(pool, 100, "gold", fixed=ali)
    used = {r["uid"] for r in gold}
    dev = _strat(pool, 40, "dev", exclude=used)
    used |= {r["uid"] for r in dev}
    human = _strat(pool, 30, "human", exclude=used - {r["uid"] for r in ali}, fixed=ali[:6])
    out = {}
    for name, rs in (
        ("verdict_gold_todo", gold),
        ("verdict_dev_todo", dev),
        ("verdict_human_todo", human),
    ):
        p = DATA / "sibling_probe" / f"{name}.jsonl"
        with p.open("w", encoding="utf-8") as fh:
            for r in rs:
                fh.write(json.dumps(calib_record(r, rend), ensure_ascii=False) + "\n")
        out[name] = p
        print(f"[calib] {len(rs)} -> {p}")
    return out


# ------------------------------------------------------------------ gold accuracy


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if not n:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return round(c - h, 4), round(c + h, 4)


def gold_accuracy(rows: list[dict], gold_path: Path) -> dict[str, Any]:
    """Judge (layer-visible) vs blind gold labels. Gold rows: {uid, label in {hallucination,
    option, unclear}, seg_forms?}. Reports accuracy on {option, hallucination} gold units (the
    target >= 85%), overall accuracy, per-class precision / recall and the confusion matrix."""
    by = {r["uid"]: r for r in rows}
    gold = [json.loads(x) for x in gold_path.read_text().splitlines() if x.strip()]
    pairs = []
    for g in gold:
        r = by.get(g["uid"])
        if r is None or not r.get("judge"):
            continue
        pairs.append((g["label"], V.judge_category(r["judge"]["form_label"])))
    labs = ("hallucination", "option", "unclear")
    conf = {a: {b: sum(1 for x, y in pairs if x == a and y == b) for b in labs} for a in labs}
    oh = [(x, y) for x, y in pairs if x in ("option", "hallucination")]
    k = sum(x == y for x, y in oh)
    per = {}
    for c in labs:
        tp = conf[c][c]
        pp = sum(conf[a][c] for a in labs)
        ap = sum(conf[c].values())
        per[c] = {
            "precision": round(tp / pp, 4) if pp else None,
            "recall": round(tp / ap, 4) if ap else None,
            "n_gold": ap,
        }
    acc_oh = round(k / len(oh), 4) if oh else None
    return {
        "n": len(pairs),
        "accuracy_option_halluc": acc_oh,
        "wilson95": wilson(k, len(oh)),
        "n_option_halluc": len(oh),
        "accuracy_all": round(sum(x == y for x, y in pairs) / len(pairs), 4) if pairs else None,
        "per_class": per,
        "confusion_gold_rows_judge_cols": conf,
        "target_met": bool(acc_oh is not None and acc_oh >= 0.85),
    }
