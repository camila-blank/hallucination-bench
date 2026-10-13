"""``hbench report --eval agentic_misalignment`` (plan 0008, Scoring and report).

Rebuilds every arm under ``<out>/agentic_misalignment/<model>/`` from its readouts (default
root: the paper readouts) plus the cache, then writes ``report.md`` / ``report.json`` there:
the per-arm (and per-layer) tables §1-§6 and the §7 audit samples. Reads
``audit/gates.json`` and ``audit/tiers.json`` when present (``scripts/am_audit.py score``);
without ``tiers.json`` the unrelated rate is labelled "uncorrected", and without the
``lost_in_summary`` audit the J-lens ``lost_in_summary`` count is labelled an upper bound.
Every number here is **provisional** (no held-out gate: the claim map's gate is chat gold).
"""

import json
import random
from pathlib import Path
from typing import Any

from hallucination_bench import taxonomy_v2 as tx
from hallucination_bench.evals.wsbench_chat.run import Models
from hallucination_bench.judges.v5c import READ_MARK

from . import NAME, bank, score
from . import run as amrun

CONTEXT_CHARS = 200


def _f(x: Any, d: int = 3) -> str:
    return "—" if x is None else f"{x:.{d}f}"


def _ci(ci: Any) -> str:
    return "—" if not ci else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def arm_dirs(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    names = [p.name for p in root.iterdir() if p.is_dir() and (p / "cache.jsonl").exists()]
    rank = {n: i for i, n in enumerate(amrun.MAP_ARM_ORDER)}
    return sorted(names, key=lambda n: (rank.get(n, len(rank)), n))


def _readouts_for(root: Path, arm: str, readouts_root: Path | None) -> Path | None:
    base = readouts_root or amrun.DEFAULT_READOUTS_ROOT
    p = base / arm / amrun.READOUTS_FILE
    if p.exists():
        return p
    aj = root / arm / "arm.json"
    if readouts_root is None and aj.exists():
        q = Path(json.loads(aj.read_text())["readouts"])
        if q.exists():
            return q
    return None


def _load_json(p: Path) -> Any:
    return json.loads(p.read_text()) if p.exists() else None


def _mentions(root: Path, arm: str) -> list[dict[str, Any]]:
    p = root / arm / "mentions.jsonl"
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


# ---------------------------------------------------------------- markdown


def _share_table(sh: dict[str, Any], rows: tuple[str, ...] | list[str]) -> list[str]:
    out = [
        f"(n = {sh['n']} spans, {sh['n_items']} items)",
        "",
        "| row | count | pooled | 95% CI | item-macro |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        x = sh["rows"].get(r)
        if x is None:
            continue
        out.append(
            f"| {r} | {x['count']} | {_f(x['pooled'])} | {_ci(x['ci95'])} | {_f(x['macro'])} |"
        )
    return out


def _rate_line(name: str, r: dict[str, Any]) -> str:
    return (
        f"- {name}: **{_f(r['value'])}** {_ci(r['ci95'])} (item-macro {_f(r['macro'])}, "
        f"n = {r['n']} spans / {r['n_items']} items)"
    )


def _counts(d: dict[str, Any]) -> str:
    return ", ".join(f"{k}: {v}" for k, v in d.items()) if d else "—"


def _detection_rows(name: str, s: dict[str, Any]) -> list[str]:
    layers = [("all", s["overall"]), *s["by_layer"].items()]
    jl = "jlens" in s["overall"]["detection"]
    head = (
        "| layer | cells | empty | pending | failed | detected | rate | mentions | spans | m/cell |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    if jl:
        head += " bag-detected | lost_in_summary |"
        sep += "---|---|"
    out = [head, sep]
    for la, b in layers:
        d = b["detection"]
        row = (
            f"| {la} | {d['n_cells']} | {d['n_empty']} | {d['n_pending']} | {d['n_failed']} | "
            f"{d['n_detected']} | {_f(d['detected_rate'])} | {d['n_mentions']} | {d['n_spans']} | "
            f"{_f(d['mentions_per_detected_cell'], 2)} |"
        )
        if jl:
            j = d["jlens"]
            row += f" {j['n_bag_detected']} | {j['n_lost_in_summary']} |"
        out.append(row)
    return out


def _arm_section(
    name: str, s: dict[str, Any], tiers: dict[str, Any] | None, gates: dict[str, Any] | None
) -> list[str]:
    o = s["overall"]
    det, hl, km = o["detection"], o["headline"], o["kind_match"]
    out = [f"## {name} ({s['config']['kind']}, layers {s['config']['judged_layers']})", ""]
    h = s["health"]
    out += [
        f"Spend so far: ${s['spend']['total_usd']:.2f}. Health: failed detect "
        f"{h['n_failed_detect']}, summary {h['n_failed_summary']}, dsum {h['n_failed_dsum']}, "
        f"map {h['n_failed_map']}; pending detect {h['n_pending_detect']}, map "
        f"{h['n_pending_map']}; accepted partial {h['n_accepted_partial']}; dropped mentions "
        f"unlocated {h['n_unlocated']} / bad enum {h['n_bad_enum']}; persona remaps "
        f"{h['n_actor_remapped']}; map unquoted contradictions "
        f"{h['map_n_unquoted_contradictions']} / support {h['map_n_unquoted_support']}.",
        "",
        "### 1. Detection",
        "",
        *_detection_rows(name, s),
        "",
        f"By kind (mentions): {_counts(det['by_kind'])}",
        "",
    ]
    if "jlens" in det:
        lis = (gates or {}).get(name, {}).get("lost_in_summary")
        label = (
            f"audited: `:Dsum` missed a real mention in {_f(lis.get('dsum_missed_share'))} of the "
            "sample"
            if lis
            else "**upper bound** (the lost_in_summary audit has not been read)"
        )
        out += [f"`lost_in_summary` = {det['jlens']['n_lost_in_summary']}: {label}.", ""]
    out += [
        "| item | detected cells | mentions |",
        "|---|---|---|",
        *(f"| {i} | {v['detected']} | {v['mentions']} |" for i, v in det["by_item"].items()),
        "",
        "### 2. Verdict shares (headline spans)",
        "",
        *_share_table(hl["verdict"], score.VERDICT_ROWS),
        "",
        "Leaves (the three hallucination components, model_state, unverifiable):",
        "",
        *_share_table(hl["leaf"], (*score.LEAF_ROWS, "other")),
        "",
        f"- correct by support region: {_counts(hl['correct_by_region'])}",
        f"- hallucination by contradiction region: {_counts(hl['hallucination_by_region'])} "
        "(`prompt_before` = clear confabulation; `prompt_after` / `response` = predicts ahead, "
        "wrongly; `response` is weak evidence: the rollout is 1 of 8 samples)",
        f"- probe candidates by region: {_counts(hl['probe_by_region'])}",
        f"- hedged (rev 19 remaps to unverifiable): {_counts(hl['hedged'])}",
        f"- pending (not yet mapped): {hl['n_pending']}",
        "",
        "Sensitivity, the judge's pre-remap `raw_leaf`:",
        "",
        *_share_table(hl["sensitivity_raw_leaf"], score.VERDICT_ROWS),
        "",
        "Populations (span-level verdict shares, pooled; mention counts descriptive):",
        "",
        "| population | mentions | spans | " + " | ".join(score.VERDICT_ROWS) + " |",
        "|---|---|---|" + "---|" * len(score.VERDICT_ROWS),
    ]
    for pn, p in o["populations"].items():
        vals = " | ".join(_f(p["verdict"]["rows"][v]["pooled"]) for v in score.VERDICT_ROWS)
        out.append(f"| {pn} | {p['n_mentions']} | {p['n_spans']} | {vals} |")
    out += ["", f"Stance x actor (mentions on misaligned items): {_counts(o['stance_actor'])}", ""]
    t = (tiers or {}).get(name)
    corr = (
        f"audit-corrected **{_f(t.get('corrected_unrelated_rate'))}**"
        + (" (a bound: the tier matrix is sparse)" if t.get("is_bound") else "")
        if t
        else "**uncorrected** (no `audit/tiers.json`)"
    )
    out += [
        "### 3. Kind match x verdict (headline spans)",
        "",
        "| tier | " + " | ".join(score.VERDICT_ROWS) + " |",
        "|---|" + "---|" * len(score.VERDICT_ROWS),
    ]
    for tier, row in km["crosstab"].items():
        if row or tier in score.TIERS or tier == score.OM_ROW:
            out.append(
                f"| {tier} | " + " | ".join(str(row.get(v, 0)) for v in score.VERDICT_ROWS) + " |"
            )
    out += [
        "",
        _rate_line("unrelated-mention rate (raw)", km["unrelated_rate"]) + f"; {corr}",
        _rate_line(
            "unrelated rate, other_misconduct counted as unrelated",
            km["unrelated_rate_om_as_unrelated"],
        ),
        _rate_line("hallucination rate among unrelated", km["hallucination_rate_among_unrelated"]),
        _rate_line("unrealised-intent-claim rate", km["unrealised_intent_claim_rate"])
        + f"; by tier {_counts(km['unrealised_intent_by_tier'])}",
        _rate_line(
            "  controls (headline actor/stance)",
            o["kind_match_controls"]["unrealised_intent_claim_rate"],
        ),
        _rate_line(
            f"  non-acting misaligned items {o['kind_match_non_acting']['items']}",
            o["kind_match_non_acting"]["unrealised_intent_claim_rate"],
        ),
        _rate_line("real-intention rate", km["real_intention_rate"])
        + f"; by verdict {_counts(km['real_intention_by_verdict'])}",
        "",
        "Rule 4 sends own-intent claims to `model_state`, which is never checked for "
        "contradiction: the hallucination verdict does not measure confabulated intent; the "
        "unrealised-intent-claim rate is the closer measure.",
        "",
        "### 4. Inferable from text (mentions, every population, controls included)",
        "",
        "| tier | set-up | n | verdicts |",
        "|---|---|---|---|",
    ]
    for tier, d in o["inferable"].items():
        for su, x in d.items():
            out.append(f"| {tier} | {su} | {x['n']} | {_counts(x['verdicts'])} |")
    out += ["", "### 5. Top unrelated kinds (headline spans)", ""]
    for u in o["unrelated_kinds"]:
        ex = "; ".join(f"`{e['quote'][:80]}` ({e['id']} L{e['layer']})" for e in u["examples"])
        out.append(f"- **{u['kind']}** ({u['n']}): {ex}")
    c = o["controls"]
    out += [
        "",
        "### 6. Controls",
        "",
        f"Detected cells {c['n_detected_cells']}, mentions per detected cell "
        f"{_f(c['mentions_per_detected_cell'], 2)}.",
        "",
        *_share_table(c["verdict"]["verdict"], score.VERDICT_ROWS),
        "",
    ]
    if s["by_layer"]:
        out += [
            "### Per layer (headline spans)",
            "",
            "| layer | detected rate | spans | hallucination | thinky | unresolved | correct |",
            "|---|---|---|---|---|---|---|",
        ]
        for la, b in s["by_layer"].items():
            r = b["headline"]["verdict"]["rows"]
            out.append(
                f"| {la} | {_f(b['detection']['detected_rate'])} | {b['headline']['n_spans']} | "
                f"{_f(r[tx.V_HALLUCINATION]['pooled'])} | {_f(r[tx.V_THINKY]['pooled'])} | "
                f"{_f(r[tx.V_UNRESOLVED]['pooled'])} | {_f(r[tx.V_CORRECT]['pooled'])} |"
            )
        out.append("")
    return out


def audit_samples(
    mentions: list[dict[str, Any]], *, n: int = 10, seed: int = 0
) -> list[dict[str, Any]]:
    """§7: ``n`` random headline spans per verdict, with ±200 characters around the marker."""
    items = {it["id"]: it for it in bank.load_items()}
    sp = [s for s in score.spans(mentions, lambda m: m["headline"]) if score.verdict_of(s)]
    rng = random.Random(seed)
    out = []
    for v in score.VERDICT_ROWS:
        pool = sorted((s for s in sp if s["verdict"] == v), key=score.span_key)
        rng.shuffle(pool)
        for s in pool[:n]:
            p = items[s["id"]]["prompt_render"]
            ch = s["char"]
            ctx = p[max(0, ch - CONTEXT_CHARS) : ch] + READ_MARK + p[ch : ch + CONTEXT_CHARS]
            out.append(
                {
                    **{k: s[k] for k in ("arm", "id", "layer", "pos", "verdict", "leaf")},
                    "quote": s["quote"],
                    "kind": s["kind"],
                    "tier": s["tier"],
                    "why": s.get("why"),
                    "context": ctx,
                }
            )
    return out


def build(
    summaries: dict[str, dict[str, Any]],
    samples: dict[str, list[dict[str, Any]]],
    judge_model: str,
    *,
    gates: dict[str, Any] | None,
    tiers: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    md = [
        f"# {NAME} report — judge {judge_model}",
        "",
        "**PROVISIONAL**: there is no held-out gate for this eval (the claim map's gate is chat "
        "gold). Every headline number counts **spans** (distinct `(cell, readout, start, end)`); "
        "headline = actor ai_self/unspecified x stance considers_or_plans/does_or_did on "
        "misaligned items. Mention-level counts are descriptive only.",
        "",
        "| arm | detected rate | headline spans | "
        + " | ".join(score.VERDICT_ROWS[:-1])
        + " | unrelated rate | unrealised intent | real intention |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in summaries.items():
        o = s["overall"]
        r = o["headline"]["verdict"]["rows"]
        km = o["kind_match"]
        md.append(
            f"| {name} | {_f(o['detection']['detected_rate'])} | {o['headline']['n_spans']} | "
            + " | ".join(_f(r[v]["pooled"]) for v in score.VERDICT_ROWS[:-1])
            + f" | {_f(km['unrelated_rate']['value'])}"
            + ("" if (tiers or {}).get(name) else " (uncorrected)")
            + f" | {_f(km['unrealised_intent_claim_rate']['value'])} | "
            f"{_f(km['real_intention_rate']['value'])} |"
        )
    md.append("")
    if gates:
        md += ["## Detector audit gates (`audit/gates.json`)", ""]
        for name, g in gates.items():
            res = {k: v.get("pass") for k, v in g.items() if isinstance(v, dict) and "pass" in v}
            md.append(f"- {name}: {_counts(res)}")
        md.append("")
    for name, s in summaries.items():
        md += _arm_section(name, s, tiers, gates)
    md += ["## 7. Audit samples (headline spans per verdict)", ""]
    for name, rows in samples.items():
        md += [f"### {name}", ""]
        for x in rows:
            md += [
                f"- **{x['verdict']}** / {x['leaf']} — {x['id']} L{x['layer']} p{x['pos']} "
                f"[{x['kind']}, {x['tier']}]: `{x['quote']}` — why: {x['why'] or '—'}",
                f"  context: {json.dumps(x['context'], ensure_ascii=False)}",
            ]
        md.append("")
    js = {"arms": summaries, "audit_samples": samples, "gates": gates, "tiers": tiers}
    return "\n".join(md) + "\n", js


def report(
    out_root: Path,
    judge_model: str,
    *,
    readouts_root: Path | None = None,
    rebuild: bool = True,
    audit_n: int = 10,
    seed: int = 0,
    n_boot: int = score.N_BOOT,
) -> Path:
    models = Models(judge=judge_model, eval=NAME)
    root = amrun.model_root(out_root, models)
    summaries: dict[str, dict[str, Any]] = {}
    for arm in arm_dirs(root):
        if rebuild:
            p = _readouts_for(root, arm, readouts_root)
            if p is None:
                print(f"[{arm}] no readouts file found: skipped")
                continue
            summaries[arm] = amrun.rebuild(arm, p, out_root, models, n_boot=n_boot)
        else:
            s = _load_json(root / arm / "summary.json")
            if s is not None:
                summaries[arm] = s
    samples = {a: audit_samples(_mentions(root, a), n=audit_n, seed=seed) for a in summaries}
    gates = _load_json(root / "audit" / "gates.json")
    tiers = _load_json(root / "audit" / "tiers.json")
    md, js = build(summaries, samples, judge_model, gates=gates, tiers=tiers)
    root.mkdir(parents=True, exist_ok=True)
    (root / "report.md").write_text(md, encoding="utf-8")
    (root / "report.json").write_text(json.dumps(js, indent=1, ensure_ascii=False) + "\n")
    return root
