"""``hbench report --view v2``: the claim-map (taxonomy v2) report, probe candidates, audit
samples, and the v2 pilot dump (plan 0002, §Run and §Metrics; rev 13 verdict layer).

Writes, under ``outputs/wsbench_chat/<model>/``: ``report_v2.md``, ``report_v2.json``,
``probe_candidates.jsonl`` (one row per probe-candidate segment; replaces
``substitution_pairs.jsonl``) and ``audit_samples_v2.md`` (``verdict_n`` per verdict, then
``audit_n`` per leaf). Every arm's ``cells.jsonl`` / ``summary_v2.json`` is first regenerated
from its cache (no API calls), as in the v1 report. Verdict shares and readout-level verdict
rates are the headline; groups and leaves are the breakdown. Prose and token arms get separate
tables. Until the held-out gate passes for the current prompt version, every output is
labelled **pre-gate**.
"""

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from hallucination_bench import taxonomy_v2 as tx
from hallucination_bench.evals.wsbench_chat import bank, score_v2
from hallucination_bench.evals.wsbench_chat.run import Models, rebuild_arm
from hallucination_bench.evals.wsbench_chat.score_v2 import (
    AGREEMENT,
    FLAGS,
    GROUP_ROWS,
    VERDICT_FLAGS,
    VERDICT_ROWS,
)
from hallucination_bench.report import _context, arm_dirs, iter_cells, load_arm_json, model_root

PRE_GATE_NOTE = (
    "**PRE-GATE**: the held-out calibration gate has not passed for this prompt version; "
    "these numbers are provisional."
)
FLAG_TITLES: dict[str, str] = {
    **{f"verdict_{v}": v for v in VERDICT_ROWS},
    "verdict_junk_only": "all junk",
    "hallucination": "hallucination (v2)",
    "hallucination_excl_unknown": "halluc. excl. unknown_incorrect",
    "hallucination_alt": "halluc. disputes re-scored",
    "false_claim": "false_claim",
    "cruxy_false_claim": "cruxy false_claim",
    "detail": "detail",
    "substitution_category": "substitution_category",
    "co_activation": "co-activation",
    "thinky": "thinky",
    "correct": "correct",
    "nonclaim_or_junk_only": "all non-claim / junk",
}


def _p(r: dict[str, Any] | None, ci: bool = True) -> str:
    if not r or r.get("value") is None:
        return "—"
    s = f"{100 * r['value']:.1f}"
    if ci and r.get("ci95"):
        lo, hi = r["ci95"]
        s += f" [{100 * lo:.1f}, {100 * hi:.1f}]"
    return s


def _table(title: str, head: list[str], rows: list[list[str]]) -> list[str]:
    return [
        f"### {title}",
        "",
        "| " + " | ".join(head) + " |",
        "|---" * len(head) + "|",
        *("| " + " | ".join(r) + " |" for r in rows),
        "",
    ]


def _grow(n: str, b: dict[str, Any], key: str, ci: bool = True, count: str = "n_share_readouts"):
    return [n, str(b[count]), *(_p(v, ci) for v in b[key].values())]


def _rates(b: dict[str, Any], f: str) -> str:
    return f"{_p(b['rates'][f]['all'], False)} / {_p(b['rates'][f]['specific'], False)}"


def _blocks(arms: dict[str, dict[str, Any]]) -> tuple[list, list]:
    overall = [(a, s["overall"]) for a, s in sorted(arms.items())]
    layered = [
        (f"{a} L{la}", b) for a, s in sorted(arms.items()) for la, b in s["by_layer"].items()
    ]
    return overall, layered


# rev 16: false_claim and detail are always reported next to the hallucination verdict
SPLIT_LEAVES: tuple[str, ...] = ("false_claim", "detail", "substitution_category")  # rev 20


def _vhead() -> list[str]:
    out = []
    for v in VERDICT_ROWS:
        out.append(v)
        if v == tx.V_HALLUCINATION:
            out += [f"↳ {lf}" for lf in SPLIT_LEAVES]
    return out


def _vshare_row(n: str, b: dict[str, Any]) -> list[str]:
    out = [n, str(b["n_share_readouts"])]
    for v in VERDICT_ROWS:
        out.append(_p(b["char_share_verdict"][v]))
        if v == tx.V_HALLUCINATION:
            out += [_p(b["char_share_leaf"][lf]) for lf in SPLIT_LEAVES]
    return out


def _vrate_head() -> list[str]:
    out = []
    for f in VERDICT_FLAGS:
        out.append(FLAG_TITLES[f])
        if f == f"verdict_{tx.V_HALLUCINATION}":
            out += [f"↳ {lf}" for lf in SPLIT_LEAVES]
    return out


def _vrow(n: str, b: dict[str, Any]) -> list[str]:
    out = [n, str(b["n_rate_readouts"]["all"])]
    for f in VERDICT_FLAGS:
        out.append(_p(b["rates"][f]["all"]))
        if f == f"verdict_{tx.V_HALLUCINATION}":
            out += [_p(b["rates"][lf]["all"]) for lf in SPLIT_LEAVES]
    return out


# rev 17: the "judge noise ±" column (half the r0 - r1 difference from consistency-v2)
NOISE_COLS: tuple[tuple[str, str], ...] = (
    ("hallucination", "h"),
    ("false_claim", "fc"),
    ("detail", "d"),
    ("substitution_category", "sc"),
    ("probe_candidate", "p"),
    ("correct", "c"),
    ("unresolved", "u"),
)
NOISE_HEAD = "judge noise ± (" + " / ".join(k for _, k in NOISE_COLS) + ")"


def _arm_of(row_name: str) -> str:
    return row_name.split(" L", 1)[0]


def _noise_cell(noise: dict[str, Any] | None, row_name: str, kind: str) -> str:
    """``kind`` = ``rate`` or ``share``; the arm's consistency sample, else the overall one."""
    if not noise:
        return "—"
    nz = noise["noise"]["by_arm"].get(_arm_of(row_name)) or noise["noise"]["overall"]
    parts = []
    for m, k in NOISE_COLS:
        pm = nz[m][kind]["pm"]
        parts.append(f"{k} {100 * pm:.1f}" if pm is not None else f"{k} —")
    return " / ".join(parts)


def load_noise(out_root: Path, judge_model: str) -> dict[str, Any] | None:
    """The consistency-v2 result of the current prompt version (``None`` if not run)."""
    from hallucination_bench.evals.wsbench_chat.run import model_slug
    from hallucination_bench.judges.claimmap_v2 import PROMPT_VERSION

    p = (
        Path(out_root)
        / "consistency"
        / model_slug(judge_model)
        / f"consistency_{PROMPT_VERSION}.json"
    )
    return json.loads(p.read_text()) if p.exists() else None


def noise_section(noise: dict[str, Any] | None) -> list[str]:
    from hallucination_bench import consistency_v2

    md = ["## Judge noise (consistency-v2: two fresh calls per sampled cell)", ""]
    if not noise:
        return [
            *md,
            "**Not run for this prompt version**: run `hbench consistency-v2` on a seeded "
            "150-cell sample of the run's arms; the headline tables' noise columns show —.",
            "",
        ]
    body = (
        consistency_v2.render_markdown(noise)
        .replace("\n## ", "\n### ")
        .replace("\n### ", "\n#### ")
    )
    body = body.replace("## Claim-map judge consistency", "### Claim-map judge consistency", 1)
    return [
        *md,
        f"Sample: `{noise.get('arms')}`, {noise.get('n_per_arm')} cells per arm, seed "
        f'{noise.get("seed")}. The headline tables\' "{NOISE_HEAD}" column is half the r0 - r1 '
        "difference of each metric in the arm's sample.",
        "",
        body,
        "",
    ]


def is_pre_gate(summaries: dict[str, dict[str, Any]]) -> bool:
    return any(s["config"].get("pre_gate", True) for s in summaries.values())


def build(
    summaries: dict[str, dict[str, Any]],
    judge_model: str,
    noise: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    prose = {a: s for a, s in summaries.items() if s["config"]["kind"] == "prose"}
    tokens = {a: s for a, s in summaries.items() if s["config"]["kind"] == "tokens"}
    pre_gate = is_pre_gate(summaries)
    anchors = sorted({s["config"].get("map_anchor", "v5c") for s in summaries.values()})
    md: list[str] = [
        f"# wsbench_chat claim map (taxonomy v2) — judge {judge_model}"
        + (" — PRE-GATE" if pre_gate else ""),
        "",
        *([PRE_GATE_NOTE, ""] if pre_gate else []),
        f"Mode: {', '.join('single-call' if a == 'none' else 'anchor ' + a for a in anchors)}. "
        "Shares in % of readout characters (per-readout shares averaged over readouts; each "
        "readout weighs the same); item-bootstrap 95% CI in brackets (2,000 draws, seed 0). The "
        f"{len(tx.VERDICTS)} verdicts + `failed` sum to 100, and so do the {len(tx.GROUPS)} "
        "groups + `unknown_incorrect` "
        "(anchored runs only: established v5c spans the judge did not label incorrect, or "
        "disputed) + `failed`. Readout-level rates are over readouts of successfully mapped "
        "cells: `all` = every non-empty readout, `spec` = v5c-specific readouts (plan 0001's "
        "denominator). A readout-level verdict rate is the share of readouts with at least one "
        "segment of that verdict. Rev 19: hedged option lists are scored `unverifiable` "
        "(`same_slot_alternative`), and `model_state` is the `thinky` verdict. Rev 18: `junk` "
        "merges the judge's `non_claim` and `off_topic` after parsing (the `reason` tag keeps "
        "the kind; see the junk-by-reason table).",
        "",
        "Verdicts: " + "; ".join(f"`{v}` = {', '.join(lv)}" for v, lv in tx.VERDICTS.items()),
        "",
    ]
    group_head = ["arm", "n readouts", *GROUP_ROWS]
    verdict_head = ["arm", "n readouts", *_vhead(), NOISE_HEAD]
    for title, arms in (("Prose arms", prose), ("Token arms (judged via the summariser)", tokens)):
        if not arms:
            continue
        overall, layered = _blocks(arms)
        md += [f"## {title}", ""]
        for t, rows in (("per arm", overall), ("per arm x layer", layered)):
            md += _table(
                f"Headline: character-weighted verdict shares, {t}",
                verdict_head,
                [[*_vshare_row(n, b), _noise_cell(noise, n, "share")] for n, b in rows],
            )
            md += _table(
                f"Headline: readout-level verdict rates (all readouts), {t}",
                ["arm", "n readouts", *_vrate_head(), NOISE_HEAD],
                [[*_vrow(n, b), _noise_cell(noise, n, "rate")] for n, b in rows],
            )
        md += _table(
            "v2 hallucination rate vs v5c on the same readouts",
            [
                "arm",
                "n compared / n readouts",
                "v5c rate (compared)",
                "v2 rate (compared)",
                "↳ v2 false_claim",
                "↳ v2 detail",
                "↳ v2 substitution_category",
                "n spec",
                "v5c hallucination_rate (spec)",
                "v2 rate (spec)",
            ],
            [
                [
                    n,
                    f"{b['v5c_comparison']['n_compared']} / {b['v5c_comparison']['n_readouts']}",
                    _p(b["v5c_comparison"]["v5c_rate"]),
                    _p(b["v5c_comparison"]["v2_rate"]),
                    _p(b["v5c_comparison"].get("v2_false_claim_rate")),
                    _p(b["v5c_comparison"].get("v2_detail_rate")),
                    _p(b["v5c_comparison"].get("v2_substitution_category_rate")),
                    str(b["n_rate_readouts"]["specific"]),
                    _p(b["v5c_comparison"]["v5c_rate_specific"]),
                    _p(b["v5c_comparison"]["v2_rate_specific"]),
                ]
                for n, b in layered
            ],
        )
        md += [
            "`compared` = readouts with both a v5c span verdict and a mapped cell (v5c rate = "
            "share v5c-`hallucinated`; v2 rate = share with a `hallucination`-verdict segment). "
            "`spec` = the v5c-specific subset (plan 0001's denominator).",
            "",
            "### Breakdown",
            "",
        ]
        md += _table(
            "Junk share by reason (rev 18: junk = non-claims + off-topic claims; % of readout "
            "characters, sums to the junk share)",
            ["arm", "n readouts", *score_v2.JUNK_REASON_ROWS],
            [_grow(n, b, "char_share_junk_reason") for n, b in overall],
        )
        md += _table(
            "Segment-count verdict shares (pooled over segments), per arm",
            ["arm", "n segments", *VERDICT_ROWS],
            [_grow(n, b, "seg_share_verdict", False, "n_segments") for n, b in overall],
        )
        for t, rows in (("per arm", overall), ("per arm x layer", layered)):
            md += _table(
                f"Character-weighted group shares, {t}",
                group_head,
                [_grow(n, b, "char_share_group") for n, b in rows],
            )
        md += _table(
            "Character-weighted leaf shares, per arm (no CI)",
            ["leaf", *(n for n, _ in overall)],
            [
                [lf, *(_p(b["char_share_leaf"][lf], False) for _, b in overall)]
                for lf in tx.ALL_LEAVES
            ],
        )
        md += _table(
            "Segment-count group shares (pooled over segments), per arm",
            ["arm", "n segments", *GROUP_ROWS],
            [_grow(n, b, "seg_share_group", False, "n_segments") for n, b in overall],
        )
        md += _table(
            "Readout-level rates, per arm (all / spec)",
            ["arm", "n all / spec", *(FLAG_TITLES[f] for f in FLAGS if f not in VERDICT_FLAGS)],
            [
                [
                    n,
                    f"{b['n_rate_readouts']['all']} / {b['n_rate_readouts']['specific']}",
                    *(_rates(b, f) for f in FLAGS if f not in VERDICT_FLAGS),
                ]
                for n, b in overall
            ],
        )
        md += _table(
            "Group-level comparison on the v5c-specific denominator (same readouts)",
            [
                "arm",
                "n specific",
                "v5c hallucination_rate",
                "hallucination_rate_v2",
                "v2 excl. unknown_incorrect",
                "v2, disputes re-scored",
            ],
            [
                [
                    n,
                    str(b["n_rate_readouts"]["specific"]),
                    _p(b["v5c_hallucination_rate_same_readouts"]),
                    _p(b["rates"]["hallucination"]["specific"]),
                    _p(b["rates"]["hallucination_excl_unknown"]["specific"]),
                    _p(b["rates"]["hallucination_alt"]["specific"]),
                ]
                for n, b in layered
            ],
        )
        md += _table(
            "Sensitivity: group shares with disputed segments re-scored by their alt_leaf",
            group_head,
            [_grow(n, b, "char_share_group_alt", False) for n, b in overall],
        )
        if arms is tokens:
            md += [
                "> Caveat: token arms are judged through the `interp-v1` summariser, which is told "
                "to commit to the most specific reading; their shares describe the summary and "
                "are not ranked against the prose arms.",
                "",
            ]
    md += ["## Comparisons", ""]
    for a, s in sorted(summaries.items()):
        b = s["overall"]
        tr = b["transition"]
        cols = [g for g in (*GROUP_ROWS,) if any(g in v for v in tr.values())]
        md += _table(
            f"{a}: v5c readout class x v2 readout group (first present: hallucination > "
            "unknown_incorrect > co_activation > supplementary > thinky > unsettled > correct > "
            "junk)",
            ["v5c class", *cols],
            [[k, *(str(v.get(g, 0)) for g in cols)] for k, v in tr.items()],
        )
        ag = b["agreement"]
        md += _table(
            f"{a}: verify status vs v2 leaf (majority over the quote; failed excluded)",
            ["verify", "acceptable v2 leaves", "n", "agree", "unlocated", "top v2 leaves"],
            [
                [
                    st,
                    ", ".join(sorted(AGREEMENT[st])),
                    str(row["n"]),
                    _p({"value": row["agree_rate"]}, False),
                    str(row["n_unlocated"]),
                    ", ".join(f"{k} {v}" for k, v in list(row["leaves"].items())[:5]),
                ]
                for st, row in ag.items()
            ],
        )
    md += ["## Judge health and spend (`:M` stage)", ""]
    hrows = []
    total = 0.0
    for a, s in sorted(summaries.items()):
        h = s["health"]
        usd = float(s["spend_map"].get("usd") or 0)
        total += usd
        fr = h["failed_cell_rate"]
        ur = h["unlocated_subspan_rate"]
        hrows.append(
            [
                a,
                str(h["n_cells"]),
                str(h["n_map_ok"]),
                f"{h['n_map_failed']} ({_p({'value': fr}, False)}%)",
                str(h["n_map_pending"]),
                str(h["n_units"]),
                str(h["n_segments"]),
                str(h["n_missing_units"]),
                str(h["n_duplicate_units"]),
                f"{h['n_unlocated_subspans']}/{h['n_splits']} ({_p({'value': ur}, False)}%)",
                str(h["n_established_units"]),
                str(h["n_forced_established"]),
                str(h["n_disputed_established"]),
                str(h["n_unlocated_established"]),
                str(h["n_missing_pairs"]),
                f"{h['n_missing_probe_targets']}/{h['n_probe_candidates']}",
                f"{h.get('n_unquoted_contradictions', 0)}/"
                f"{h.get('n_contradiction_quotes_checked', 0)}",
                f"{h.get('n_unquoted_support', 0)}/{h.get('n_support_quotes_checked', 0)}",
                f"{h.get('n_hedged_remapped', 0)} ("
                + " / ".join(
                    str(h.get(f"n_hedged_from_{k}", 0))
                    for k in ("hallucination", "substitution", "correct", "other")
                )
                + ")",
                f"{usd:.2f}",
            ]
        )
    md += _table(
        "Health",
        [
            "arm",
            "cells",
            "ok",
            "failed",
            "pending",
            "units",
            "segments",
            "missing units",
            "dup units",
            "unlocated subspans",
            "established",
            "forced",
            "disputed",
            "unlocated established",
            "missing pairs",
            "missing probe_target",
            "unquoted contradictions",
            "unquoted support",
            "hedged remapped (from halluc. / subst. / correct / other)",
            "map spend USD",
        ],
        hrows,
    )
    md += [f"Total `:M` spend (all cached calls): ${total:.2f}", ""]
    js = {
        "judge_model": judge_model,
        "pre_gate": pre_gate,
        "arms": summaries,
        "total_map_usd": total,
    }
    return "\n".join(md), js


def probe_section(rows: list[dict[str, Any]], top: int = 15) -> list[str]:
    """Probe candidates by leaf, with each leaf's most frequent probe targets."""
    if not rows:
        return []
    by_leaf: dict[str, Counter] = {}
    for r in rows:
        by_leaf.setdefault(r["leaf"], Counter())[r["probe_target"] or "(missing)"] += 1
    md = ["## Probe candidates", "", f"{len(rows)} probe-candidate segments.", ""]
    md += _table(
        "Probe candidates by leaf",
        ["leaf", "n segments", "n missing probe_target", f"top probe targets (of {top})"],
        [
            [
                lf,
                str(sum(c.values())),
                str(c.get("(missing)", 0)),
                "; ".join(f"{t} ({n})" for t, n in c.most_common(5) if t != "(missing)"),
            ]
            for lf in tx.PROBE_CANDIDATE_LEAVES
            if (c := by_leaf.get(lf))
        ],
    )
    allc = Counter(r["probe_target"] for r in rows if r["probe_target"])
    md += _table(
        "Top probe targets overall",
        ["probe_target", "n segments"],
        [[t, str(n)] for t, n in allc.most_common(top)],
    )
    return md


def _segments(rec: dict[str, Any]) -> list[dict[str, Any]]:
    m = rec.get("map")
    return m["segments"] if m and rec.get("stages", {}).get("map") in ("ok", "failed") else []


def probe_candidates(dirs: list[Path], *, pre_gate: bool = True) -> list[dict[str, Any]]:
    """One row per probe-candidate segment (``taxonomy_v2.PROBE_CANDIDATE_LEAVES``, scored
    leaf): ``{arm, layer, item, pos, leaf, probe_target, actual|null, segment, readout}`` plus
    provenance. ``actual`` is the substitution pair's actual object."""
    out = []
    for d in dirs:
        for rec in iter_cells(d):
            for s in _segments(rec):
                if s["leaf"] not in tx.PROBE_CANDIDATE_LEAVES:
                    continue
                pair = s.get("pair") or None
                out.append(
                    {
                        "arm": d.name,
                        "layer": rec["layer"],
                        "item": rec["id"],
                        "pos": rec["pos"],
                        "leaf": s["leaf"],
                        "probe_target": s.get("probe_target"),
                        "actual": pair["actual"] if pair else None,
                        "segment": s["text"],
                        "readout": (rec.get("samples") or [""])[s["readout"]],
                        "readout_idx": s["readout"],
                        "unit": s["unit"],
                        "key": rec["key"],
                        "contradicted_by": s.get("contradicted_by"),
                        "why": s["why"],
                        "pre_gate": pre_gate,
                    }
                )
    return out


def hallucination_examples(dirs: list[Path], *, n: int = 8, seed: int = 0) -> list[str]:
    """``n`` random hallucination segments per arm, each with its ``contradicted_by`` quote."""
    out = ["## Hallucination examples (with the quote that contradicts them)", ""]
    for d in dirs:
        res, seen = _reservoir(
            [d],
            lambda s: "h" if s["leaf"] in tx.VERDICTS[tx.V_HALLUCINATION] else "x",
            n,
            random.Random(f"{seed}:{d.name}"),
        )
        rows = []
        for _arm, rec, s in res.get("h", []):
            rows.append(
                [
                    rec["key"],
                    s["leaf"],
                    json.dumps(s["text"][:160], ensure_ascii=False).replace("|", "\\|"),
                    json.dumps((s.get("contradicted_by") or "")[:200], ensure_ascii=False).replace(
                        "|", "\\|"
                    ),
                ]
            )
        out += _table(
            f"{d.name} ({seen.get('h', 0)} hallucination segments)",
            ["cell", "leaf", "segment", "contradicted_by"],
            rows,
        )
    return out


def _reservoir(
    dirs: list[Path], key: Any, n: int, rng: random.Random
) -> tuple[dict[str, list[tuple[str, dict, dict]]], dict[str, int]]:
    res: dict[str, list[tuple[str, dict, dict]]] = {}
    seen: dict[str, int] = {}
    for d in dirs:
        for rec in iter_cells(d):
            for s in _segments(rec):
                k = key(s)
                seen[k] = seen.get(k, 0) + 1
                pool = res.setdefault(k, [])
                if len(pool) < n:
                    pool.append((d.name, rec, s))
                else:
                    j = rng.randrange(seen[k])
                    if j < n:
                        pool[j] = (d.name, rec, s)
    return res, seen


def _sample_lines(arm: str, rec: dict[str, Any], s: dict[str, Any]) -> list[str]:
    prompt, ctx = _context(rec)
    readout = (rec.get("samples") or [""])[s["readout"]]
    tags = [k for k in ("hedged", "world_false", "cruxy", "established") if s.get(k)]
    return [
        f"### {arm} {rec['key']} {s['unit']} ({s['kind']}, status {s['status']})",
        "",
        f"- segment: {json.dumps(s['text'], ensure_ascii=False)}",
        f"- leaf {s['leaf']} (verdict {tx.verdict_of(s['leaf'])}); raw leaf {s['raw_leaf']}; "
        f"alt_leaf {s.get('alt_leaf')}; tags {tags}; pair {s.get('pair')}; probe_target "
        f"{json.dumps(s.get('probe_target'), ensure_ascii=False)}",
        f"- contradicted_by: {json.dumps(s.get('contradicted_by'), ensure_ascii=False)}; "
        f"supported_by: {json.dumps(s.get('supported_by'), ensure_ascii=False)}"
        + (f"; judge leaf {s['judge_leaf']} (unquoted)" if s.get("judge_leaf") else ""),
        f"- why: {s['why']}",
        f"- prompt: {json.dumps(prompt[:500], ensure_ascii=False)}",
        f"- response around the marker: {json.dumps(ctx, ensure_ascii=False)}",
        f"- readout {s['readout']}: {json.dumps(readout[:1500], ensure_ascii=False)}",
        "",
    ]


def audit_samples(
    dirs: list[Path], *, n: int = 10, seed: int = 0, verdict_n: int = 20, pre_gate: bool = True
) -> str:
    """``verdict_n`` random segments per verdict, then ``n`` per leaf (incl. the pseudo-leaves),
    across arms, in context."""
    out = [
        "# Audit samples (claim map v2)" + (" — PRE-GATE" if pre_gate else ""),
        "",
        *([PRE_GATE_NOTE, ""] if pre_gate else []),
        f"{verdict_n} random segments per verdict, then {n} per leaf; seed {seed}.",
        "",
    ]
    res, seen = _reservoir(dirs, lambda s: tx.verdict_of(s["leaf"]), verdict_n, random.Random(seed))
    out += ["# By verdict", ""]
    for v in VERDICT_ROWS:
        out += [f"## verdict {v} ({seen.get(v, 0)} segments in total)", ""]
        for arm, rec, s in res.get(v, []):
            out += _sample_lines(arm, rec, s)
    res, seen = _reservoir(dirs, lambda s: s["leaf"], n, random.Random(seed))
    out += ["# By leaf", ""]
    for lf in tx.ALL_LEAVES:
        out += [f"## {lf} (group {tx.group_of(lf)}; {seen.get(lf, 0)} segments in total)", ""]
        for arm, rec, s in res.get(lf, []):
            out += _sample_lines(arm, rec, s)
    return "\n".join(out)


def report(
    out_root: Path,
    judge_model: str,
    *,
    rebuild: bool = True,
    audit_n: int = 10,
    seed: int = 0,
    verdict_n: int = 20,
) -> Path:
    root = model_root(out_root, judge_model)
    dirs = arm_dirs(root)
    if not dirs:
        raise SystemExit(f"no arms under {root}")
    summaries: dict[str, dict[str, Any]] = {}
    for d in dirs:
        if rebuild:
            a = load_arm_json(d)
            models = Models(
                judge=a["judge"],
                summarizer=a["summarizer"],
                classifier=a["classifier"],
                effort=a["effort"],
                map_model=a.get("map_model"),
                map_anchor=a.get("map_anchor", "v5c"),
            )
            rebuild_arm(d.name, Path(a["readouts"]), out_root, models)
        p = d / "summary_v2.json"
        if p.exists():
            summaries[d.name] = json.loads(p.read_text())
    if not summaries:
        raise SystemExit(f"no arm under {root} has claim-map (:M) results")
    v2_dirs = [d for d in dirs if d.name in summaries]
    pre_gate = is_pre_gate(summaries)
    noise = load_noise(out_root, judge_model)
    md, js = build(summaries, judge_model, noise)
    probes = probe_candidates(v2_dirs, pre_gate=pre_gate)
    md = "\n".join(
        [
            md,
            *probe_section(probes),
            *hallucination_examples(v2_dirs, seed=seed),
            *noise_section(noise),
        ]
    )
    js["judge_noise"] = noise
    js["n_probe_candidates"] = len(probes)
    (root / "report_v2.md").write_text(md + "\n")
    (root / "report_v2.json").write_text(json.dumps(js, indent=1) + "\n")
    with (root / "probe_candidates.jsonl").open("w", encoding="utf-8") as fh:
        for row in probes:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    stale = root / "substitution_pairs.jsonl"  # replaced by probe_candidates.jsonl (rev 13)
    if stale.exists():
        stale.unlink()
    (root / "audit_samples_v2.md").write_text(
        audit_samples(v2_dirs, n=audit_n, seed=seed, verdict_n=verdict_n, pre_gate=pre_gate) + "\n"
    )
    return root


# ---------------------------------------------------------------- pilot dump (v2)


def pilot_dump_v2(
    out_root: Path,
    judge_model: str,
    arms: list[str],
    *,
    per_stratum: int,
    seed: int = 0,
    layers: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Real cells with production units, as unlabelled v2 calibration cases (``part = real``,
    every gold leaf ``null``), stratified by arm and v5c cell class (the cell's ``hallucinated``
    if any readout is, else ``off_topic``, ``consistent``, ``generic``). Needs the span stage in
    the cache (``cells.jsonl`` from a run or ``--rebuild``); never shows claim-map output, so
    the labeller stays blind to the classifier."""
    from hallucination_bench.calibrate_v2 import skeleton

    rng = random.Random(seed)
    root = model_root(out_root, judge_model)
    items = {it["id"]: it for it in bank.load_items()}
    cases: list[dict[str, Any]] = []
    for arm in arms:
        d = root / arm
        kind = load_arm_json(d).get("kind", "prose")
        strata: dict[str, list[dict[str, Any]]] = {}
        for r in iter_cells(d):
            if r["status"] != "judged" or (layers and r["layer"] not in layers):
                continue
            classes = {v["class"] for v in r["verdict"]}
            cls = next(
                (c for c in ("hallucinated", "off_topic", "consistent", "generic") if c in classes),
                "other",
            )
            strata.setdefault(cls, []).append(
                {
                    "key": r["key"],
                    "id": r["id"],
                    "pos": r["pos"],
                    "layer": r["layer"],
                    "samples": r["samples"],
                    "wrong": [v["wrong"] for v in r["verdict"]],
                }
            )
        for cls, pool in sorted(strata.items()):
            for r in rng.sample(pool, min(per_stratum, len(pool))):
                item = items[r["id"]]
                site = next(s for s in item["sites"] if int(s["pos"]) == r["pos"])
                case = {
                    "case_id": f"real-{arm}-{r['key']}",
                    "part": "real",
                    "rules": [],
                    "kind": kind,
                    "source": {"arm": arm, "key": r["key"], "v5c_cell_class": cls},
                    "prompt": item["prompt"],
                    "response": bank.marked_response(item, site),
                    "readouts": r["samples"],
                    "established": r["wrong"],
                    "gold": {"units": []},
                }
                cases.append(skeleton(case))
    return cases
