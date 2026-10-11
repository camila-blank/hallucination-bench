"""``hbench report --view v2``: the claim-map (taxonomy v2) report, substitution pairs, audit
samples, and the v2 pilot dump (plan 0002, §Run and §Metrics).

Writes, under ``outputs/wsbench_chat/<model>/``: ``report_v2.md``, ``report_v2.json``,
``substitution_pairs.jsonl`` and ``audit_samples_v2.md``. Every arm's ``cells.jsonl`` /
``summary_v2.json`` is first regenerated from its cache (no API calls), as in the v1 report.
Prose and token arms get separate tables.
"""

import json
import random
from pathlib import Path
from typing import Any

from hallucination_bench import taxonomy_v2 as tx
from hallucination_bench.evals.wsbench_chat import bank
from hallucination_bench.evals.wsbench_chat.run import Models, rebuild_arm
from hallucination_bench.evals.wsbench_chat.score_v2 import AGREEMENT, FLAGS, GROUP_ROWS
from hallucination_bench.report import _context, arm_dirs, iter_cells, load_arm_json, model_root

FLAG_TITLES: dict[str, str] = {
    "hallucination": "hallucination (v2)",
    "hallucination_excl_unknown": "halluc. excl. unknown_incorrect",
    "hallucination_alt": "halluc. disputes re-scored",
    "false_claim": "false_claim",
    "cruxy_false_claim": "cruxy false_claim",
    "detail": "detail",
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
    return [n, str(b[count]), *(_p(b[key][g], ci) for g in GROUP_ROWS)]


def _rates(b: dict[str, Any], f: str) -> str:
    return f"{_p(b['rates'][f]['all'], False)} / {_p(b['rates'][f]['specific'], False)}"


def _blocks(arms: dict[str, dict[str, Any]]) -> tuple[list, list]:
    overall = [(a, s["overall"]) for a, s in sorted(arms.items())]
    layered = [
        (f"{a} L{la}", b) for a, s in sorted(arms.items()) for la, b in s["by_layer"].items()
    ]
    return overall, layered


def build(summaries: dict[str, dict[str, Any]], judge_model: str) -> tuple[str, dict[str, Any]]:
    prose = {a: s for a, s in summaries.items() if s["config"]["kind"] == "prose"}
    tokens = {a: s for a, s in summaries.items() if s["config"]["kind"] == "tokens"}
    md: list[str] = [
        f"# wsbench_chat claim map (taxonomy v2) — judge {judge_model}",
        "",
        "Shares in % of readout characters (per-readout shares averaged over readouts; each "
        "readout weighs the same); item-bootstrap 95% CI in brackets (2,000 draws, seed 0). The "
        "7 groups + `unknown_incorrect` (established v5c spans the judge did not label "
        "incorrect, or disputed) + `failed` sum to 100. Readout-level rates are over readouts "
        "of successfully mapped cells: `all` = every non-empty readout, `spec` = v5c-specific "
        "readouts (plan 0001's denominator).",
        "",
    ]
    group_head = ["arm", "n readouts", *GROUP_ROWS]
    for title, arms in (("Prose arms", prose), ("Token arms (judged via the summariser)", tokens)):
        if not arms:
            continue
        overall, layered = _blocks(arms)
        md += [f"## {title}", ""]
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
            ["arm", "n all / spec", *(FLAG_TITLES[f] for f in FLAGS)],
            [
                [
                    n,
                    f"{b['n_rate_readouts']['all']} / {b['n_rate_readouts']['specific']}",
                    *(_rates(b, f) for f in FLAGS),
                ]
                for n, b in overall
            ],
        )
        md += _table(
            "v5c vs v2 on the v5c-specific denominator (same readouts)",
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
            "unknown_incorrect > co_activation > thinky > unsettled > correct > junk > non_claim)",
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
            "map spend USD",
        ],
        hrows,
    )
    md += [f"Total `:M` spend (all cached calls): ${total:.2f}", ""]
    return "\n".join(md), {"judge_model": judge_model, "arms": summaries, "total_map_usd": total}


def _segments(rec: dict[str, Any]) -> list[dict[str, Any]]:
    m = rec.get("map")
    return m["segments"] if m and rec.get("stages", {}).get("map") in ("ok", "failed") else []


def substitution_pairs(dirs: list[Path]) -> list[dict[str, Any]]:
    out = []
    for d in dirs:
        for rec in iter_cells(d):
            for s in _segments(rec):
                if s["raw_leaf"] in tx.SUBSTITUTION_LEAVES:
                    out.append(
                        {
                            "arm": d.name,
                            "key": rec["key"],
                            "layer": rec["layer"],
                            "unit": s["unit"],
                            "text": s["text"],
                            "leaf": s["leaf"],
                            "raw_leaf": s["raw_leaf"],
                            "status": s["status"],
                            "established": s["established"],
                            "cruxy": s["cruxy"],
                            "pair": s["pair"],
                            "why": s["why"],
                        }
                    )
    return out


def audit_samples(dirs: list[Path], *, n: int = 10, seed: int = 0) -> str:
    """``n`` random segments per leaf (incl. the pseudo-leaves) across arms, in context."""
    rng = random.Random(seed)
    res: dict[str, list[tuple[str, dict, dict]]] = {}
    seen: dict[str, int] = {}
    for d in dirs:
        for rec in iter_cells(d):
            for s in _segments(rec):
                lf = s["leaf"]
                seen[lf] = seen.get(lf, 0) + 1
                pool = res.setdefault(lf, [])
                if len(pool) < n:
                    pool.append((d.name, rec, s))
                else:
                    j = rng.randrange(seen[lf])
                    if j < n:
                        pool[j] = (d.name, rec, s)
    out = ["# Audit samples (claim map v2)", "", f"{n} random segments per leaf, seed {seed}.", ""]
    for lf in tx.ALL_LEAVES:
        rows = res.get(lf, [])
        out += [f"## {lf} (group {tx.group_of(lf)}; {seen.get(lf, 0)} segments in total)", ""]
        for arm, rec, s in rows:
            prompt, ctx = _context(rec)
            readout = (rec.get("samples") or [""])[s["readout"]]
            tags = [k for k in ("hedged", "world_false", "cruxy", "established") if s.get(k)]
            out += [
                f"### {arm} {rec['key']} {s['unit']} ({s['kind']}, status {s['status']})",
                "",
                f"- segment: {json.dumps(s['text'], ensure_ascii=False)}",
                f"- raw leaf {s['raw_leaf']}; alt_leaf {s['alt_leaf']}; tags {tags}; "
                f"pair {s['pair']}",
                f"- why: {s['why']}",
                f"- prompt: {json.dumps(prompt[:500], ensure_ascii=False)}",
                f"- response around the marker: {json.dumps(ctx, ensure_ascii=False)}",
                f"- readout {s['readout']}: {json.dumps(readout[:1500], ensure_ascii=False)}",
                "",
            ]
    return "\n".join(out)


def report(
    out_root: Path, judge_model: str, *, rebuild: bool = True, audit_n: int = 10, seed: int = 0
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
            )
            rebuild_arm(d.name, Path(a["readouts"]), out_root, models)
        p = d / "summary_v2.json"
        if p.exists():
            summaries[d.name] = json.loads(p.read_text())
    if not summaries:
        raise SystemExit(f"no arm under {root} has claim-map (:M) results")
    v2_dirs = [d for d in dirs if d.name in summaries]
    md, js = build(summaries, judge_model)
    (root / "report_v2.md").write_text(md + "\n")
    (root / "report_v2.json").write_text(json.dumps(js, indent=1) + "\n")
    with (root / "substitution_pairs.jsonl").open("w", encoding="utf-8") as fh:
        for row in substitution_pairs(v2_dirs):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (root / "audit_samples_v2.md").write_text(audit_samples(v2_dirs, n=audit_n, seed=seed) + "\n")
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
