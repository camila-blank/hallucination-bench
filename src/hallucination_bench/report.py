"""Cross-arm report (markdown + JSON), the audit sample, and the pilot span dump.

``hbench report`` first regenerates every arm's ``cells.jsonl`` / ``summary.json`` from its
cache (no API calls), then writes ``outputs/wsbench_chat/<model>/report.{md,json}`` and
``audit_samples.md``. Prose and token arms go in separate tables: token arms are judged through
the ``interp-v1`` summariser, which is told to "commit to the most specific reading", so their
unsupported / strict / lenient rates are not ranked beside the prose arms.
"""

import json
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from hallucination_bench.evals.wsbench_chat import NAME, bank
from hallucination_bench.evals.wsbench_chat.run import STAGES, Models, model_slug, rebuild_arm
from hallucination_bench.judges.v5c import SPECIFIC_CLASSES

CONTRA_COLS: tuple[tuple[str, str], ...] = (
    ("v5c rate", "hallucination_rate"),
    ("concept", "concept_rate"),
    ("detail", "detail_rate"),
    ("unclassified", "unclassified_only_rate"),
    ("major", "major_rate"),
    ("mean max sev", "mean_max_severity"),
    ("off-topic", "off_topic_rate"),
    ("assert share", "assert_share"),
)
STRICT_COLS: tuple[tuple[str, str], ...] = (
    ("strict rate", "strict_hallucination_rate"),
    ("lenient rate", "lenient_hallucination_rate"),
    ("strict concept", "strict_concept_rate"),
    ("strict detail", "strict_detail_rate"),
    ("unsupported", "unsupported_rate"),
    ("unsupported (lenient)", "unsupported_rate_lenient"),
    ("disputed", "disputed_rate"),
    ("verify coverage", "verify_coverage"),
)


def model_root(out_root: Path, judge_model: str) -> Path:
    return Path(out_root) / NAME / model_slug(judge_model)


def arm_dirs(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.glob("*/arm.json"))


def _fmt(v: Any, ci: Any = None, pct: bool = True) -> str:
    if v is None:
        return "—"
    s = f"{100 * v:.1f}" if pct else f"{v:.2f}"
    if ci:
        s += f" [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]" if pct else f" [{ci[0]:.2f}, {ci[1]:.2f}]"
    return s


def _row(name: str, blk: dict[str, Any], cols: tuple[tuple[str, str], ...]) -> str:
    cells = []
    for _label, key in cols:
        pct = key != "mean_max_severity"
        cells.append(_fmt(blk["rates"].get(key), blk["ci95"].get(key), pct=pct))
    n = blk["counts"]["n_specific"]
    return f"| {name} | {n} | " + " | ".join(cells) + " |"


def _table(
    title: str, rows: list[tuple[str, dict]], cols: tuple[tuple[str, str], ...]
) -> list[str]:
    head = "| arm | n specific | " + " | ".join(c[0] for c in cols) + " |"
    sep = "|---" * (len(cols) + 2) + "|"
    return [f"### {title}", "", head, sep, *(_row(n, b, cols) for n, b in rows), ""]


def build(summaries: dict[str, dict[str, Any]], judge_model: str) -> tuple[str, dict[str, Any]]:
    prose = {a: s for a, s in summaries.items() if s["config"]["kind"] == "prose"}
    tokens = {a: s for a, s in summaries.items() if s["config"]["kind"] == "tokens"}
    md: list[str] = [
        f"# wsbench_chat report — judge {judge_model}",
        "",
        "Rates in %, item-bootstrap 95% CI in brackets (2,000 draws, seed 0). Readout-level "
        "rates over specific readouts; strict / lenient / unsupported / disputed over "
        "verify-tallied specific readouts (coverage shown).",
        "",
    ]
    for group, arms in (
        ("Prose arms", prose),
        ("Token arms (judged through the summariser)", tokens),
    ):
        if not arms:
            continue
        md += [f"## {group}", ""]
        overall = [(a, s["overall"]) for a, s in sorted(arms.items())]
        layered = [
            (f"{a} L{la}", b) for a, s in sorted(arms.items()) for la, b in s["by_layer"].items()
        ]
        md += _table("Contradicted spans, per arm", overall, CONTRA_COLS)
        md += _table("Contradicted spans, per arm x layer", layered, CONTRA_COLS)
        if arms is tokens:
            md += [
                "> Caveat: the summariser is told to commit to the most specific reading, so it "
                "produces unsupported specifics by design. The strict / unsupported rates below "
                "describe the summary, are not comparable to the prose arms and are not ranked.",
                "",
            ]
        md += _table("Strict / lenient view, per arm", overall, STRICT_COLS)
        md += _table("Strict / lenient view, per arm x layer", layered, STRICT_COLS)
    # span shares
    md += ["## Contradicted spans: type and target mix (span-level shares)", ""]
    md += [
        "| arm | n spans | concept | detail | unknown | target mix |",
        "|---|---|---|---|---|---|",
    ]
    for a, s in sorted(summaries.items()):
        g = s["spans"]["by_grounding"]["contradicted"]
        t = g["type"]

        def share(k: str, t: dict = t) -> str:
            return _fmt((t.get(k) or {}).get("share"))

        mix = ", ".join(
            f"{k} {100 * v['share']:.0f}%"
            for k, v in sorted(g["target"].items(), key=lambda x: -x[1]["n"])
        )
        shares = " | ".join(share(k) for k in ("concept", "detail", "unknown"))
        md.append(f"| {a} | {g['n']} | {shares} | {mix} |")
    md += ["", "## Judge health and spend", ""]
    md += [
        "| arm | cells | missing | empty | pending | failed summ/span/verify/classify | "
        "unclassified spans | unverified spans | spend USD |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    total = 0.0
    for a, s in sorted(summaries.items()):
        h = s["health"]
        fr = h["failed_rate_by_stage"]
        failed = "/".join(
            f"{h[f'n_failed_{st}']} ({_fmt(fr[st])}%)" if fr[st] is not None else "—"
            for st in STAGES
        )
        usd = s["spend"]["total_usd"]
        total += usd
        md.append(
            f"| {a} | {h['n_cells']} | {h['n_missing']} | {h['n_empty_cells']} | "
            f"{h['n_pending_cells']} | {failed} | {h['n_unclassified']} | "
            f"{h['n_unverified_spans']} | {usd:.2f} |"
        )
    md += ["", f"Total spend (all cached calls): ${total:.2f}", ""]
    js = {"judge_model": judge_model, "arms": summaries, "total_usd": total}
    return "\n".join(md), js


def iter_cells(d: Path) -> Iterator[dict[str, Any]]:
    p = d / "cells.jsonl"
    if not p.exists():
        return
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _context(rec: dict[str, Any], width: int = 400) -> tuple[str, str]:
    item = next(it for it in bank.load_items() if it["id"] == rec["id"])
    site = next(s for s in item["sites"] if int(s["pos"]) == rec["pos"])
    mr = bank.marked_response(item, site)
    i = mr.index("⟦READ⟧")
    return item["prompt"], mr[max(0, i - width) : i + width]


def audit_samples(dirs: list[Path], *, n: int = 15, seed: int = 0) -> str:
    """``n`` random spans per (type x grounding) across arms (reservoir sampling, streaming)."""
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[tuple[str, dict, dict]]] = {}
    seen: dict[tuple[str, str], int] = {}
    for d in dirs:
        for rec in iter_cells(d):
            for sp in rec.get("spans") or []:
                k = (sp["type"], sp["grounding"])
                seen[k] = seen.get(k, 0) + 1
                res = strata.setdefault(k, [])
                if len(res) < n:
                    res.append((d.name, rec, sp))
                else:
                    j = rng.randrange(seen[k])
                    if j < n:
                        res[j] = (d.name, rec, sp)
    out = [
        "# Audit samples",
        "",
        f"{n} random spans per (type x grounding) across arms, seed {seed}.",
        "",
    ]
    for (typ, g), rows in sorted(strata.items()):
        out += [f"## type={typ}, grounding={g} ({seen[(typ, g)]} spans in total)", ""]
        for arm, rec, sp in rows:
            prompt, ctx = _context(rec)
            readout = (rec.get("samples") or [""])[sp["readout_idx"]]
            out += [
                f"### {arm} {rec['key']} {sp['span_id']}",
                "",
                f"- span: {json.dumps(sp['text'], ensure_ascii=False)}",
                f"- labels: type={sp['type']} target={sp['target']} severity={sp['severity']} "
                f"unsupported_kind={sp['unsupported_kind']}",
                f"- why: {sp['why']}",
                f"- prompt: {json.dumps(prompt[:500], ensure_ascii=False)}",
                f"- response around the marker: {json.dumps(ctx, ensure_ascii=False)}",
                f"- readout {sp['readout_idx']}: {json.dumps(readout[:1500], ensure_ascii=False)}",
                "",
            ]
    return "\n".join(out)


def load_arm_json(d: Path) -> dict[str, Any]:
    return json.loads((d / "arm.json").read_text())


def report(
    out_root: Path, judge_model: str, *, rebuild: bool = True, audit_n: int = 15, seed: int = 0
) -> Path:
    root = model_root(out_root, judge_model)
    dirs = arm_dirs(root)
    if not dirs:
        raise SystemExit(f"no arms under {root}")
    summaries: dict[str, dict[str, Any]] = {}
    for d in dirs:
        a = load_arm_json(d)
        models = Models(
            judge=a["judge"],
            summarizer=a["summarizer"],
            classifier=a["classifier"],
            effort=a["effort"],
            map_model=a.get("map_model"),
            map_anchor=a.get("map_anchor", "v5c"),  # arms written before rev 13 were anchored
        )
        if rebuild:
            summaries[d.name] = rebuild_arm(d.name, Path(a["readouts"]), out_root, models)
        else:
            summaries[d.name] = json.loads((d / "summary.json").read_text())
    md, js = build(summaries, judge_model)
    (root / "report.md").write_text(md + "\n")
    (root / "report.json").write_text(json.dumps(js, indent=1) + "\n")
    (root / "audit_samples.md").write_text(audit_samples(dirs, n=audit_n, seed=seed) + "\n")
    return root


# ---------------------------------------------------------------- pilot dump


def pilot_dump(
    out_root: Path,
    judge_model: str,
    arms: list[str],
    *,
    per_stratum: int,
    seed: int = 0,
    layers: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Sample contradicted / disputed / unsupported spans from a pilot (span + verify) run,
    stratified by arm and grounding, as calibration cases (``part = "real"``, gold ``null``).
    A case holds the whole cell (all its spans, specific readouts only, original indices kept
    in ``source``) so the classifier prompt matches production; ``sampled`` marks the spans that
    were drawn."""
    rng = random.Random(seed)
    root = model_root(out_root, judge_model)
    cases: list[dict[str, Any]] = []
    for arm in arms:
        recs = [
            r
            for r in iter_cells(root / arm)
            if r["status"] == "judged" and r["spans"] and (not layers or r["layer"] in layers)
        ]
        by_g: dict[str, list[tuple[int, int]]] = {}
        for ri, r in enumerate(recs):
            for si, sp in enumerate(r["spans"]):
                by_g.setdefault(sp["grounding"], []).append((ri, si))
        picked: dict[int, set[int]] = {}
        for _g, pool in sorted(by_g.items()):
            for ri, si in rng.sample(pool, min(per_stratum, len(pool))):
                picked.setdefault(ri, set()).add(si)
        for ri, sis in sorted(picked.items()):
            r = recs[ri]
            item = next(it for it in bank.load_items() if it["id"] == r["id"])
            site = next(s for s in item["sites"] if int(s["pos"]) == r["pos"])
            idxs = [v["idx"] for v in r["verdict"] if v["class"] in SPECIFIC_CLASSES]
            remap = {old: new for new, old in enumerate(idxs)}
            cases.append(
                {
                    "case_id": f"real-{arm}-{r['key']}",
                    "part": "real",
                    "rule": None,
                    "source": {"arm": arm, "key": r["key"], "readout_idx": idxs},
                    "prompt": item["prompt"],
                    "response": bank.marked_response(item, site),
                    "readouts": [r["samples"][i] for i in idxs],
                    "spans": [
                        {
                            "readout_idx": remap[sp["readout_idx"]],
                            "text": sp["text"],
                            "grounding": sp["grounding"],
                            "sampled": si in sis,
                            "gold": None,
                        }
                        for si, sp in enumerate(r["spans"])
                    ],
                }
            )
    return cases
