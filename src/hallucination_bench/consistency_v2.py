"""Judge-consistency check for the claim map (plan 0002 rev 16): two fresh ``:M`` calls per
real cell, leaf agreement between the runs.

``hbench consistency-v2 --arms s3d-rl600:44,nla-rl-L42:42,jlens:44 --n 50 --seed 0`` samples
``n`` non-empty cells per ``arm:layer`` (seeded; token arms only among cells whose summary is
cached in ``<arm>/cache.jsonl``), maps each twice in single-call mode (repeat keys ``r0`` / ``r1``
in ``outputs/consistency/<model>/cache.jsonl``; the repeat index is in the fingerprint, so each
is a fresh call), and compares the two runs **per unit** (each run's majority-character leaf over
the unit, as in calibration). Units failed in either run are left out.

Reported: Cohen's κ and raw agreement over the 12 leaves (overall, per arm and per lens family),
κ of the detail-vs-false_claim split on units both runs call hallucination, the headline metrics
(hallucination, false_claim, detail, probe, correct, unresolved: readout rate and character
share) computed from r0 and from r1 with their difference (``noise``; the report's "judge noise
±" is half of it, plan rev 17), the leaf × leaf
between-run confusion table, per-leaf self-agreement (2·both / (r0 + r1), the Dice overlap of
the two runs' units with that leaf), and the **flagged pairs**: a leaf pair swapped between runs
in more than 10% of either leaf's units (units where at least one run gives that leaf). Plan
rule: each flagged pair gets a sharper rule or is merged, reported to Camila before the full run.
"""

import asyncio
import json
import random
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hallucination_bench import summarizer
from hallucination_bench import taxonomy_v2 as tx
from hallucination_bench.cache import Cache, fingerprint
from hallucination_bench.evals.wsbench_chat import bank
from hallucination_bench.evals.wsbench_chat import run as wrun
from hallucination_bench.evals.wsbench_chat.score_v2 import majority_leaf
from hallucination_bench.judges import claimmap_v2 as cm
from hallucination_bench.llm import LLM, Usage
from hallucination_bench.readouts import Cell, decode_token
from hallucination_bench.units import Unit, split_cell

FLAG_SHARE = 0.10
REPEATS = 2
NOISE_METRICS: dict[str, frozenset[str]] = {
    "hallucination": frozenset(tx.VERDICTS[tx.V_HALLUCINATION]),
    "false_claim": frozenset({"false_claim"}),
    "detail": frozenset({"detail"}),
    "substitution_category": frozenset({"substitution_category"}),  # rev 20
    "probe_candidate": frozenset(tx.VERDICTS[tx.V_PROBE]),
    "correct": frozenset(tx.VERDICTS[tx.V_CORRECT]),
    "unresolved": frozenset(tx.VERDICTS[tx.V_UNRESOLVED]),
}


def family(arm: str) -> str:
    """Lens family of an arm name (``s3d-rl600`` -> ``s3d``, ``nla-rl-L42`` -> ``nla``)."""
    return arm.split("-", 1)[0]


def _readout(uid: str) -> int:
    return int(uid.split(".", 1)[0][1:])


def noise(cells: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Each headline metric from r0 and from r1 over the same readouts (readouts with a failed
    unit in either run are left out): readout rate (>= 1 unit with the metric's leaves) and
    character share (per readout, averaged), with ``diff = r0 - r1`` and ``pm = |diff| / 2``."""
    per: dict[str, dict[str, list[float]]] = {
        m: {"rate_r0": [], "rate_r1": [], "share_r0": [], "share_r1": []} for m in NOISE_METRICS
    }
    n_readouts = 0
    for c in cells:
        by_r: dict[int, list[dict[str, Any]]] = {}
        for u in c["units"]:
            by_r.setdefault(_readout(u["id"]), []).append(u)
        for units in by_r.values():
            if any(tx.FAILED in (c["r0"][u["id"]], c["r1"][u["id"]]) for u in units):
                continue
            total = sum(len(u["text"]) for u in units) or 1
            n_readouts += 1
            for m, leaves in NOISE_METRICS.items():
                for r in ("r0", "r1"):
                    hit = [u for u in units if c[r][u["id"]] in leaves]
                    per[m][f"rate_{r}"].append(1.0 if hit else 0.0)
                    per[m][f"share_{r}"].append(sum(len(u["text"]) for u in hit) / total)
    out: dict[str, Any] = {"n_readouts": n_readouts}
    for m, d in per.items():
        row: dict[str, Any] = {}
        for kind in ("rate", "share"):
            a, b = d[f"{kind}_r0"], d[f"{kind}_r1"]
            v0 = sum(a) / len(a) if a else None
            v1 = sum(b) / len(b) if b else None
            diff = v0 - v1 if v0 is not None and v1 is not None else None
            row[kind] = {
                "r0": v0,
                "r1": v1,
                "diff": diff,
                "pm": abs(diff) / 2 if diff is not None else None,
            }
        out[m] = row
    return out


def split_kappa(pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """κ of the hallucination split (rev 20: false_claim / detail / substitution_category) on the
    units both runs call hallucination (the key keeps its rev-17 name)."""
    hal = set(tx.VERDICTS[tx.V_HALLUCINATION])
    sub = [(a, b) for a, b in pairs if a in hal and b in hal]
    return {
        "n_units": len(sub),
        "kappa": cohen_kappa(sub),
        "raw_agreement": sum(a == b for a, b in sub) / len(sub) if sub else None,
    }


def parse_arms(spec: str) -> list[tuple[str, int]]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"--arms expects arm:layer, got {part!r}")
        a, la = part.rsplit(":", 1)
        out.append((a, int(la)))
    return out


def cell_samples(arm: wrun.Arm, cell: Cell, models: wrun.Models) -> list[str] | None:
    """The readouts the claim map judges (cache only): prose samples, or the cached summary of
    a token cell (``None`` if it is not cached)."""
    if arm.kind != "tokens":
        s = [x for x in cell.samples or () if x.strip()][: wrun.HAL_K]
        return s or None
    decoded = [decode_token(t) for t in cell.tokens or ()]
    bag = summarizer.render_bag(decoded, cell.scores)
    fp = fingerprint(
        summarizer.SUMMARIZER_PROMPT_VERSION,
        models.summ,
        models.effort,
        summarizer.INTERP_SYSTEM,
        summarizer.render_user(bag),
    )
    row = arm.cache.get(f"summ:{cell.key}", fp)
    text = summarizer.interpretation_of(row["result"]) if row else None
    return [text] if text else None


def sample_cells(
    arm: wrun.Arm, layer: int, n: int, seed: int, models: wrun.Models
) -> list[tuple[Cell, list[str]]]:
    pool = []
    for c in sorted(arm.cells, key=lambda c: c.key):
        if c.layer != layer or c.empty:
            continue
        s = cell_samples(arm, c, models)
        if s:
            pool.append((c, s))
    rng = random.Random(f"{seed}:{arm.name}:{layer}")
    return rng.sample(pool, min(n, len(pool)))


def unit_leaves(segs: Sequence[cm.Segment], units: Sequence[Unit]) -> dict[str, str]:
    by: dict[str, list[tuple[int, int, str]]] = {}
    for s in segs:
        by.setdefault(s.unit, []).append((s.start, s.end, s.leaf))
    return {u.id: majority_leaf(by.get(u.id, []), u.start, u.end) or tx.FAILED for u in units}


def cohen_kappa(pairs: Sequence[tuple[str, str]]) -> float | None:
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def agreement(pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """κ, raw agreement, confusion, per-leaf self-agreement and flagged pairs over unit pairs
    ``(leaf in r0, leaf in r1)``."""
    n = len(pairs)
    conf: Counter = Counter(pairs)
    n0, n1 = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    either = Counter()
    for a, b in pairs:
        either[a] += 1
        if b != a:
            either[b] += 1
    self_agree = {
        lf: {
            "dice": (2 * conf[(lf, lf)] / (n0[lf] + n1[lf])) if n0[lf] + n1[lf] else None,
            "n_r0": n0[lf],
            "n_r1": n1[lf],
            "n_both": conf[(lf, lf)],
        }
        for lf in tx.LEAF_NAMES
    }
    swaps: Counter = Counter()
    for (a, b), k in conf.items():
        if a != b:
            swaps[tuple(sorted((a, b)))] += k
    flagged = []
    for (a, b), k in swaps.most_common():
        sa = k / either[a] if either[a] else 0.0
        sb = k / either[b] if either[b] else 0.0
        row = {"pair": [a, b], "n_swaps": k, "share_a": sa, "share_b": sb}
        if max(sa, sb) > FLAG_SHARE:
            flagged.append(row)
    return {
        "n_units": n,
        "raw_agreement": sum(a == b for a, b in pairs) / n if n else None,
        "kappa": cohen_kappa(pairs),
        "confusion": {f"{a}|{b}": k for (a, b), k in sorted(conf.items())},
        "self_agreement": self_agree,
        "top_swaps": [
            {
                "pair": list(p),
                "n_swaps": k,
                "share_a": k / either[p[0]],
                "share_b": k / either[p[1]],
            }
            for p, k in swaps.most_common(10)
        ],
        "flagged_pairs": flagged,
    }


def cons_key(arm: str, cell_key: str, r: int) -> str:
    return f"cons:{arm}:{cell_key}:r{r}"


async def map_twice(
    work: Sequence[tuple[str, wrun.Arm, Cell, list[str]]],
    *,
    models: wrun.Models,
    cache: Cache,
    llm: Any,
    usage: Usage,
) -> list[dict[str, Any]]:
    """For each (arm, cell): two fresh single-call ``:M`` answers, parsed; invalid answers are
    retried in-run up to 2 times, then scored as failed."""
    out: list[dict[str, Any]] = []

    async def one(arm_name: str, arm: wrun.Arm, cell: Cell, samples: list[str]) -> None:
        item = arm.items[cell.id]
        site = arm.sites[cell.id][cell.pos]
        units = cm.plain_units(split_cell(samples, [[] for _ in samples])[0])
        system, user = cm.render_prompt(
            item["prompt"], bank.marked_response(item, site), units, kind=arm.kind
        )
        runs = []
        for r in range(REPEATS):
            key = cons_key(arm_name, cell.key, r)
            fp = fingerprint(cm.PROMPT_VERSION, models.mapper, models.effort, f"r{r}", system, user)
            row = cache.get(key, fp)
            raw = row["result"] if row is not None else None
            if row is None and llm is not None:
                call_usage = Usage()
                n_invalid = 0
                while True:
                    res = await llm.call(
                        model=models.mapper, system=system, user=user, schema=cm.SCHEMA
                    )
                    call_usage.add(res.usage)
                    usage.add(res.usage)
                    raw = res.result
                    if raw is None or cm.validate(raw, units):
                        break
                    n_invalid += 1
                    if n_invalid > wrun.MAP_INVALID_RETRIES:
                        break
                meta = {"stage": "consistency_v2", "usage": call_usage.to_dict()}
                if raw is not None and not cm.validate(raw, units):
                    cache.put(key, fp, {"result": None, "meta": {**meta, "raw": raw}})
                    raw = None
                else:
                    cache.put(key, fp, {"result": raw, "meta": meta})
            segs, _ = cm.parse(
                raw, units, cm.conversation_text(item["prompt"], bank.marked_response(item, site))
            )
            runs.append(unit_leaves(segs, units))
        out.append(
            {
                "arm": arm_name,
                "key": cell.key,
                "units": [{"id": u.id, "text": u.text} for u in units],
                "r0": runs[0],
                "r1": runs[1],
            }
        )

    await asyncio.gather(*(one(*w) for w in work))
    return sorted(out, key=lambda r: (r["arm"], r["key"]))


def merge_cell(c: dict[str, Any]) -> dict[str, Any]:
    """Rev 18: map the judge's ``non_claim`` / ``off_topic`` unit leaves (cells written before
    the merge) to ``junk``; already-merged cells pass through unchanged."""
    m = tx.JUDGE_TO_LEAF
    return {
        **c,
        "r0": {k: m.get(v, v) for k, v in c["r0"].items()},
        "r1": {k: m.get(v, v) for k, v in c["r1"].items()},
    }


def score(cells: Sequence[dict[str, Any]]) -> dict[str, Any]:
    cells = [merge_cell(c) for c in cells]

    def pairs(rows: Sequence[dict[str, Any]]) -> list[tuple[str, str]]:
        return [
            (r["r0"][u["id"]], r["r1"][u["id"]])
            for r in rows
            for u in r["units"]
            if tx.FAILED not in (r["r0"][u["id"]], r["r1"][u["id"]])
        ]

    n_failed = sum(
        tx.FAILED in (r["r0"][u["id"]], r["r1"][u["id"]]) for r in cells for u in r["units"]
    )
    arms = sorted({r["arm"] for r in cells})
    fams = sorted({family(a) for a in arms})
    by_arm = {a: agreement(pairs([r for r in cells if r["arm"] == a])) for a in arms}
    return {
        "prompt_version": cm.PROMPT_VERSION,
        "taxonomy_version": tx.TAXONOMY_VERSION,
        "n_cells": len(cells),
        "n_units_failed_either_run": n_failed,
        "overall": agreement(pairs(cells)),
        "by_arm": by_arm,
        "by_family": {
            f: agreement(pairs([r for r in cells if family(r["arm"]) == f])) for f in fams
        },
        "detail_vs_false_claim": {
            "overall": split_kappa(pairs(cells)),
            "by_arm": {a: split_kappa(pairs([r for r in cells if r["arm"] == a])) for a in arms},
        },
        "noise": {
            "overall": noise(cells),
            "by_arm": {a: noise([r for r in cells if r["arm"] == a]) for a in arms},
        },
    }


def noise_table(nz: dict[str, Any]) -> list[str]:
    def p(v: Any) -> str:
        return "—" if v is None else f"{100 * v:.1f}"

    lines = [
        "| scope | metric | rate r0 | rate r1 | diff | ± | share r0 | share r1 | diff | ± |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for scope, block in [("overall", nz["overall"]), *nz["by_arm"].items()]:
        for m in NOISE_METRICS:
            r, sh = block[m]["rate"], block[m]["share"]
            lines.append(
                f"| {scope} | {m} | {p(r['r0'])} | {p(r['r1'])} | {p(r['diff'])} | {p(r['pm'])} "
                f"| {p(sh['r0'])} | {p(sh['r1'])} | {p(sh['diff'])} | {p(sh['pm'])} |"
            )
    return lines


def render_markdown(res: dict[str, Any]) -> str:
    def f(v: Any) -> str:
        return "—" if v is None else f"{v:.3f}"

    o = res["overall"]
    lines = [
        f"## Claim-map judge consistency ({res['prompt_version']}), {res['n_cells']} cells, "
        f"{o['n_units']} units compared ({res['n_units_failed_either_run']} failed in a run)",
        "",
        "| scope | units | κ | raw agreement |",
        "|---|---|---|---|",
        f"| overall | {o['n_units']} | {f(o['kappa'])} | {f(o['raw_agreement'])} |",
        *(
            f"| {a} | {b['n_units']} | {f(b['kappa'])} | {f(b['raw_agreement'])} |"
            for a, b in res["by_arm"].items()
        ),
        *(
            f"| family {fa} | {b['n_units']} | {f(b['kappa'])} | {f(b['raw_agreement'])} |"
            for fa, b in res.get("by_family", {}).items()
        ),
        "",
        "### Hallucination split (false_claim / detail / substitution_category), on units both "
        "runs call hallucination",
        "",
        "| scope | units | κ | raw agreement |",
        "|---|---|---|---|",
        *(
            f"| {a} | {b['n_units']} | {f(b['kappa'])} | {f(b['raw_agreement'])} |"
            for a, b in [
                ("overall", res["detail_vs_false_claim"]["overall"]),
                *res["detail_vs_false_claim"]["by_arm"].items(),
            ]
        ),
        "",
        "### Headline metrics from run r0 vs run r1 (readout rate / character share, %)",
        "",
        *noise_table(res["noise"]),
        "",
        "### Per-leaf self-agreement (Dice of the two runs)",
        "",
        "| leaf | r0 | r1 | both | self-agreement |",
        "|---|---|---|---|---|",
    ]
    for lf, s in o["self_agreement"].items():
        lines.append(f"| {lf} | {s['n_r0']} | {s['n_r1']} | {s['n_both']} | {f(s['dice'])} |")
    lines += [
        "",
        f"### Flagged pairs (swapped in > {FLAG_SHARE:.0%} of either leaf's units)",
        "",
    ]
    if o["flagged_pairs"]:
        lines += ["| pair | swaps | share of A | share of B |", "|---|---|---|---|"]
        for p in o["flagged_pairs"]:
            a, b = p["pair"]
            lines.append(f"| {a} ↔ {b} | {p['n_swaps']} | {f(p['share_a'])} | {f(p['share_b'])} |")
    else:
        lines.append("none")
    lines += ["", "### Top swaps", "", "| pair | swaps | share of A | share of B |"]
    lines.append("|---|---|---|---|")
    for p in o["top_swaps"]:
        a, b = p["pair"]
        lines.append(f"| {a} ↔ {b} | {p['n_swaps']} | {f(p['share_a'])} | {f(p['share_b'])} |")
    leaves = [lf for lf in tx.LEAF_NAMES if any(lf in k.split("|") for k in o["confusion"])]
    lines += [
        "",
        "### Between-run confusion (rows r0, columns r1)",
        "",
        "| r0 \\ r1 | " + " | ".join(leaves) + " |",
        "|---" * (len(leaves) + 1) + "|",
    ]
    for a in leaves:
        cells = [str(o["confusion"].get(f"{a}|{b}", 0)) for b in leaves]
        lines.append(f"| {a} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _reparse_cell(
    c: dict[str, Any],
    *,
    cache: Cache,
    items: dict[str, dict[str, Any]],
    sites: dict[str, dict[int, dict[str, Any]]],
    kinds: dict[str, str],
    model: str,
    effort: str | None,
) -> dict[str, Any] | None:
    """Re-parse one cell's two cached answers with the current parser (no API calls). The
    prompt is rebuilt from the stored unit texts (the unit block uses only ids and texts) and the
    bank item, so the fingerprints match the original calls; ``None`` if either is missing."""
    m = re.match(r"^(.*)__L(\d{3})__p(\d+)$", c["key"])
    if m is None or m.group(1) not in items:
        return None
    item, site = items[m.group(1)], sites[m.group(1)][int(m.group(3))]
    units: list[Unit] = []
    pos: dict[int, int] = {}
    for u in c["units"]:
        r = _readout(u["id"])
        n = int(u["id"].split(".U", 1)[1])
        start = pos.get(r, 0)
        units.append(Unit(u["id"], r, n, start, start + len(u["text"]), u["text"]))
        pos[r] = start + len(u["text"]) + 1
    marked = bank.marked_response(item, site)
    system, user = cm.render_prompt(
        item["prompt"], marked, units, kind=kinds.get(c["arm"], "prose")
    )
    runs = []
    for r in range(REPEATS):
        fp = fingerprint(cm.PROMPT_VERSION, model, effort, f"r{r}", system, user)
        row = cache.get(cons_key(c["arm"], c["key"], r), fp, include_failed=True)
        if row is None:
            return None
        segs, _ = cm.parse(row["result"], units, cm.conversation_text(item["prompt"], marked))
        runs.append(unit_leaves(segs, units))
    return {**c, "r0": runs[0], "r1": runs[1]}


def rescore(out_root: Path, model: str, effort: str | None = "low") -> dict[str, Any]:
    """Re-score the cached r0 / r1 calls of the current prompt version with the current parser
    and scoring (no API calls): used when scoring changes but the prompt does not (rev 18-19).
    Each cell is re-parsed from the cached answers; a cell whose answers cannot be found falls
    back to its stored unit leaves (``n_cells_fallback``). Rewrites the ``.json`` / ``.md``."""
    d = Path(out_root) / "consistency" / wrun.model_slug(model)
    stem = f"consistency_{cm.PROMPT_VERSION}"
    old = json.loads((d / f"{stem}.json").read_text())
    rows = (d / f"{stem}_cells.jsonl").read_text(encoding="utf-8").splitlines()
    stored = [json.loads(x) for x in rows if x.strip()]
    items_l = bank.load_items()
    items = {it["id"]: it for it in items_l}
    sites = bank.sites_by_item(items_l)
    kinds = {}
    for a in {c["arm"] for c in stored}:
        p = Path(out_root) / "wsbench_chat" / wrun.model_slug(model) / a / "arm.json"
        kinds[a] = json.loads(p.read_text()).get("kind", "prose") if p.exists() else "prose"
    cells, n_fallback = [], 0
    with Cache(d / "cache.jsonl", readonly=True) as cache:
        for c in stored:
            new = _reparse_cell(
                c, cache=cache, items=items, sites=sites, kinds=kinds, model=model, effort=effort
            )
            if new is None:
                n_fallback += 1
                new = c
            cells.append(merge_cell(new))
    res = score(cells)
    res["n_cells_fallback"] = n_fallback
    keep = ("arms", "n_per_arm", "seed", "model")
    res.update({k: old[k] for k in keep if k in old})
    res["spend_this_run"] = {**Usage().to_dict()}
    res["spend_original"] = old.get("spend_original") or old.get("spend_this_run")
    res["rescored"] = {"scoring_version": tx.SCORING_VERSION}
    (d / f"{stem}.json").write_text(json.dumps(res, indent=1) + "\n")
    (d / f"{stem}.md").write_text(render_markdown(res) + "\n")
    with (d / f"{stem}_cells.jsonl").open("w", encoding="utf-8") as fh:
        for c in cells:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    return res


def run(
    arms_spec: str,
    *,
    readouts_root: Path,
    out_root: Path,
    models: wrun.Models,
    n: int = 50,
    seed: int = 0,
    rpm: float = 1500.0,
    concurrency: int = 64,
) -> dict[str, Any]:
    targets = parse_arms(arms_spec)
    opened: list[tuple[str, wrun.Arm, int]] = []
    for name, layer in targets:
        p = Path(readouts_root) / name / "hallucination.jsonl"
        opened.append((name, wrun.open_arm(name, p, out_root, models, readonly=True), layer))
    work = []
    for name, arm, layer in opened:
        picked = sample_cells(arm, layer, n, seed, models)
        print(f"[{name}] L{layer}: {len(picked)} cells sampled", flush=True)
        work += [(name, arm, c, s) for c, s in picked]
    d = Path(out_root) / "consistency" / wrun.model_slug(models.mapper)
    usage = Usage()

    async def go() -> list[dict[str, Any]]:
        llm = LLM(rpm=rpm, concurrency=concurrency, effort=models.effort)
        try:
            await llm.preflight(models.mapper)
            with Cache(d / "cache.jsonl") as cache:
                return await map_twice(work, models=models, cache=cache, llm=llm, usage=usage)
        finally:
            await llm.aclose()

    try:
        cells = asyncio.run(go())
    finally:
        for _n, arm, _la in opened:
            arm.cache.close()
    res = score(cells)
    res.update(
        {
            "arms": arms_spec,
            "n_per_arm": n,
            "seed": seed,
            "model": models.mapper,
            "spend_this_run": usage.to_dict(),
        }
    )
    stem = f"consistency_{cm.PROMPT_VERSION}"
    (d / f"{stem}.json").write_text(json.dumps(res, indent=1) + "\n")
    (d / f"{stem}.md").write_text(render_markdown(res) + "\n")
    with (d / f"{stem}_cells.jsonl").open("w", encoding="utf-8") as fh:
        for c in cells:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    return res
