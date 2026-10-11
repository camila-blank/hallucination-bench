"""Judge-consistency check for the claim map (plan 0002 rev 16): two fresh ``:M`` calls per
real cell, leaf agreement between the runs.

``hbench consistency-v2 --arms s3d-rl600:44,nla-rl-L42:42,jlens:44 --n 50 --seed 0`` samples
``n`` non-empty cells per ``arm:layer`` (seeded; token arms only among cells whose summary is
cached in ``<arm>/cache.jsonl``), maps each twice in single-call mode (repeat keys ``r0`` / ``r1``
in ``outputs/consistency/<model>/cache.jsonl``; the repeat index is in the fingerprint, so each
is a fresh call), and compares the two runs **per unit** (each run's majority-character leaf over
the unit, as in calibration). Units failed in either run are left out.

Reported: Cohen's κ and raw agreement over the 12 leaves (overall and per arm), the leaf × leaf
between-run confusion table, per-leaf self-agreement (2·both / (r0 + r1), the Dice overlap of
the two runs' units with that leaf), and the **flagged pairs**: a leaf pair swapped between runs
in more than 10% of either leaf's units (units where at least one run gives that leaf). Plan
rule: each flagged pair gets a sharper rule or is merged, reported to Camila before the full run.
"""

import asyncio
import json
import random
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
            segs, _ = cm.parse(raw, units)
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


def score(cells: Sequence[dict[str, Any]]) -> dict[str, Any]:
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
    by_arm = {a: agreement(pairs([r for r in cells if r["arm"] == a])) for a in arms}
    return {
        "prompt_version": cm.PROMPT_VERSION,
        "taxonomy_version": tx.TAXONOMY_VERSION,
        "n_cells": len(cells),
        "n_units_failed_either_run": n_failed,
        "overall": agreement(pairs(cells)),
        "by_arm": by_arm,
    }


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
