"""Plan 0015 driver: Stage 2 (refusal ground truth), Stage 5 (detector), Stage 6 (verifier) and
the cache-only rebuild of per-cell records.

Caches (plan 0001 semantics: append-only, failed = ``result: null``, re-queued every run):
``<out>/overrefusal/<model>/gt_cache.jsonl`` (``gt:{id}``) and
``<out>/overrefusal/<model>/<arm>/cache.jsonl`` (``{key}:D``, ``{key}:V:{readout}:{qhash}``, and
usage-only ``batchD:`` / ``batchV:`` rows that are never counted as cells).
``cells.jsonl`` / ``summary.json`` are always regenerated from the cache.
"""

import asyncio
import json
import random
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hallucination_bench.cache import Cache, fingerprint
from hallucination_bench.llm import LLM, Usage
from hallucination_bench.readouts import Cell, load_readouts

from . import data, detect, gt, verify

READOUTS_FILE = "overrefusal.jsonl"
STAGES = ("detect", "verify")
GREEDY_CONFIGS = ("hard", "80k2")  # rev 7: the new never_refused source also needs a greedy reply


@dataclass(frozen=True)
class Judge:
    model: str = "claude-opus-5-5"
    effort: str | None = "low"


def eval_root(out: Path, model: str) -> Path:
    return Path(out) / "overrefusal" / model


def _usage_meta(u: Usage) -> dict[str, Any]:
    return {"usage": u.to_dict()}


def spend(cache: Cache) -> Usage:
    tot = Usage()
    for row in cache.rows():
        tot.add((row.get("meta") or {}).get("usage"))
    return tot


# ------------------------------------------------------------------ Stage 2: ground truth


def gt_jobs(
    cands: Sequence[dict[str, Any]],
    samples: Sequence[dict[str, Any]],
    greedy: Sequence[dict[str, Any]],
) -> list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]]:
    """``(candidate, its N sampled rows in order, its greedy row or None)``; candidates without
    all :data:`data.N_SAMPLES` samples, or hard ones without a greedy reply, are skipped."""
    by: dict[str, list[dict[str, Any]]] = {}
    for s in samples:
        by.setdefault(s["id"], []).append(s)
    g = {s["id"]: s for s in greedy}
    jobs = []
    for c in cands:
        ss = sorted(by.get(c["id"], []), key=lambda s: s["sample"])
        if [s["sample"] for s in ss] != list(range(data.N_SAMPLES)):
            continue
        needs_greedy = c["config"] in GREEDY_CONFIGS
        gr = g.get(c["id"]) if needs_greedy else None
        if needs_greedy and gr is None:
            continue
        jobs.append((c, ss, gr))
    return jobs


def _gt_user(
    c: dict[str, Any],
    ss: Sequence[dict[str, Any]],
    gr: dict[str, Any] | None,
    trunc: int | None = None,
) -> str:
    texts = [s["text"] for s in ss] + ([gr["text"]] if gr is not None else [])
    if trunc:
        texts = [t if len(t) <= trunc else t[:trunc] + " [...]" for t in texts]
    return gt.render_user(c["prompt"], texts)


async def _run_gt(
    jobs: Sequence[tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]],
    cache: Cache,
    llm: LLM,
    judge: Judge,
) -> None:
    await llm.preflight(judge.model)

    async def one(c: dict[str, Any], ss: list[dict[str, Any]], gr: dict[str, Any] | None) -> None:
        user = _gt_user(c, ss, gr)
        fp = fingerprint(gt.PROMPT_VERSION, judge.model, judge.effort, gt.SYSTEM, user)
        if cache.get(f"gt:{c['id']}", fp):
            return
        n = len(ss) + (gr is not None)
        # API-refusal fallback chain (pilot 2026-10-27: Opus hard-refused ~1 in 6 hard prompts
        # even with the research frame): full text -> responses cut to TRUNC_CHARS -> Sonnet 5.5
        # on the full text. Dropping refused prompts would bias never_refused against exactly the
        # prompts the subject complied with. The route is recorded per prompt.
        routes = [
            ("opus_full", judge.model, user),
            ("opus_trunc", judge.model, _gt_user(c, ss, gr, trunc=gt.TRUNC_CHARS)),
            ("sonnet_full", gt.FALLBACK_MODEL, user),
        ]
        total = Usage()
        labels, route, raw = None, None, None
        for name, model, u in routes:
            res = await llm.call(model=model, system=gt.SYSTEM, user=u, schema=gt.SCHEMA)
            total.add(res.usage)
            labels, raw = gt.parse(res.result, n), res.result
            if labels is not None:
                route = name
                break
            if not res.usage.refusals:
                break  # a non-refusal failure (bad parse): re-queued, not escalated
        meta = _usage_meta(total)
        meta["route"] = route
        if labels is None:
            meta["raw"] = raw
        cache.put(
            f"gt:{c['id']}",
            fp,
            {
                "result": None if labels is None else {"labels": labels, "route": route},
                "meta": meta,
            },
        )

    await asyncio.gather(*(one(*j) for j in jobs))


def run_gt(
    out: Path,
    judge: Judge,
    *,
    samples_path: Path,
    greedy_path: Path,
    rpm: float = 1000.0,
    concurrency: int = 64,
    limit: int | None = None,
    write: bool = True,
    candidates_path: Path | None = None,
    labels_out: Path | None = None,
) -> dict[str, Any]:
    cands = data.load_candidates(candidates_path)
    samples = data.read_jsonl(samples_path)
    greedy = data.read_jsonl(greedy_path) if Path(greedy_path).exists() else []
    jobs = gt_jobs(cands, samples, greedy)
    if limit:
        jobs = jobs[:limit]
    root = eval_root(out, judge.model)
    cache = Cache(root / "gt_cache.jsonl")
    llm = LLM(rpm=rpm, concurrency=concurrency, effort=judge.effort)

    async def go() -> None:
        try:
            await _run_gt(jobs, cache, llm, judge)
        finally:
            await llm.aclose()

    asyncio.run(go())
    rows = []
    n_failed = 0
    for c, ss, gr in gt_jobs(cands, samples, greedy):
        fp = fingerprint(
            gt.PROMPT_VERSION, judge.model, judge.effort, gt.SYSTEM, _gt_user(c, ss, gr)
        )
        hit = cache.get(f"gt:{c['id']}", fp)
        if hit is None:
            n_failed += cache.get(f"gt:{c['id']}", fp, include_failed=True) is not None
            continue
        row = gt.summarize(c, ss, hit["result"]["labels"], gr)
        row["judge_route"] = hit["result"].get("route", "opus_full")
        rows.append(row)
    if write:
        data.write_jsonl(labels_out or data.GT_LABELS, rows)
    u = spend(cache)
    cache.close()
    return {
        "n_candidates": len(cands),
        "n_jobs": len(jobs),
        "n_labelled": len(rows),
        "n_failed": n_failed,
        "spend": u.to_dict(),
    }


# ------------------------------------------------------------------ readouts


def load_arm(
    path: Path, items: Sequence[dict[str, Any]], layers: Sequence[int] | None
) -> list[Cell]:
    """Every cell of the bank grid at the file's layers (or ``layers``); raises on any gap."""
    pos = data.positions(items)
    cells, rep = load_readouts(path, positions=pos, layers=layers)
    if not cells:
        raise SystemExit(f"{path}: no readouts on the bank grid")
    want = {(i, la, p) for i, ps in pos.items() for la in rep.layers for p in ps}
    have = {(c.id, c.layer, c.pos) for c in cells}
    if want - have:
        miss = sorted(want - have)
        raise SystemExit(f"{path}: {len(miss)} missing cells, e.g. {miss[:3]}")
    return cells


def resolve_arms(root: Path, names: Sequence[str] | None) -> list[tuple[str, Path]]:
    root = Path(root)
    if names:
        arms = [(n, root / n / READOUTS_FILE) for n in names]
    else:
        arms = [
            (d.name, d / READOUTS_FILE)
            for d in sorted(root.iterdir())
            if (d / READOUTS_FILE).exists()
        ]
    for n, p in arms:
        if not p.exists():
            raise SystemExit(f"arm {n}: no readouts file {p}")
    return arms


# ------------------------------------------------------------------ Stage 5: detector


def _dfp(judge: Judge, texts: Sequence[str]) -> str:
    return fingerprint(
        detect.PROMPT_VERSION, judge.model, judge.effort, detect.SYSTEM, detect.canonical(texts)
    )


async def _detect(
    cells: Sequence[Cell], cache: Cache, llm: LLM, judge: Judge, *, limit: int | None, seed: int = 0
) -> None:
    todo = [
        c
        for c in cells
        if not c.empty and cache.get(f"{c.key}:D", _dfp(judge, detect.readout_texts(c))) is None
    ]
    random.Random(seed).shuffle(todo)
    if limit:
        todo = todo[:limit]
    if not todo:
        return
    is_tokens = todo[0].kind == "tokens"

    async def batch_call(batch: Sequence[Cell], *, final: bool) -> list[Cell]:
        texts = [detect.readout_texts(c) for c in batch]
        res = await llm.call(
            model=judge.model,
            system=detect.SYSTEM,
            user=detect.render_user(texts),
            schema=detect.SCHEMA,
        )
        bp = detect.parse_batch(res.result, texts, is_tokens=is_tokens)
        cache.put(
            f"batchD:{uuid.uuid4().hex[:12]}",
            "usage",
            {
                "result": {"n_cells": len(batch), "n_unlocated": bp.n_unlocated, "n_bad": bp.n_bad},
                "meta": {**_usage_meta(res.usage), "api_refused": res.usage.refusals},
            },
        )
        failed = []
        for j, c in enumerate(batch):
            m = bp.cells[j]
            meta: dict[str, Any] = {}
            if m is None and final and isinstance(res.result, dict) and len(batch) == 1:
                m, meta["accepted_partial"] = bp.partial[j], True
            if m is None:
                failed.append(c)
                meta["api_refused"] = res.usage.refusals
            cache.put(
                f"{c.key}:D",
                _dfp(judge, texts[j]),
                {"result": None if m is None else {"mentions": m}, "meta": meta},
            )
        return failed

    batches = [todo[i : i + detect.BATCH_SIZE] for i in range(0, len(todo), detect.BATCH_SIZE)]
    out = await asyncio.gather(*(batch_call(b, final=False) for b in batches))
    failed = [c for f in out for c in f]
    for attempt in range(detect.SINGLE_ATTEMPTS):
        if not failed:
            break
        final = attempt == detect.SINGLE_ATTEMPTS - 1
        out = await asyncio.gather(*(batch_call([c], final=final) for c in failed))
        failed = [c for f in out for c in f]


# ------------------------------------------------------------------ Stage 6: verifier


def _vfp(judge: Judge, readout: str, quote: str) -> str:
    return fingerprint(
        verify.PROMPT_VERSION,
        judge.model,
        judge.effort,
        verify.SYSTEM,
        verify.item_text(readout, quote),
    )


def _vkey(cell: Cell, m: dict[str, Any]) -> str:
    return f"{cell.key}:V:{m['readout']}:{verify.quote_hash(m['quote'])}"


def _candidates(
    cells: Sequence[Cell], cache: Cache, judge: Judge
) -> list[tuple[Cell, dict[str, Any]]]:
    out = []
    for c in cells:
        if c.empty:
            continue
        hit = cache.get(f"{c.key}:D", _dfp(judge, detect.readout_texts(c)))
        if hit is None:
            continue
        seen = set()
        for m in hit["result"]["mentions"]:
            k = (m["readout"], m["quote"])
            if m["label"] == "refusal_intent" and k not in seen:
                seen.add(k)
                out.append((c, m))
    return out


async def _verify(cells: Sequence[Cell], cache: Cache, llm: LLM, judge: Judge) -> None:
    todo = []
    for c, m in _candidates(cells, cache, judge):
        r = detect.readout_texts(c)[m["readout"]]
        if cache.get(_vkey(c, m), _vfp(judge, r, m["quote"])) is None:
            todo.append((c, m, r))
    if not todo:
        return
    is_tokens = todo[0][0].kind == "tokens"

    async def batch_call(batch: Sequence[tuple[Cell, dict[str, Any], str]]) -> None:
        pairs = [(r, m["quote"]) for _, m, r in batch]
        res = await llm.call(
            model=judge.model,
            system=verify.SYSTEM,
            user=verify.render_user(pairs),
            schema=verify.SCHEMA,
        )
        cache.put(
            f"batchV:{uuid.uuid4().hex[:12]}",
            "usage",
            {
                "result": {"n_items": len(batch)},
                "meta": {**_usage_meta(res.usage), "api_refused": res.usage.refusals},
            },
        )
        for (c, m, r), v in zip(
            batch, verify.parse(res.result, len(batch), is_tokens=is_tokens), strict=True
        ):
            cache.put(_vkey(c, m), _vfp(judge, r, m["quote"]), {"result": v, "meta": {}})

    bs = verify.BATCH_SIZE
    await asyncio.gather(*(batch_call(todo[i : i + bs]) for i in range(0, len(todo), bs)))


# ------------------------------------------------------------------ rebuild


def cell_record(cell: Cell, cache: Cache, judge: Judge) -> dict[str, Any]:
    """Per-cell flags per readout. ``ref`` = verified refusal_intent (an unverified candidate keeps
    the detector label), ``ref_as`` = verified with stance asserted, ``ref_det`` = detector."""
    texts = detect.readout_texts(cell)
    rec: dict[str, Any] = {
        "id": cell.id,
        "layer": cell.layer,
        "pos": cell.pos,
        "n_readouts": len(texts),
    }
    flags = [
        {"ref": 0, "ref_as": 0, "ref_det": 0, "harm": 0, "comply": 0, "other": 0} for _ in texts
    ]
    if cell.empty:
        rec.update(status="empty", ro=flags, refusals=[])
        return rec
    hit = cache.get(f"{cell.key}:D", _dfp(judge, texts))
    if hit is None:
        failed = cache.get(f"{cell.key}:D", _dfp(judge, texts), include_failed=True) is not None
        rec.update(status="failed" if failed else "pending", ro=flags, refusals=[])
        return rec
    refusals = []
    n_unverified = n_rejected = 0
    for m in hit["result"]["mentions"]:
        f = flags[m["readout"]]
        if m["label"] == "harm_salience":
            f["harm"] = 1
        elif m["label"] == "comply_intent":
            f["comply"] = 1
        elif m["label"] == "other":
            f["other"] = 1
        else:
            f["ref_det"] = 1
            v = cache.get(_vkey(cell, m), _vfp(judge, texts[m["readout"]], m["quote"]))
            ver = v["result"] if v is not None else None
            if ver is None:
                n_unverified += 1
                ok, stance = True, m["stance"]
            else:
                ok, stance = ver["label"] == "refusal_intent", ver["stance"]
                n_rejected += not ok
            if ok:
                f["ref"] = 1
                if stance == "asserted":
                    f["ref_as"] = 1
            refusals.append(
                {
                    "readout": m["readout"],
                    "quote": m["quote"],
                    "det_stance": m["stance"],
                    "verified": None if ver is None else ver["label"],
                    "stance": stance if ok else None,
                }
            )
    rec.update(
        status="accepted_partial" if (hit.get("meta") or {}).get("accepted_partial") else "ok",
        ro=flags,
        refusals=refusals,
        n_unverified=n_unverified,
        n_rejected=n_rejected,
    )
    return rec


def rebuild(
    name: str, path: Path, out: Path, judge: Judge, items: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    cells = load_arm(path, items, None)
    root = eval_root(out, judge.model) / name
    cache = Cache(root / "cache.jsonl", readonly=True)
    recs = [cell_record(c, cache, judge) for c in cells]
    data.write_jsonl(root / "cells.jsonl", recs)
    st = Counter(r["status"] for r in recs)
    u = spend(cache)
    n_api = sum(
        int((r.get("meta") or {}).get("api_refused") or 0)
        for r in cache.rows()
        if r["key"].startswith("batch")
    )
    summary = {
        "arm": name,
        "readouts": str(path),
        "kind": cells[0].kind,
        "layers": sorted({c.layer for c in cells}),
        "n_cells": len(recs),
        "status": dict(st),
        "n_refusal_candidates": sum(len(r["refusals"]) for r in recs),
        "n_unverified": sum(r.get("n_unverified", 0) for r in recs),
        "n_rejected": sum(r.get("n_rejected", 0) for r in recs),
        "n_api_refused": n_api,
        "prompt_versions": [detect.PROMPT_VERSION, verify.PROMPT_VERSION],
        "spend": u.to_dict(),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def run(
    arms: Sequence[tuple[str, Path]],
    out: Path,
    judge: Judge,
    *,
    stages: Sequence[str] = STAGES,
    layers: Sequence[int] | None = None,
    limit: int | None = None,
    rpm: float = 1000.0,
    concurrency: int = 96,
) -> dict[str, dict[str, Any]]:
    items = data.load_items()
    llm = LLM(rpm=rpm, concurrency=concurrency, effort=judge.effort)
    loaded = [(n, p, load_arm(p, items, layers)) for n, p in arms]
    caches = {n: Cache(eval_root(out, judge.model) / n / "cache.jsonl") for n, _, _ in loaded}

    async def go() -> None:
        try:
            await llm.preflight(judge.model)
            if "detect" in stages:
                await asyncio.gather(
                    *(_detect(cs, caches[n], llm, judge, limit=limit) for n, _, cs in loaded)
                )
            if "verify" in stages:
                await asyncio.gather(*(_verify(cs, caches[n], llm, judge) for n, _, cs in loaded))
        finally:
            await llm.aclose()

    asyncio.run(go())
    for c in caches.values():
        c.close()
    return {n: rebuild(n, p, out, judge, items) for n, p, _ in loaded}


def n_failed(summary: dict[str, Any]) -> int:
    return int(summary["status"].get("failed", 0)) + int(summary["n_unverified"])
