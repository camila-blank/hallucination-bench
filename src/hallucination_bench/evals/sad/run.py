"""The sad driver (plan 0018): detector (Stage D) and verifier (Stage H) over one arm's readouts,
then a cache-only rebuild of ``cells.jsonl`` + ``summary.json``.

Per-arm output ``<out>/sad/<judge>/<arm>/``: ``cache.jsonl`` (append-only; failed rows
``result: null`` are re-queued), ``cells.jsonl`` and ``summary.json`` (always regenerated).
Readouts: ``<root>/<arm>/sad.jsonl`` (the readout contract), checked against the full grid of
``data/sad/selected.json`` before anything runs."""

import asyncio
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from hallucination_bench.cache import Cache, fingerprint
from hallucination_bench.llm import LLM, Usage
from hallucination_bench.readouts import Cell, load_readouts

from . import bank, detect, score, verify

READOUTS_FILE = "sad.jsonl"
SEED = 0
VERIFY_ALL_MAX = 5000  # plan: verify every mention up to this many per arm
VERIFY_PER_STRATUM = 150
SMALL_BATCH = 4
SMALL_BATCH_SOURCES = ("test_wmdp",)


def load_selected(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or bank.SELECTED_JSON).read_text(encoding="utf-8"))


def expected_grid(selected: dict[str, Any], layers: list[int]) -> set[tuple[str, int, int]]:
    return {
        (it["id"], la, s["pos"]) for it in selected["items"] for la in layers for s in it["sites"]
    }


def load_arm(path: Path, selected: dict[str, Any], layers: list[int] | None) -> list[Cell]:
    """Every cell of the selected grid at ``layers`` (default: every layer in the file); raises
    on missing / off-grid cells."""
    ids = [it["id"] for it in selected["items"]]
    pos = {it["id"]: [s["pos"] for s in it["sites"]] for it in selected["items"]}
    cells, rep = load_readouts(path, ids=ids, layers=layers, positions=pos)
    got_layers = sorted({c.layer for c in cells})
    want = expected_grid(selected, layers or got_layers)
    have = {(c.id, c.layer, c.pos) for c in cells}
    miss = want - have
    off = rep.skipped.get("pos_not_selected", 0) + rep.skipped.get("unknown_id", 0)
    if miss or not cells:
        raise SystemExit(f"{path}: {len(miss)} grid cells missing (of {len(want)})")
    if off:
        print(f"[sad] {path}: {off} rows outside the selected grid ignored", flush=True)
    return sorted(cells, key=lambda c: (c.layer, c.id, c.pos))


def arm_dir(out: Path, judge: str, arm: str) -> Path:
    return out / "sad" / judge / arm


def _det_fp(model: str, effort: str | None, canon: str) -> str:
    return fingerprint(detect.PROMPT_VERSION, model, effort, detect.SYSTEM, canon)


def _ver_fp(model: str, effort: str | None, canon: str) -> str:
    return fingerprint(verify.VERIFY_VERSION, model, effort, verify.SYSTEM, canon)


def _batch_key(prefix: str, keys: list[str]) -> str:
    return prefix + hashlib.sha256("\n".join(keys).encode()).hexdigest()[:16]


# ---------------------------------------------------------------- Stage D


async def _detect(
    llm: LLM,
    cache: Cache,
    cells: list[Cell],
    model: str,
    effort: str | None,
    small_batch_ids: frozenset[str] = frozenset(),
) -> dict[str, int]:
    todo = []
    for c in cells:
        ro = detect.readouts_of(c)
        if detect.is_empty(ro):
            continue
        canon = detect.canonical(ro)
        if cache.get(f"{c.key}:D", _det_fp(model, effort, canon)) is None:
            todo.append((c, ro, canon))
    random.Random(SEED).shuffle(todo)
    stats: Counter = Counter()
    single_attempts: Counter = Counter()

    async def one(batch: list[tuple[Cell, list[str], str]]) -> list[tuple[Cell, list[str], str]]:
        system, user = detect.render_batch([b[2] for b in batch])
        res = await llm.call(model=model, system=system, user=user, schema=detect.SCHEMA)
        keys = [b[0].key for b in batch]
        cache.put(
            _batch_key("batchD:", keys),
            "usage",
            {"result": {"cells": keys}, "meta": {"usage": res.usage.to_dict()}},
        )
        kinds = ["prose" if b[0].kind == "prose" else "tokens" for b in batch]
        bp = detect.parse_batch(res.result, [b[1] for b in batch], kinds)
        retry = []
        for j, (c, ro, canon) in enumerate(batch):
            fp = _det_fp(model, effort, canon)
            if j in bp.failed or res.result is None:
                partial = bp.partial(j)
                if len(batch) == 1:
                    single_attempts[c.key] += 1
                    if single_attempts[c.key] >= detect.SINGLE_ATTEMPTS and partial is not None:
                        cache.put(
                            f"{c.key}:D",
                            fp,
                            {"result": partial, "meta": {"accepted_partial": True}},
                        )
                        stats["accepted_partial"] += 1
                        continue
                cache.put(f"{c.key}:D", fp, {"result": None, "meta": {"batch_size": len(batch)}})
                retry.append((c, ro, canon))
                stats["failed_attempts"] += 1
            else:
                cache.put(f"{c.key}:D", fp, {"result": bp.cells[j].to_result()})
                stats["ok"] += 1
        return retry

    # WMDP-item cells (hazard-domain continuations) draw most API refusals: they go in batches of
    # SMALL_BATCH so a refusal does not take 39 other cells with it. Batching only; the detector
    # still sees no item, provenance or belief.
    small = [x for x in todo if x[0].id in small_batch_ids]
    big = [x for x in todo if x[0].id not in small_batch_ids]
    batches = [big[i : i + detect.BATCH_SIZE] for i in range(0, len(big), detect.BATCH_SIZE)]
    batches += [small[i : i + SMALL_BATCH] for i in range(0, len(small), SMALL_BATCH)]
    print(
        f"[sad] detect: {len(todo)} cells in {len(batches)} batches ({len(small)} small-batch)",
        flush=True,
    )
    retry = [x for r in await asyncio.gather(*(one(b) for b in batches)) for x in r]
    for _ in range(detect.SINGLE_ATTEMPTS):
        if not retry:
            break
        retry = [x for r in await asyncio.gather(*(one([x]) for x in retry)) for x in r]
    stats["still_failed"] = len(retry)
    return dict(stats)


# ---------------------------------------------------------------- Stage H


def all_mentions(
    cache: Cache, cells: list[Cell], model: str, effort: str | None
) -> list[dict[str, Any]]:
    """Every detector mention: ``{cell, readout, start, end, label, quote, text}``."""
    out = []
    for c in cells:
        ro = detect.readouts_of(c)
        if detect.is_empty(ro):
            continue
        row = cache.get(f"{c.key}:D", _det_fp(model, effort, detect.canonical(ro)))
        if row is None:
            continue
        for m in row["result"]["mentions"]:
            out.append({"cell": c, **m, "text": ro[m["readout"]]})
    return out


def verify_sample(mentions: list[dict[str, Any]], belief: dict[str, str]) -> list[dict[str, Any]]:
    """All mentions if at most ``VERIFY_ALL_MAX``; else a seeded sample stratified by label x
    belief group x layer (``VERIFY_PER_STRATUM`` each, or all of a smaller stratum)."""
    if len(mentions) <= VERIFY_ALL_MAX:
        return mentions
    strata: dict[tuple, list] = defaultdict(list)
    for m in mentions:
        strata[(m["label"], belief[m["cell"].id], m["cell"].layer)].append(m)
    rng = random.Random(SEED)
    out = []
    for k in sorted(strata):
        xs = strata[k]
        out += xs if len(xs) <= VERIFY_PER_STRATUM else rng.sample(xs, VERIFY_PER_STRATUM)
    return out


def mention_key(m: dict[str, Any]) -> str:
    return f"{m['cell'].key}:H:{m['readout']}:{m['start']}-{m['end']}"


async def _verify(
    llm: LLM, cache: Cache, mentions: list[dict[str, Any]], model: str, effort: str | None
) -> dict[str, int]:
    todo = []
    for m in mentions:
        canon = verify.canonical(m["text"], m["quote"], m["label"])
        if cache.get(mention_key(m), _ver_fp(model, effort, canon)) is None:
            todo.append((m, canon))
    random.Random(SEED).shuffle(todo)
    stats: Counter = Counter()

    async def one(batch: list[tuple[dict[str, Any], str]]) -> list:
        system, user = verify.render_batch([b[1] for b in batch])
        res = await llm.call(model=model, system=system, user=user, schema=verify.SCHEMA)
        keys = [mention_key(b[0]) for b in batch]
        cache.put(
            _batch_key("batchH:", keys),
            "usage",
            {"result": {"mentions": keys}, "meta": {"usage": res.usage.to_dict()}},
        )
        parsed = verify.parse_batch(res.result, len(batch))
        retry = []
        for j, (m, canon) in enumerate(batch):
            fp = _ver_fp(model, effort, canon)
            if j in parsed:
                cache.put(mention_key(m), fp, {"result": parsed[j]})
                stats["ok"] += 1
            else:
                cache.put(mention_key(m), fp, {"result": None})
                retry.append((m, canon))
        return retry

    batches = [todo[i : i + verify.BATCH_SIZE] for i in range(0, len(todo), verify.BATCH_SIZE)]
    print(f"[sad] verify: {len(todo)} mentions in {len(batches)} batches", flush=True)
    retry = [x for r in await asyncio.gather(*(one(b) for b in batches)) for x in r]
    if retry:
        retry = [x for r in await asyncio.gather(*(one([x]) for x in retry)) for x in r]
    stats["still_failed"] = len(retry)
    return dict(stats)


# ---------------------------------------------------------------- rebuild


def rebuild(
    arm: str,
    cells: list[Cell],
    selected: dict[str, Any],
    out: Path,
    judge: str,
    effort: str | None,
) -> dict[str, Any]:
    """Cache-only: per-cell records -> ``cells.jsonl``; scores -> ``summary.json``."""
    d = arm_dir(out, judge, arm)
    cache = Cache(d / "cache.jsonl", readonly=True)
    items = {it["id"]: it for it in selected["items"]}
    region = {(it["id"], s["pos"]): s["region"] for it in selected["items"] for s in it["sites"]}
    final_pos = {it["id"]: it["sites"][-1]["pos"] for it in selected["items"]}
    belief = {i: it["belief"] for i, it in items.items()}
    sit = all_mentions(cache, cells, judge, effort)
    sampled = {mention_key(m) for m in verify_sample(sit, belief)}
    health: Counter = Counter()
    records = []
    for c in cells:
        ro = detect.readouts_of(c)
        rec: dict[str, Any] = {
            "key": c.key,
            "id": c.id,
            "layer": c.layer,
            "pos": c.pos,
            "region": region[(c.id, c.pos)],
            "final": c.pos == final_pos[c.id],
            "k": len(ro),
            "status": "ok",
            "mentions": [],
        }
        if detect.is_empty(ro):
            rec["status"] = "empty"
            health["n_empty"] += 1
            records.append(rec)
            continue
        fp = _det_fp(judge, effort, detect.canonical(ro))
        row = cache.get(f"{c.key}:D", fp)
        if row is None:
            failed = cache.get(f"{c.key}:D", fp, include_failed=True)
            rec["status"] = "failed" if failed is not None else "pending"
            health[f"n_{rec['status']}_detect"] += 1
            records.append(rec)
            continue
        res = row["result"]
        health["n_unlocated"] += res.get("n_unlocated", 0)
        health["n_bad_enum"] += res.get("n_bad_enum", 0)
        if (row.get("meta") or {}).get("accepted_partial"):
            health["n_accepted_partial"] += 1
        for m in res["mentions"]:
            mm = dict(m)
            mk = f"{c.key}:H:{m['readout']}:{m['start']}-{m['end']}"
            canon = verify.canonical(ro[m["readout"]], m["quote"], m["label"])
            vrow = cache.get(mk, _ver_fp(judge, effort, canon))
            mm["verify"] = vrow["result"] if vrow else None
            mm["verified"] = verify.confirmed(m["label"], vrow["result"] if vrow else None)
            mm["in_verify_sample"] = mk in sampled
            if mk in sampled and vrow is None:
                health["n_unverified_in_sample"] += 1
            rec["mentions"].append(mm)
        health["n_detect_ok"] += 1
        records.append(rec)
    spend = Usage()
    spend_by: dict[str, Usage] = {"detect": Usage(), "verify": Usage()}
    for r in cache.rows():
        if r["key"].startswith(("batchD:", "batchH:")):
            u = (r.get("meta") or {}).get("usage")
            spend.add(u)
            spend_by["detect" if r["key"].startswith("batchD:") else "verify"].add(u)
    d.mkdir(parents=True, exist_ok=True)
    with (d / "cells.jsonl").open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    summary = {
        "arm": arm,
        "judge": judge,
        "detect_version": detect.PROMPT_VERSION,
        "verify_version": verify.VERIFY_VERSION,
        "layers": sorted({c.layer for c in cells}),
        "kind": cells[0].kind if cells else None,
        "n_cells": len(records),
        "n_mentions": len(sit),
        "n_verify_sample": len(sampled),
        "verify_sampled": len(sampled) < len(sit),
        "health": dict(health),
        "spend": {
            "total_usd": spend.usd,
            "by_stage": {k: v.to_dict() for k, v in spend_by.items()},
        },
        "scores": score.score_arm(records, items),
    }
    (d / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def run_arm(
    arm: str,
    path: Path,
    out: Path,
    *,
    judge: str,
    effort: str | None,
    layers: list[int] | None,
    stages: set[str],
    rpm: float,
    concurrency: int,
    limit: int | None = None,
    selected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected = selected or load_selected()
    cells = load_arm(path, selected, layers)
    if limit:
        cells = cells[:limit]
    d = arm_dir(out, judge, arm)
    d.mkdir(parents=True, exist_ok=True)
    belief = {it["id"]: it["belief"] for it in selected["items"]}

    async def go() -> dict[str, Any]:
        llm = LLM(rpm=rpm, concurrency=concurrency, effort=effort)
        stats: dict[str, Any] = {}
        try:
            await llm.preflight(judge)
            with Cache(d / "cache.jsonl") as cache:
                if "detect" in stages:
                    small = frozenset(
                        it["id"]
                        for it in selected["items"]
                        if it.get("source") in SMALL_BATCH_SOURCES
                    )
                    stats["detect"] = await _detect(llm, cache, cells, judge, effort, small)
                if "verify" in stages:
                    sit = all_mentions(cache, cells, judge, effort)
                    stats["verify"] = await _verify(
                        llm, cache, verify_sample(sit, belief), judge, effort
                    )
        finally:
            await llm.aclose()
        return stats

    stats = asyncio.run(go()) if stages else {}
    summary = rebuild(arm, cells, selected, out, judge, effort)
    summary["run_stats"] = stats
    return summary
