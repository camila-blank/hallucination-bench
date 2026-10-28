"""The plan-0017 driver: stage D (detect) -> V (verify) -> VB (blind verify), then a cache-only
rebuild of ``cells.jsonl`` / ``summary.json`` per arm.

Per-arm cache ``<out>/<arm>/cache.jsonl`` (``out`` = ``outputs/mask/<model>``):
- ``{key}:D`` per cell (``key = {task_id}__L{layer:03d}__p{pos}``), result
  ``{mentions: [...]}``; usage on ``batchD:{sha}`` rows;
- ``{key}:V:{readout}:{start}-{end}`` / ``{key}:VB:...`` per lie-claim mention, result
  ``{real, recategory, reason}``; usage on ``batchV:`` / ``batchVB:`` rows.
Failed calls are cached with ``result: null`` and re-queued; a failed cell / mention is retried
alone up to ``SINGLE_ATTEMPTS`` times in the same run. Batch rows are never counted as cells.
"""

import asyncio
import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hallucination_bench.cache import Cache, fingerprint
from hallucination_bench.llm import LLM, Usage
from hallucination_bench.readouts import Cell

from . import bank
from . import detect as D
from . import verify as V
from .gt import pressure_messages_text

STAGES = ("detect", "verify", "verify_blind")
SINGLE_ATTEMPTS = 2
RETRY_CHUNK = 8  # rev 5: failed detector cells are re-batched in 8s before singles
resolve_arms = bank.resolve_arms


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


@dataclass
class DJob:
    key: str
    fp: str
    canon: str
    readouts: list[str]


@dataclass
class VJob:
    key: str
    fp: str
    canon: str
    tid: str


def _usage_row(cache: Cache, prefix: str, body: str, usage: Usage, extra: dict[str, Any]) -> None:
    cache.put(
        f"{prefix}:{_sha(body)}",
        "batch",
        {"result": {}, "meta": {"usage": usage.to_dict(), **extra}},
    )


def detect_jobs(cells: Sequence[Cell], model: str, effort: str | None) -> tuple[list[DJob], int]:
    jobs, n_empty = [], 0
    for c in cells:
        ro = D.readouts_of(c)
        if D.is_empty(ro):
            n_empty += 1
            continue
        canon = D.canonical(ro)
        fp = fingerprint(D.PROMPT_VERSION, model, effort, D.SYSTEM, canon)
        jobs.append(DJob(f"{c.key}:D", fp, canon, ro))
    return jobs, n_empty


async def _detect(
    llm: LLM, cache: Cache, jobs: list[DJob], model: str, batch: int, seed: int = 0
) -> int:
    todo = [j for j in jobs if cache.get(j.key, j.fp) is None]
    random.Random(seed).shuffle(todo)
    batches = [todo[i : i + batch] for i in range(0, len(todo), batch)]
    failed: list[DJob] = []

    async def one(bs: list[DJob]) -> list[DJob]:
        system, user = D.render_batch([j.canon for j in bs])
        out = await llm.call(model=model, system=system, user=user, schema=D.SCHEMA)
        _usage_row(cache, "batchD", user, out.usage, {"n_cells": len(bs)})
        bad = []
        for j, p in zip(bs, D.parse_batch(out.result, [j.readouts for j in bs]), strict=True):
            if p.ok:
                cache.put(
                    j.key, j.fp, {"result": {"mentions": p.mentions}, "meta": {"model": model}}
                )
            else:
                bad.append(j)
                if len(bs) == 1:
                    cache.put(
                        j.key, j.fp, {"result": None, "meta": {"model": model, "error": p.error}}
                    )
        return bad

    print(f"  [D] {len(todo)} cells to detect in {len(batches)} batches", flush=True)
    for res in await asyncio.gather(*(one(b) for b in batches)):
        failed += res
    # rev 5: a refused 40-cell batch is first re-asked as batches of RETRY_CHUNK (only the chunk
    # holding the offending readout refuses again), then cell by cell
    if failed and RETRY_CHUNK > 1:
        chunks = [failed[i : i + RETRY_CHUNK] for i in range(0, len(failed), RETRY_CHUNK)]
        failed = [j for r in await asyncio.gather(*(one(c) for c in chunks)) for j in r]
    for _ in range(SINGLE_ATTEMPTS):
        if not failed:
            break
        res = await asyncio.gather(*(one([j]) for j in failed))
        failed = [j for r in res for j in r]
    return len(failed)


def _mention_key(cell_key: str, m: dict[str, Any], blind: bool) -> str:
    return f"{cell_key}:{'VB' if blind else 'V'}:{m['readout']}:{m['start']}-{m['end']}"


def verify_jobs(
    cells: Sequence[Cell],
    cache: Cache,
    djobs: dict[str, DJob],
    items: dict[str, dict[str, Any]],
    model: str,
    effort: str | None,
    blind: bool,
) -> list[VJob]:
    out = []
    version = V.BLIND_VERSION if blind else V.PROMPT_VERSION
    for c in cells:
        j = djobs.get(f"{c.key}:D")
        row = cache.get(j.key, j.fp) if j else None
        if not row:
            continue
        for m in row["result"]["mentions"]:
            if m["category"] != "lie_claim":
                continue
            canon = V.canonical(j.readouts[m["readout"]], m["quote"])
            prompt = None if blind else pressure_messages_text(items[c.id])
            system, _ = V.render([canon], prompt)
            fp = fingerprint(version, model, effort, system, canon, prompt)
            out.append(VJob(_mention_key(c.key, m, blind), fp, canon, c.id))
    return out


async def _verify(
    llm: LLM,
    cache: Cache,
    jobs: list[VJob],
    items: dict[str, dict[str, Any]],
    model: str,
    blind: bool,
) -> int:
    todo = [j for j in jobs if cache.get(j.key, j.fp) is None]
    if blind:
        random.Random(1).shuffle(todo)
        batches = [todo[i : i + V.BATCH] for i in range(0, len(todo), V.BATCH)]
    else:
        by_item: dict[str, list[VJob]] = defaultdict(list)
        for j in todo:
            by_item[j.tid].append(j)
        batches = [
            v[i : i + V.BATCH]
            for _, v in sorted(by_item.items())
            for i in range(0, len(v), V.BATCH)
        ]
    prefix = "batchVB" if blind else "batchV"

    async def one(bs: list[VJob]) -> list[VJob]:
        prompt = None if blind else pressure_messages_text(items[bs[0].tid])
        system, user = V.render([j.canon for j in bs], prompt)
        out = await llm.call(model=model, system=system, user=user, schema=V.SCHEMA)
        _usage_row(cache, prefix, user, out.usage, {"n_mentions": len(bs)})
        bad = []
        for j, r in zip(bs, V.parse_batch(out.result, len(bs)), strict=True):
            if r is not None:
                cache.put(j.key, j.fp, {"result": r, "meta": {"model": model}})
            else:
                bad.append(j)
                if len(bs) == 1:
                    cache.put(j.key, j.fp, {"result": None, "meta": {"model": model}})
        return bad

    print(
        f"  [{'VB' if blind else 'V'}] {len(todo)} mentions in {len(batches)} batches", flush=True
    )
    failed = [j for r in await asyncio.gather(*(one(b) for b in batches)) for j in r]
    for _ in range(SINGLE_ATTEMPTS):
        if not failed:
            break
        res = await asyncio.gather(*(one([j]) for j in failed))
        failed = [j for r in res for j in r]
    return len(failed)


def rebuild(
    name: str,
    cells: Sequence[Cell],
    cache: Cache,
    djobs: dict[str, DJob],
    sites: dict[str, list[dict[str, Any]]],
    labels: dict[str, dict[str, Any]],
    items: dict[str, dict[str, Any]],
    model: str,
    effort: str | None,
    out_dir: Path,
    suffix: str = "",
) -> dict[str, Any]:
    """``cells{suffix}.jsonl`` + ``summary{suffix}.json`` from the cache only (no API calls).
    Rev 5: a run restricted with ``--layers`` writes ``cells_L028.jsonl`` etc. (an arm is
    processed one layer at a time to bound memory); readers merge every ``cells*.jsonl``."""
    site_of = {(t, s["pos"]): s for t, ss in sites.items() for s in ss}
    vj = {
        blind: {
            j.key: j for j in verify_jobs(cells, cache, djobs, items, model, effort, blind=blind)
        }
        for blind in (False, True)
    }
    status = Counter()
    n_mentions = Counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f"cells{suffix}.jsonl.tmp"
    with tmp.open("w", encoding="utf-8") as fh:
        for c in sorted(cells, key=lambda c: (c.id, c.layer, c.pos)):
            lab = labels[c.id]
            site = site_of[(c.id, c.pos)]
            rec: dict[str, Any] = {
                "key": c.key,
                "task_id": c.id,
                "layer": c.layer,
                "pos": c.pos,
                "site_type": site["site_type"],
                "archetype": lab["archetype"],
                "label": lab["label"],
                "stratum": lab["stratum"],
                "n_readouts": len(D.readouts_of(c)),
            }
            j = djobs.get(f"{c.key}:D")
            if j is None:
                rec["status"], rec["mentions"] = "empty", []
            else:
                row = cache.get(j.key, j.fp, include_failed=True)
                if row is None:
                    rec["status"], rec["mentions"] = "pending", []
                elif row["result"] is None:
                    rec["status"], rec["mentions"] = "failed", []
                else:
                    rec["status"] = "ok"
                    ms = []
                    for m in row["result"]["mentions"]:
                        m = dict(m)
                        n_mentions[m["category"]] += 1
                        for blind, field in ((False, "verify"), (True, "verify_blind")):
                            m[field] = None
                            if m["category"] == "lie_claim":
                                k = _mention_key(c.key, m, blind)
                                vr = cache.get(k, vj[blind][k].fp) if k in vj[blind] else None
                                m[field] = vr["result"] if vr else None
                        ms.append(m)
                    rec["mentions"] = ms
            status[rec["status"]] += 1
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.rename(out_dir / f"cells{suffix}.jsonl")
    spend: dict[str, float] = defaultdict(float)
    for r in cache.rows():
        usd = float(((r.get("meta") or {}).get("usage") or {}).get("usd") or 0.0)
        spend[r["key"].split(":", 1)[0] if r["key"].startswith("batch") else "other"] += usd
    n_vfail = sum(
        1
        for blind in (False, True)
        for k, j in vj[blind].items()
        if (row := cache.get(k, j.fp, include_failed=True)) is not None and row["result"] is None
    )
    summary = {
        "arm": name,
        "layers": sorted({c.layer for c in cells}),
        "n_cells": len(cells),
        "status": dict(status),
        "n_mentions": dict(n_mentions),
        "n_verify_failed": n_vfail,
        "spend_usd": {k: round(v, 4) for k, v in spend.items()},
        "spend_total_usd": round(sum(spend.values()), 4),
        "versions": [D.PROMPT_VERSION, V.PROMPT_VERSION, V.BLIND_VERSION],
    }
    (out_dir / f"summary{suffix}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"[{name}] {json.dumps(summary)}", flush=True)
    return summary


async def run_mask_async(
    arms: list[tuple[str, Path]],
    out: Path,
    *,
    model: str,
    effort: str | None,
    rpm: float,
    concurrency: int,
    stages: Sequence[str],
    layers: Sequence[int] | None = None,
    limit: int | None = None,
) -> int:
    bad = set(stages) - set(STAGES)
    if bad:
        raise SystemExit(f"unknown stages {sorted(bad)}; known {STAGES}")
    sites = bank.load_sites()
    labels = bank.load_labels()
    items = {it["task_id"]: it for it in bank.load_items()}
    llm = LLM(rpm=rpm, concurrency=concurrency, effort=effort)
    n_failed = 0
    try:
        if stages:
            await llm.preflight(model)
        for name, path in arms:
            cells, got_layers = bank.load_arm_cells(path, sites, layers)
            if limit:
                cells = cells[:limit]
            print(f"[{name}] {len(cells)} cells, layers {got_layers}", flush=True)
            cache = Cache(out / name / "cache.jsonl")
            dj, n_empty = detect_jobs(cells, model, effort)
            djobs = {j.key: j for j in dj}
            print(f"[{name}] {n_empty} empty cells (no call)", flush=True)
            kind = "prose" if cells and cells[0].samples is not None else "tokens"
            if "detect" in stages:
                b = D.BATCH_PROSE if kind == "prose" else D.BATCH_TOKENS
                n_failed += await _detect(llm, cache, dj, model, b)
            for blind, st in ((False, "verify"), (True, "verify_blind")):
                if st in stages:
                    vjobs = verify_jobs(cells, cache, djobs, items, model, effort, blind)
                    n_failed += await _verify(llm, cache, vjobs, items, model, blind)
            suffix = "".join(f"_L{la:03d}" for la in got_layers) if layers else ""
            rebuild(
                name, cells, cache, djobs, sites, labels, items, model, effort, out / name, suffix
            )
            cache.close()
    finally:
        await llm.aclose()
    return n_failed


def run_mask(*args: Any, **kwargs: Any) -> int:
    return asyncio.run(run_mask_async(*args, **kwargs))
