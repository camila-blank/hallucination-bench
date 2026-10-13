"""agentic_misalignment driver (plan 0008): Stage D -> Stage S (J-lens) -> Stage M, plus intent.

One process judges every arm: one :class:`~hallucination_bench.llm.LLM` (one RPM pacer, one
semaphore). The chat pipeline (``wsbench_chat/run.py``) is not used: its cells assume the marker
sits in the response and its stage machine (span, verify, classify) does not apply.

Stages (``--stages detect,summ,map,intent``, default all; an unselected stage is read from the
cache only, and no stage pulls in another):

* ``detect`` (Stage D, ``amdetect-v1``): batches of up to 40 cells from a seeded shuffle of the
  arm's uncached non-empty cells (mixed across items and layers); per-cell rows ``{key}:D``;
  per-call usage rows ``batchD:{sha}``. Failed cells are retried as batches of one in the same
  run; after 2 failed single-cell attempts the latest partial parse is accepted
  (``meta.accepted_partial``). The index of single-cell failures is built from
  ``Cache.rows()`` once per run and updated on every put.
* ``summ`` (Stage S, J-lens only): for each bag-detected cell the interp-v1 summary
  (``summ:{key}``, the chat J-lens user text) and the detector on the summary (``{key}:Dsum``,
  a batch of one). J-lens mentions that enter scoring are the summary's.
* ``map`` (Stage M, ``claimmap-v2.7+am2``): ``{key}:M`` on every cell with a located scoring
  mention; arms in the order nla-rl-L42, s3d-rl600, jlens.
* ``intent`` (``amintent-v2``): one call per item, cached at
  ``<out>/agentic_misalignment/<model>/intent.jsonl`` (keys ``intent:{id}``), not per arm.

``cells.jsonl``, ``mentions.jsonl`` and ``summary.json`` are always regenerated from the cache by
:func:`rebuild` (no API calls).
"""

import asyncio
import hashlib
import json
import os
import random
import re
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hallucination_bench import summarizer
from hallucination_bench.cache import Cache, fingerprint
from hallucination_bench.evals.wsbench_chat.run import (
    MAP_INVALID_RETRIES,
    Models,
    model_slug,
)
from hallucination_bench.judges import claimmap_am as cam
from hallucination_bench.judges import claimmap_v2 as cm
from hallucination_bench.llm import LLM, JudgeConfigError, Usage
from hallucination_bench.readouts import Cell, decode_token, load_readouts, sha256_file
from hallucination_bench.units import Normalised, split_cell

from . import NAME, bank, detect, intent, score
from . import kinds as K

READOUTS_FILE = "agentic_misalignment.jsonl"
DEFAULT_READOUTS_ROOT = Path("/workspace/wsbench-paper-data/readouts")
DEFAULT_ARMS: tuple[str, ...] = ("s3d-rl600", "nla-rl-L42", "jlens")
STAGES: tuple[str, ...] = ("detect", "summ", "map", "intent")
MAP_ARM_ORDER: tuple[str, ...] = ("nla-rl-L42", "s3d-rl600", "jlens")
INTENT_INVALID_RETRIES = 2
DETECT_SEED = 0
SPEND_STAGES: tuple[str, ...] = ("detect", "summ", "map", "audit")
_LAYER_RE = re.compile(r"__L(\d{3})__p")
_EXCLUDED_PREFIXES = ("batchD:", "audit:")  # never per-cell rows


def parse_stages(s: str | None) -> frozenset[str]:
    if not s:
        return frozenset(STAGES)
    sel = [x.strip() for x in s.split(",") if x.strip()]
    bad = [x for x in sel if x not in STAGES]
    if bad:
        raise ValueError(f"unknown agentic_misalignment stages {bad} (known: {', '.join(STAGES)})")
    return frozenset(sel)


def resolve_arms(
    arm_specs: Sequence[str] | None, readouts_root: str | Path | None, arms: str | None
) -> list[tuple[str, Path]]:
    """``--arm name=path`` plus ``<root>/<arm>/agentic_misalignment.jsonl``, kept to ``--arms``
    (default: the three in-scope arms)."""
    found: list[tuple[str, Path]] = []
    for a in arm_specs or []:
        if "=" not in a:
            raise SystemExit(f"--arm expects name=path, got {a!r}")
        n, p = a.split("=", 1)
        found.append((n, Path(p)))
    root = Path(readouts_root) if readouts_root else DEFAULT_READOUTS_ROOT
    explicit = {n for n, _ in found}
    if root.is_dir():
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            f = d / READOUTS_FILE
            if f.exists() and d.name not in explicit:
                found.append((d.name, f))
    keep = arms.split(",") if arms else (list(explicit) if explicit else list(DEFAULT_ARMS))
    unknown = set(keep) - {n for n, _ in found}
    if unknown:
        raise SystemExit(f"--arms names not found (root {root}): {sorted(unknown)}")
    out = [(n, p) for n, p in found if n in set(keep)]
    if not out:
        raise SystemExit("no agentic_misalignment arms selected")
    return out


def am_models(models: Models) -> Models:
    if models.eval != NAME:
        raise ValueError(f"models.eval must be {NAME!r}, got {models.eval!r}")
    return models


def model_root(out_root: Path, models: Models) -> Path:
    return Path(out_root) / NAME / model_slug(models.judge)


def arm_dir(out_root: Path, models: Models, arm: str) -> Path:
    return model_root(out_root, models) / arm


def intent_cache_path(out_root: Path, models: Models) -> Path:
    return model_root(out_root, models) / "intent.jsonl"


# ---------------------------------------------------------------- arm context


@dataclass
class AMArm:
    name: str
    readouts: Path
    dir: Path
    cells: list[Cell]
    kind: str  # prose | tokens
    layers: list[int]
    cache: Cache
    items: dict[str, dict[str, Any]]
    sites: dict[str, dict[int, dict[str, Any]]]
    spend: dict[str, Usage] = field(default_factory=lambda: {s: Usage() for s in SPEND_STAGES})
    dump: Any = None
    n_dumped: dict[str, int] = field(default_factory=dict)
    index: dict[tuple[str, str], list[Any]] | None = None  # (key, fp) -> [n_single_failed, partial]


def cached_layers(cache: Cache) -> set[int]:
    return {
        int(m.group(1))
        for k in cache.keys()
        if not k.startswith(_EXCLUDED_PREFIXES) and (m := _LAYER_RE.search(k))
    }


def load_arm_cells(path: Path, layers: Sequence[int] | None) -> tuple[list[Cell], str, list[int]]:
    """Load and check one arm's AM readouts, then keep ``layers`` (default: all). The checks run
    on the whole file against the full 32-item bank first (non-empty; no unknown-id, malformed or
    off-grid rows; every layer has the full 50,452-site grid), so ``--items`` / ``--layers`` /
    ``--limit`` (applied afterwards) never trip them."""
    items = bank.load_items()
    cells, rep = load_readouts(path, positions=bank.positions(items))
    if rep.n_rows == 0:
        raise SystemExit(f"{path}: readouts file is empty")
    bad = {k: v for k, v in rep.skipped.items() if v}
    if bad:
        raise SystemExit(f"{path}: rows outside the agentic_misalignment grid: {bad}")
    per_layer: dict[int, int] = {}
    for c in cells:
        per_layer[c.layer] = per_layer.get(c.layer, 0) + 1
    short = {la: n for la, n in per_layer.items() if n != bank.N_SITES}
    if short:
        raise SystemExit(f"{path}: layers without the full {bank.N_SITES}-site grid: {short}")
    if layers is not None:
        keep = set(layers)
        cells = [c for c in cells if c.layer in keep]
    return cells, rep.kind or "prose", sorted({c.layer for c in cells})


def open_arm(
    name: str,
    readouts: Path,
    out_root: Path,
    models: Models,
    *,
    layers: Sequence[int] | None = None,
    readonly: bool = False,
) -> AMArm:
    d = arm_dir(out_root, models, name)
    if not readonly:
        d.mkdir(parents=True, exist_ok=True)
    cache = Cache(d / "cache.jsonl", readonly=readonly)
    cells, kind, file_layers = load_arm_cells(Path(readouts), layers)
    items = bank.load_items()
    return AMArm(
        name=name,
        readouts=Path(readouts),
        dir=d,
        cells=cells,
        kind=kind,
        layers=file_layers,
        cache=cache,
        items={it["id"]: it for it in items},
        sites=bank.sites_by_item(items),
    )


# ---------------------------------------------------------------- one cached call


@dataclass
class CallOut:
    status: str  # ok | failed | pending
    result: dict[str, Any] | None = None


def _dump(arm: AMArm | None, stage: str, key: str, system: str, user: str, limit: int) -> None:
    if arm is None or arm.dump is None or arm.n_dumped.get(stage, 0) >= limit:
        return
    arm.n_dumped[stage] = arm.n_dumped.get(stage, 0) + 1
    arm.dump.write(f"===== {key} [{stage}]\n----- system\n{system}\n----- user\n{user}\n\n")
    arm.dump.flush()


async def cached_call(
    cache: Cache,
    llm: LLM | None,
    *,
    stage: str,
    key: str,
    version: str,
    model: str,
    effort: str | None,
    system: str,
    user: str,
    schema: dict[str, Any],
    spend: Usage,
    validate: Callable[[dict[str, Any]], bool] | None = None,
    invalid_retries: int = 0,
) -> CallOut:
    """The chat ``cached_call`` semantics on a bare cache (the chat one takes a chat ``Arm``):
    a hit returns the row; ``llm=None`` reads only; an invalid answer is re-asked up to
    ``invalid_retries`` times, then cached as failed (``meta.raw``)."""
    fp = fingerprint(version, model, effort, system, user)
    row = cache.get(key, fp)
    if row is not None:
        return CallOut("ok", row["result"])
    if llm is None:
        failed = cache.get(key, fp, include_failed=True)
        return CallOut("failed" if failed is not None else "pending")
    usage = Usage()
    n_invalid = 0
    while True:
        out = await llm.call(model=model, system=system, user=user, schema=schema)
        spend.add(out.usage)
        usage.add(out.usage)
        r = out.result
        if r is None or validate is None or validate(r):
            break
        n_invalid += 1
        if n_invalid > invalid_retries:
            meta = _meta(stage, model, usage, n_invalid)
            cache.put(key, fp, {"result": None, "meta": {**meta, "raw": r}})
            return CallOut("failed")
    cache.put(key, fp, {"result": r, "meta": _meta(stage, model, usage, n_invalid)})
    return CallOut("ok" if r is not None else "failed", r)


def _meta(stage: str, model: str, usage: Usage, n_invalid: int) -> dict[str, Any]:
    meta: dict[str, Any] = {"stage": stage, "model": model, "usage": usage.to_dict()}
    if n_invalid:
        meta["n_invalid_attempts"] = n_invalid
    return meta


# ---------------------------------------------------------------- Stage D machinery


@dataclass
class DetectJob:
    key: str  # cache key ({cell}:D, {cell}:Dsum, audit:Dsum:{cell})
    readouts: list[str]
    hay_kind: str  # prose | tokens (how a literal backslash-n in a quote is retried)
    canon: str
    fp: str


def detect_fp(model: str, effort: str | None, canon: str) -> str:
    return fingerprint(detect.PROMPT_VERSION, model, effort, detect.SYSTEM, canon)


def make_job(key: str, readouts: list[str], hay_kind: str, model: str, effort: str | None):
    canon = detect.canonical(readouts)
    return DetectJob(key, readouts, hay_kind, canon, detect_fp(model, effort, canon))


def cell_job(cell: Cell, models: Models) -> DetectJob:
    hay = "prose" if cell.samples is not None else "tokens"
    return make_job(f"{cell.key}:D", detect.readouts_of(cell), hay, models.judge, models.effort)


def summary_job(cell_key: str, summary: str, models: Models, *, prefix: str = "") -> DetectJob:
    key = f"{prefix}Dsum:{cell_key}" if prefix else f"{cell_key}:Dsum"
    return make_job(key, [summary], "prose", models.judge, models.effort)


def job_status(cache: Cache, job: DetectJob) -> tuple[str, dict[str, Any] | None]:
    row = cache.get(job.key, job.fp)
    if row is not None:
        return "ok", row["result"]
    if cache.get(job.key, job.fp, include_failed=True) is not None:
        return "failed", None
    return "pending", None


class Detector:
    """Runs detector jobs against one cache with the plan-0008 terminal rule."""

    def __init__(
        self, cache: Cache, llm: LLM, models: Models, spend: Usage, stage: str, name: str = ""
    ):
        self.cache, self.llm, self.models, self.spend = cache, llm, models, spend
        self.stage, self.name = stage, name
        self.index: dict[tuple[str, str], list[Any]] = {}
        for row in cache.rows():  # built once per run ...
            self._note(row["key"], row.get("fingerprint", ""), row)
        self.arm: AMArm | None = None
        self.show_prompts = 0

    def _note(self, key: str, fp: str, row: dict[str, Any]) -> None:
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        if row.get("result") is not None or meta.get("batch_size") != 1:
            return
        ent = self.index.setdefault((key, fp), [0, None])
        ent[0] += 1
        if meta.get("partial") is not None:
            ent[1] = meta["partial"]

    def put(self, key: str, fp: str, payload: dict[str, Any]) -> None:
        self.cache.put(key, fp, payload)
        self._note(key, fp, payload)  # ... and updated on every put

    async def call(self, jobs: Sequence[DetectJob]) -> list[bool]:
        """One detector call over ``jobs``; caches every cell; returns per-job ok."""
        m = self.models
        system, user = detect.render_batch([j.canon for j in jobs])
        _dump(self.arm, self.stage, jobs[0].key, system, user, self.show_prompts)
        out = await self.llm.call(model=m.judge, system=system, user=user, schema=detect.SCHEMA)
        self.spend.add(out.usage)
        keys = [j.key for j in jobs]
        bkey = "batchD:" + hashlib.sha256("\n".join(keys).encode()).hexdigest()[:16]
        self.cache.put(
            bkey,
            fingerprint(detect.PROMPT_VERSION, m.judge, m.effort, detect.SYSTEM),
            {
                "result": {"cells": keys},
                "meta": {
                    "stage": self.stage,
                    "model": m.judge,
                    "usage": out.usage.to_dict(),
                    "batch_size": len(jobs),
                    "ok": out.result is not None,
                },
            },
        )
        bp = detect.parse_batch(out.result, [j.readouts for j in jobs], [j.hay_kind for j in jobs])
        oks = []
        for i, j in enumerate(jobs):
            meta: dict[str, Any] = {
                "stage": self.stage,
                "model": m.judge,
                "batch_size": len(jobs),
                "batch": bkey,
            }
            if i in bp.failed:
                part = bp.partial(i)
                if part is not None:
                    meta["partial"] = part
                self.put(j.key, j.fp, {"result": None, "meta": meta})
                oks.append(False)
            else:
                self.put(j.key, j.fp, {"result": bp.cells[i].to_result(), "meta": meta})
                oks.append(True)
        return oks

    async def single(self, job: DetectJob) -> bool:
        """Up to 2 single-cell attempts this run; after each failure, once the cell has 2 failed
        single-cell rows, accept its latest partial parse."""
        for _ in range(detect.SINGLE_ATTEMPTS):
            if (await self.call([job]))[0]:
                return True
            n, partial = self.index.get((job.key, job.fp), [0, None])
            if n >= detect.SINGLE_ATTEMPTS and partial is not None:
                meta = {
                    "stage": self.stage,
                    "model": self.models.judge,
                    "batch_size": 1,
                    "accepted_partial": True,
                }
                self.put(job.key, job.fp, {"result": partial, "meta": meta})
                return True
        return False

    def uncached(self, jobs: Iterable[DetectJob]) -> list[DetectJob]:
        return [j for j in jobs if self.cache.get(j.key, j.fp) is None]


async def _pool(factories: Iterable[Callable[[], Awaitable[Any]]], workers: int) -> None:
    it = iter(factories)

    async def worker() -> None:
        for f in it:
            await f()

    async with asyncio.TaskGroup() as tg:
        for _ in range(max(1, workers)):
            tg.create_task(worker())


class _Progress:
    def __init__(self, label: str, total: int, every: float = 60.0):
        self.label, self.total, self.every = label, total, every
        self.done = 0
        self.t0 = self.last = time.monotonic()

    def tick(self, n: int = 1) -> None:
        self.done += n
        now = time.monotonic()
        if now - self.last >= self.every or self.done == self.total:
            self.last = now
            rate = self.done / max(now - self.t0, 1e-9)
            eta = (self.total - self.done) / rate if rate else float("inf")
            print(
                f"  [{self.label} {self.done}/{self.total}] {rate * 60:.0f}/min "
                f"eta={eta / 60:.1f} min",
                flush=True,
            )


async def detect_jobs(
    pairs: Sequence[tuple[Detector, list[DetectJob]]],
    *,
    batch_size: int,
    workers: int,
    seed: int = DETECT_SEED,
    label: str = "detect",
) -> None:
    """Batch pass (a seeded shuffle per detector; batches interleaved round-robin across
    detectors), then the single-cell retries of every failed job, in the same run. With
    ``batch_size == 1`` (``:Dsum``) every job goes straight to the single-cell attempts."""
    per_det: list[list[tuple[Detector, list[DetectJob]]]] = []
    failed: list[tuple[Detector, DetectJob]] = []
    for det, jobs in pairs:
        todo = det.uncached(jobs)
        random.Random(f"amdetect:{seed}:{det.name}:{det.stage}").shuffle(todo)
        if batch_size > 1:
            per_det.append(
                [(det, todo[i : i + batch_size]) for i in range(0, len(todo), batch_size)]
            )
        else:
            failed += [(det, j) for j in todo]
    order: list[tuple[Detector, list[DetectJob]]] = []
    for i in range(max((len(b) for b in per_det), default=0)):
        order += [b[i] for b in per_det if i < len(b)]
    if order:
        prog = _Progress(f"{label} batches", len(order))

        async def run_batch(det: Detector, jobs: list[DetectJob]) -> None:
            oks = await det.call(jobs)
            failed.extend((det, j) for j, ok in zip(jobs, oks, strict=True) if not ok)
            prog.tick()

        await _pool((lambda d=d, js=js: run_batch(d, js) for d, js in order), workers)
    if failed:
        prog2 = _Progress(f"{label} single-cell", len(failed))

        async def run_single(det: Detector, job: DetectJob) -> None:
            await det.single(job)
            prog2.tick()

        await _pool((lambda d=d, j=j: run_single(d, j) for d, j in failed), workers)


# ---------------------------------------------------------------- per-cell state (judge + rebuild)


def summary_user(cell: Cell) -> str:
    """The interp-v1 user text, exactly as the chat J-lens arm builds it."""
    decoded = [decode_token(t) for t in cell.tokens or ()]
    return summarizer.render_user(summarizer.render_bag(decoded, cell.scores))


async def summary_call(
    cache: Cache, llm: LLM | None, cell: Cell, models: Models, spend: Usage, *, prefix: str = ""
) -> tuple[str, str | None]:
    out = await cached_call(
        cache,
        llm,
        stage="audit" if prefix else "summ",
        key=f"{prefix}summ:{cell.key}",
        version=summarizer.SUMMARIZER_PROMPT_VERSION,
        model=models.summ,
        effort=models.effort,
        system=summarizer.INTERP_SYSTEM,
        user=summary_user(cell),
        schema=summarizer.INTERP_SCHEMA,
        spend=spend,
        validate=lambda r: summarizer.interpretation_of(r) is not None,
    )
    text = summarizer.interpretation_of(out.result) if out.status == "ok" else None
    return out.status, text


@dataclass
class CellState:
    status: str  # empty | pending | failed | ok (the scoring detection)
    mentions: list[dict[str, Any]]
    samples: list[str] | None = None  # the readouts the map sees (prose samples[0] / summary)
    bag_status: str | None = None
    bag_mentions: list[dict[str, Any]] = field(default_factory=list)
    summary_status: str | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)  # ok detect rows (health)


def _row(cache: Cache, job: DetectJob) -> tuple[str, dict[str, Any] | None]:
    row = cache.get(job.key, job.fp)
    if row is not None:
        return "ok", row
    if cache.get(job.key, job.fp, include_failed=True) is not None:
        return "failed", None
    return "pending", None


async def cell_state(arm: AMArm, cell: Cell, models: Models) -> CellState:
    """The cell's cached detection outcome (cache only)."""
    job = cell_job(cell, models)
    if detect.is_empty(job.readouts):
        return CellState("empty", [])
    st, row = _row(arm.cache, job)
    found = list(row["result"]["mentions"]) if row is not None else []
    if arm.kind != "tokens":
        return CellState(st, found, job.readouts, rows=[row] if row else [])
    cs = CellState(st, [], None, bag_status=st, bag_mentions=found, rows=[row] if row else [])
    if st != "ok" or not found:
        return cs
    sst, text = await summary_call(arm.cache, None, cell, models, Usage())
    cs.summary_status = sst
    if text is None:
        cs.status = "failed" if sst == "failed" else "pending"
        return cs
    dst, drow = _row(arm.cache, summary_job(cell.key, text, models))
    cs.status = dst
    cs.samples = [text]
    if drow is not None:
        cs.rows.append(drow)
        cs.mentions = list(drow["result"]["mentions"])
    return cs


def map_prompt(arm: AMArm, cell: Cell, samples: list[str]) -> tuple[list[Any], str, str]:
    units, _ = split_cell(samples)
    units = cm.plain_units(units)
    item = arm.items[cell.id]
    site = arm.sites[cell.id][cell.pos]
    system, user = cam.render_prompt_am(
        bank.marked_prompt(item, int(site["char"])), item["response"], units, arm.kind
    )
    return units, system, user


async def map_call(
    arm: AMArm, cell: Cell, samples: list[str], llm: LLM | None, models: Models, dump: int = 0
) -> tuple[CallOut, list[Any]]:
    units, system, user = map_prompt(arm, cell, samples)
    if llm is not None:
        _dump(arm, "map", f"{cell.key}:M", system, user, dump)
    out = await cached_call(
        arm.cache,
        llm,
        stage="map",
        key=f"{cell.key}:M",
        version=cam.PROMPT_VERSION,
        model=models.mapper,
        effort=models.effort,
        system=system,
        user=user,
        schema=cam.SCHEMA,
        spend=arm.spend["map"],
        validate=lambda r: cm.validate(r, units),
        invalid_retries=MAP_INVALID_RETRIES,
    )
    return out, units


# ---------------------------------------------------------------- intent


def intent_call_args(item: dict[str, Any], models: Models) -> dict[str, Any]:
    system, user = intent.render(item)
    return {
        "stage": "intent",
        "key": f"intent:{item['id']}",
        "version": intent.PROMPT_VERSION,
        "model": models.judge,
        "effort": models.effort,
        "system": system,
        "user": user,
        "schema": intent.SCHEMA,
        "validate": lambda r: intent.validate(r, item),
        "invalid_retries": INTENT_INVALID_RETRIES,
    }


async def intent_stage(
    out_root: Path, models: Models, items: Sequence[dict[str, Any]], llm: LLM, workers: int
) -> Usage:
    spend = Usage()
    path = intent_cache_path(out_root, models)
    path.parent.mkdir(parents=True, exist_ok=True)
    with Cache(path) as cache:

        async def one(item: dict[str, Any]) -> None:
            await cached_call(cache, llm, spend=spend, **intent_call_args(item, models))

        await _pool((lambda it=it: one(it) for it in items), workers)
    return spend


def intent_results(out_root: Path, models: Models) -> dict[str, tuple[str, Any]]:
    """``{item id: (status, raw answer or None)}`` from the intent cache (no calls)."""
    path = intent_cache_path(out_root, models)
    cache = Cache(path, readonly=True)
    out = {}
    for item in bank.load_items():
        a = intent_call_args(item, models)
        fp = fingerprint(a["version"], a["model"], a["effort"], a["system"], a["user"])
        row = cache.get(a["key"], fp)
        if row is not None:
            out[item["id"]] = ("ok", row["result"])
        elif cache.get(a["key"], fp, include_failed=True) is not None:
            out[item["id"]] = ("failed", None)
        else:
            out[item["id"]] = ("pending", None)
    return out


def intent_status(out_root: Path, models: Models) -> dict[str, Any]:
    res = intent_results(out_root, models)
    by = {
        s: sorted(i for i, (st, _) in res.items() if st == s) for s in ("ok", "failed", "pending")
    }
    return {
        "n_ok": len(by["ok"]),
        "n_failed": len(by["failed"]),
        "n_pending": len(by["pending"]),
        "failed": by["failed"],
        "overrides": sorted(intent.load_overrides()),
    }


def effective_intents(out_root: Path, models: Models) -> dict[str, intent.Intent | None]:
    res = intent_results(out_root, models)
    ov = intent.load_overrides()
    return {it["id"]: intent.effective(it, res[it["id"]][1], ov) for it in bank.load_items()}


def non_acting_items(intents: dict[str, intent.Intent | None]) -> frozenset[str]:
    """Misaligned items whose (audited) rollout kinds do not include their design kind."""
    out = set()
    for it in bank.load_items():
        i = intents.get(it["id"])
        if it["arm"] == "misaligned" and i is not None:
            if not i.rollout_kinds & set(K.design_kinds(it["category"])):
                out.add(it["id"])
    return frozenset(out)


# ---------------------------------------------------------------- run


def select(arm: AMArm, item_ids: Sequence[str] | None, limit: int | None) -> list[Cell]:
    keep = set(item_ids) if item_ids else None
    out: list[Cell] = []
    for la in arm.layers:
        cl = [
            c
            for c in arm.cells
            if c.layer == la
            and (keep is None or c.id in keep)
            and not detect.is_empty(detect.readouts_of(c))
        ]
        out += cl[:limit] if limit else cl
    return out


def _write_arm_json(arm: AMArm, models: Models) -> None:
    (arm.dir / "arm.json").write_text(
        json.dumps(
            {
                "arm": arm.name,
                "eval": NAME,
                "readouts": str(arm.readouts),
                "kind": arm.kind,
                **models.to_dict(),
            },
            indent=1,
        )
        + "\n"
    )


async def run_am_async(
    arms: Sequence[tuple[str, Path]],
    out_root: Path,
    models: Models,
    *,
    layers: Sequence[int] | None = None,
    items: Sequence[str] | None = None,
    limit: int | None = None,
    rpm: float = 1500.0,
    concurrency: int = 64,
    stages: Iterable[str] | str | None = None,
    llm: LLM | None = None,
    show_prompts: int = 0,
    batch_size: int = detect.BATCH_SIZE,
    seed: int = DETECT_SEED,
) -> dict[str, Any]:
    """Judge the selected stages, then rebuild every opened arm. Returns ``{"arms": {name:
    summary}, "intent": intent_status}``."""
    am_models(models)
    sel = parse_stages(stages) if stages is None or isinstance(stages, str) else frozenset(stages)
    bad = sel - set(STAGES)
    if bad:
        raise ValueError(f"unknown agentic_misalignment stages {sorted(bad)}")
    sel_items = bank.load_items(items)
    arm_stages = sel - {"intent"}
    opened: list[AMArm] = []
    try:
        if arm_stages:
            for n, p in arms:
                a = open_arm(n, p, out_root, models, layers=layers)
                if not a.layers:
                    print(f"[{n}] no selected layer in this file: skipped")
                    a.cache.close()
                    continue
                opened.append(a)
                _write_arm_json(a, models)
                if show_prompts:
                    a.dump = (a.dir / "prompts_sample.txt").open("w", encoding="utf-8")
        sels = {a.name: select(a, items, limit) for a in opened}
        for a in opened:
            print(
                f"[{a.name}] kind={a.kind} layers={a.layers} cells_selected={len(sels[a.name])}",
                flush=True,
            )
        own = llm is None
        llm = llm or LLM(rpm=rpm, concurrency=concurrency, effort=models.effort)
        intent_spend = Usage()
        try:
            needed: set[str] = set()
            if sel & {"detect", "intent"} or ("summ" in sel and opened):
                needed.add(models.judge)
            if "summ" in sel and any(a.kind == "tokens" for a in opened):
                needed.add(models.summ)
            if "map" in sel and opened:
                needed.add(models.mapper)
            for m in sorted(needed):
                await llm.preflight(m)
                print(f"preflight ok: {m} (structured output, effort={models.effort})", flush=True)
            workers = 2 * concurrency
            if "intent" in sel:
                intent_spend = await intent_stage(out_root, models, sel_items, llm, workers)
            if "detect" in sel and opened:
                pairs = []
                for a in opened:
                    det = Detector(a.cache, llm, models, a.spend["detect"], "detect", a.name)
                    det.arm, det.show_prompts = a, show_prompts
                    pairs.append((det, [cell_job(c, models) for c in sels[a.name]]))
                await detect_jobs(pairs, batch_size=batch_size, workers=workers, seed=seed)
            if "summ" in sel:
                await summ_stage(
                    [a for a in opened if a.kind == "tokens"], sels, llm, models, workers, seed
                )
            if "map" in sel and opened:
                await map_stage(opened, sels, llm, models, workers, show_prompts)
        finally:
            if own:
                await llm.aclose()
        if "intent" in sel:
            print(f"[intent] this run: {intent_spend.report()}")
        for a in opened:
            print(f"[{a.name}] this run:")
            for st in SPEND_STAGES[:3]:
                print(f"    {st:7s} {a.spend[st].report()}")
    finally:
        for a in opened:
            if a.dump is not None:
                a.dump.close()
            a.cache.close()
    summaries = {}
    for a in opened:
        summaries[a.name] = await rebuild_async(a.name, a.readouts, out_root, models)
    return {"arms": summaries, "intent": intent_status(out_root, models)}


async def summ_stage(
    arms: Sequence[AMArm],
    sels: dict[str, list[Cell]],
    llm: LLM,
    models: Models,
    workers: int,
    seed: int,
) -> None:
    """Stage S: summaries of bag-detected J-lens cells, then ``:Dsum`` (batches of one)."""
    pairs: list[tuple[Detector, list[DetectJob]]] = []
    for a in arms:
        todo = []
        for c in sels[a.name]:
            _st, row = _row(a.cache, cell_job(c, models))
            if row is not None and row["result"]["mentions"]:
                todo.append(c)
        texts: dict[str, str] = {}
        prog = _Progress(f"{a.name} summ", len(todo))

        async def one(c: Cell, a: AMArm = a, texts: dict[str, str] = texts, prog=prog) -> None:
            _st, text = await summary_call(a.cache, llm, c, models, a.spend["summ"])
            if text is not None:
                texts[c.key] = text
            prog.tick()

        if todo:
            await _pool((lambda c=c: one(c) for c in todo), workers)
        det = Detector(a.cache, llm, models, a.spend["summ"], "summ", a.name)
        pairs.append((det, [summary_job(k, t, models) for k, t in texts.items()]))
    if pairs:
        await detect_jobs(pairs, batch_size=1, workers=workers, seed=seed, label="dsum")


async def map_stage(
    arms: Sequence[AMArm],
    sels: dict[str, list[Cell]],
    llm: LLM,
    models: Models,
    workers: int,
    show_prompts: int,
) -> None:
    """Stage M on every selected cell with a located scoring mention, arm by arm in
    :data:`MAP_ARM_ORDER` (the first complete arm lands early)."""
    rank = {n: i for i, n in enumerate(MAP_ARM_ORDER)}
    ordered = sorted(arms, key=lambda a: rank.get(a.name, len(rank)))
    work: list[tuple[AMArm, Cell, list[str]]] = []
    for a in ordered:
        for c in sels[a.name]:
            cs = await cell_state(a, c, models)
            if cs.status == "ok" and cs.mentions and cs.samples:
                work.append((a, c, cs.samples))
    if not work:
        return
    prog = _Progress("map", len(work))

    async def one(a: AMArm, c: Cell, samples: list[str]) -> None:
        await map_call(a, c, samples, llm, models, show_prompts)
        prog.tick()

    await _pool((lambda a=a, c=c, s=s: one(a, c, s) for a, c, s in work), workers)


def run_am(*args: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        return asyncio.run(run_am_async(*args, **kwargs))
    except* JudgeConfigError as eg:
        raise eg.exceptions[0] from None


# ---------------------------------------------------------------- rebuild (no API calls)

HEALTH: tuple[str, ...] = (
    "n_cells",
    "n_empty",
    "n_failed_detect",
    "n_pending_detect",
    "n_failed_summary",
    "n_pending_summary",
    "n_failed_dsum",
    "n_pending_dsum",
    "n_map_ok",
    "n_failed_map",
    "n_pending_map",
    "n_accepted_partial",
    "n_unlocated",
    "n_bad_enum",
    *(f"map_{k}" for k in cm.PARSE_COUNTS),
)
FAILED_KEYS: tuple[str, ...] = (
    "n_failed_detect",
    "n_failed_summary",
    "n_failed_dsum",
    "n_failed_map",
)


def _spend(cache: Cache) -> dict[str, dict[str, Any]]:
    """Every appended row's usage (all ``batchD:`` rows included), by stage."""
    tot = {s: Usage() for s in (*SPEND_STAGES, "other")}
    for row in cache.rows():
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        if not meta.get("usage"):
            continue
        st = meta.get("stage")
        if st not in tot:
            k = row["key"]
            st = (
                "audit"
                if k.startswith("audit:")
                else "summ"
                if k.startswith("summ:") or k.endswith(":Dsum")
                else "map"
                if k.endswith(":M")
                else "detect"
            )
        tot[st].add(meta["usage"])
    out = {k: v.to_dict() for k, v in tot.items()}
    out["total_usd"] = sum(v.usd for v in tot.values())  # type: ignore[assignment]
    return out


def rebuild(
    name: str, readouts: Path, out_root: Path, models: Models, *, n_boot: int = score.N_BOOT
) -> dict[str, Any]:
    return asyncio.run(rebuild_async(name, readouts, out_root, models, n_boot=n_boot))


async def rebuild_async(
    name: str, readouts: Path, out_root: Path, models: Models, *, n_boot: int = score.N_BOOT
) -> dict[str, Any]:
    """Regenerate ``cells.jsonl``, ``mentions.jsonl`` and ``summary.json`` from the cache over
    every judged layer (layers with any per-cell cache row; with an empty cache, every layer of
    the file, all pending). Streams records; keeps slim score records only."""
    am_models(models)
    d = arm_dir(out_root, models, name)
    cache = Cache(d / "cache.jsonl", readonly=True)
    lay = sorted(cached_layers(cache)) or None
    cells, kind, layers = load_arm_cells(Path(readouts), lay)
    items = bank.load_items()
    arm = AMArm(
        name=name,
        readouts=Path(readouts),
        dir=d,
        cells=cells,
        kind=kind,
        layers=layers,
        cache=cache,
        items={it["id"]: it for it in items},
        sites=bank.sites_by_item(items),
    )
    intents = effective_intents(out_root, models)
    non_acting = non_acting_items(intents)
    convs = {it["id"]: Normalised(bank.conversation(it)) for it in items}
    h = dict.fromkeys(HEALTH, 0)
    slim: list[score.CellSlim] = []
    mentions: list[dict[str, Any]] = []
    d.mkdir(parents=True, exist_ok=True)
    tmp_c = d / f"cells.jsonl.{os.getpid()}.tmp"
    tmp_m = d / f"mentions.jsonl.{os.getpid()}.tmp"
    with tmp_c.open("w", encoding="utf-8") as fc, tmp_m.open("w", encoding="utf-8") as fm:
        for c in cells:
            cs = await cell_state(arm, c, models)
            h["n_cells"] += 1
            rec: dict[str, Any] = {
                "key": c.key,
                "id": c.id,
                "layer": c.layer,
                "pos": c.pos,
                "status": cs.status,
                "n_mentions": len(cs.mentions),
            }
            if cs.status == "empty":
                h["n_empty"] += 1
                slim.append(score.CellSlim(c.id, c.layer, "empty", 0))
                fc.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            dst = cs.bag_status if kind == "tokens" else cs.status
            if dst in ("failed", "pending"):
                h[f"n_{dst}_detect"] += 1
            if kind == "tokens":
                rec["bag"] = {"status": cs.bag_status, "n_mentions": len(cs.bag_mentions)}
                rec["summary_status"] = cs.summary_status
                if cs.summary_status in ("failed", "pending"):
                    h[f"n_{cs.summary_status}_summary"] += 1
                elif cs.summary_status == "ok" and cs.status in ("failed", "pending"):
                    h[f"n_{cs.status}_dsum"] += 1
            for row in cs.rows:
                res = row["result"]
                h["n_unlocated"] += int(res.get("n_unlocated") or 0)
                h["n_bad_enum"] += int(res.get("n_bad_enum") or 0)
                meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
                h["n_accepted_partial"] += bool(meta.get("accepted_partial"))
            map_status = None
            if cs.status == "ok" and cs.mentions and cs.samples:
                out, units = await map_call(arm, c, cs.samples, None, models)
                map_status = out.status
                item = arm.items[c.id]
                site = arm.sites[c.id][c.pos]
                if out.status == "pending":
                    h["n_pending_map"] += 1
                    segs = None
                else:
                    h["n_map_ok" if out.status == "ok" else "n_failed_map"] += 1
                    raw = out.result if out.status == "ok" else None
                    seg_objs, counts = cm.parse(raw, units, bank.conversation(item))
                    segs = [s.to_dict() for s in seg_objs]
                    for k2, v in counts.items():
                        h[f"map_{k2}"] += int(v)
                    rec["map"] = {
                        "status": out.status,
                        "units": [u.to_dict() for u in units],
                        "segments": segs,
                        "counts": counts,
                        "rationale": str(raw.get("rationale") or "") if raw else "",
                    }
                rec["samples"] = cs.samples
                rec["mentions"] = cs.mentions
                ms = score.mention_records(
                    arm=name,
                    cell_key=c.key,
                    item=item,
                    layer=c.layer,
                    pos=c.pos,
                    char=int(site["char"]),
                    mentions=cs.mentions,
                    segments=segs,
                    intent=intents.get(c.id),
                    conv=convs[c.id],
                )
                for m in ms:
                    fm.write(json.dumps(m, ensure_ascii=False) + "\n")
                mentions += ms
            rec["map_status"] = map_status
            slim.append(
                score.CellSlim(
                    c.id,
                    c.layer,
                    cs.status,
                    len(cs.mentions),
                    map_status,
                    cs.bag_status,
                    len(cs.bag_mentions),
                    cs.summary_status,
                )
            )
            fc.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp_c.replace(d / "cells.jsonl")
    tmp_m.replace(d / "mentions.jsonl")
    spend = _spend(cache)
    cache.close()
    h["n_actor_remapped"] = sum(1 for m in mentions if m["actor_remapped"])
    h["n_mentions_no_intent"] = sum(1 for m in mentions if m["tier"] == score.NO_INTENT)
    summary = {
        "config": {
            "eval": NAME,
            "arm": name,
            "kind": kind,
            "models": models.to_dict(),
            "prompt_versions": {
                "detect": detect.PROMPT_VERSION,
                "summary": summarizer.SUMMARIZER_PROMPT_VERSION if kind == "tokens" else None,
                "map": cam.PROMPT_VERSION,
                "intent": intent.PROMPT_VERSION,
            },
            "bank_source": bank.BANK_SOURCE,
            "bank_commit": bank.BANK_COMMIT,
            "readouts": str(readouts),
            "readouts_sha256": sha256_file(Path(readouts)),
            "judged_layers": layers,
            "provisional": True,  # no held-out gate: the claim map's gate is chat gold
            "intent_overrides": sorted(intent.load_overrides()),
            "non_acting_items": sorted(non_acting),
        },
        "health": h,
        "intent": intent_status(out_root, models),
        "spend": spend,
        **score.summarize(slim, mentions, kind=kind, non_acting=non_acting, n_boot=n_boot),
    }
    (d / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def n_failed(summary: dict[str, Any]) -> int:
    """Failed per-cell calls of one arm summary (exit 4 when non-zero)."""
    return sum(int(summary["health"].get(k, 0)) for k in FAILED_KEYS)
