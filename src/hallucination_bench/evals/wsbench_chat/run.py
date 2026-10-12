"""wsbench_chat pipeline: cells -> (summarise) -> span -> verify -> map -> classify -> cells.jsonl.

One process judges every arm: one :class:`~hallucination_bench.llm.LLM` (one RPM pacer, one
concurrency semaphore) and a pool of asyncio workers that each take a cell through all its
stages in order (stages pipeline per cell; this deliberately replaces the source's
stage-by-stage batching in ``run_calls``).

The same :func:`process_cell` runs in two modes:

* **judge** (``llm`` given): a cache miss calls the API and stores the result;
* **rebuild** (``llm=None``): cache only, no API calls. ``cells.jsonl`` and ``summary.json`` are
  always regenerated this way, over every judged layer, at the end of ``hbench run``, by
  ``hbench run --rebuild`` and by ``hbench report``. Every prompt is re-rendered, so a cached row
  is used only when its key AND fingerprint match the current prompt; a stale classify row (e.g.
  from a contradicted-only run) is never attached to the current S-ids: those spans are
  ``unknown``.

Cache keys (per-arm ``<arm>/cache.jsonl``): ``summ:{key}`` (summary), ``{key}`` (span),
``{key}:V`` (verify), ``{key}:C`` (classify), ``{key}:M`` (claim map, plan 0002), with
``key = {id}__L{layer:03d}__p{pos}``. Fingerprints = (prompt version, model, effort, system,
user).

Stage selection (``stages``, the CLI's ``--stages``): a selected stage calls the API on a cache
miss; an unselected one is read from the cache only. Dependencies are added automatically
(:func:`expand_stages`): ``map`` needs ``summary`` (token arms), and ``span`` only under
``--anchor v5c``; ``classify`` needs ``verify``; ``verify`` needs ``span``. The Python default
(``stages=None``) keeps plan 0001's behaviour (summary, span, verify, classify unless
``classify=False``); the CLI default is every stage.

Claim map modes (plan rev 13, ``Models.map_anchor``): **single-call** (``none``, the default)
splits the readouts with no established spans and makes one ``:M`` call; it does not need the
span stage, and the verify stage is not part of it. For the v5c comparison it calls the span
stage on a fixed, seeded 10% item subsample (:func:`in_v5c_subsample`) when ``map`` is
selected; cached span results are used for every cell. ``v5c`` (``--anchor v5c``) restores
plan rev 12: spans first, located v5c wrong strings become ESTABLISHED units.
"""

import asyncio
import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hallucination_bench import WSBENCH_PUB_COMMIT, summarizer
from hallucination_bench.cache import Cache, fingerprint
from hallucination_bench.judges import claimmap_v2 as cm
from hallucination_bench.judges import taxonomy_v1 as tv1
from hallucination_bench.judges import v5c
from hallucination_bench.llm import LLM, JudgeConfigError, Usage
from hallucination_bench.readouts import (
    Cell,
    LoadReport,
    decode_token,
    load_readouts,
    missing_cells,
    sha256_file,
)
from hallucination_bench.units import split_cell

from . import NAME, bank

STAGES: tuple[str, ...] = ("summary", "span", "verify", "classify", "map")
V1_STAGES: tuple[str, ...] = ("summary", "span", "verify", "classify")
STAGE_DEPS: dict[str, tuple[str, ...]] = {
    "summary": (),
    "span": ("summary",),
    "verify": ("summary", "span"),
    "classify": ("summary", "span", "verify"),
    "map": ("summary",),  # single-call; --anchor v5c adds span (MAP_DEPS_V5C)
}
MAP_DEPS_V5C: tuple[str, ...] = ("summary", "span")
V5C_SUBSAMPLE_SEED = 0
V5C_SUBSAMPLE_FRACTION = 0.10


def in_v5c_subsample(item_id: str, seed: int = V5C_SUBSAMPLE_SEED) -> bool:
    """The fixed, seeded 10% item subsample of the single-call v5c comparison."""
    h = hashlib.sha256(f"v5c-subsample:{seed}:{item_id}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64 < V5C_SUBSAMPLE_FRACTION


def expand_stages(
    stages: Sequence[str] | None, *, classify: bool = True, anchor: str = "none"
) -> frozenset[str]:
    """The stages that may call the API: the selection plus its dependencies. ``None`` = plan
    0001's default (every v1 stage; classify only if ``classify``). ``map`` needs ``span``
    only under ``anchor="v5c"``."""
    anchored = cm.check_anchor(anchor)
    if stages is None:
        sel = [s for s in V1_STAGES if classify or s != "classify"]
    else:
        sel = list(stages)
    bad = [s for s in sel if s not in STAGES]
    if bad:
        raise ValueError(f"unknown stages {bad} (known: {', '.join(STAGES)})")
    out = set(sel)
    for s in sel:
        out.update(MAP_DEPS_V5C if s == "map" and anchored else STAGE_DEPS[s])
    return frozenset(out)


HAL_K = v5c.HAL_K
_LAYER_RE = re.compile(r"__L(\d{3})__p")


@dataclass
class Models:
    judge: str = "claude-opus-5-5"
    summarizer: str | None = None
    classifier: str | None = None
    effort: str | None = "low"
    map_model: str | None = None
    map_anchor: str = "none"  # claim map: none (single-call, default) | v5c (plan rev 12)

    @property
    def summ(self) -> str:
        return self.summarizer or self.judge

    @property
    def clf(self) -> str:
        return self.classifier or self.judge

    @property
    def mapper(self) -> str:
        return self.map_model or self.judge

    def to_dict(self) -> dict[str, Any]:
        return {
            "judge": self.judge,
            "summarizer": self.summ,
            "classifier": self.clf,
            "map_model": self.mapper,
            "map_anchor": self.map_anchor,
            "effort": self.effort,
        }


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def arm_dir(out_root: Path, models: Models, arm: str) -> Path:
    return Path(out_root) / NAME / model_slug(models.judge) / arm


def cell_key(id_: str, layer: int, pos: int) -> str:
    return f"{id_}__L{layer:03d}__p{pos}"


def stage_of_key(key: str) -> str:
    if key.startswith("summ:"):
        return "summary"
    if key.endswith(":V"):
        return "verify"
    if key.endswith(":C"):
        return "classify"
    if key.endswith(":M"):
        return "map"
    return "span"


# ---------------------------------------------------------------- arm context


@dataclass
class Arm:
    name: str
    readouts: Path
    dir: Path
    cells: list[Cell]
    report: LoadReport
    file_layers: list[int]
    cache: Cache
    items: dict[str, dict[str, Any]]
    sites: dict[str, dict[int, dict[str, Any]]]
    spend: dict[str, Usage] = field(default_factory=lambda: {s: Usage() for s in STAGES})
    dump: Any = None  # open prompt-sample file handle (judge mode, --show-prompts)

    @property
    def kind(self) -> str:
        return self.report.kind or "prose"

    @property
    def k(self) -> int:
        return 1 if self.kind == "tokens" else HAL_K


def open_arm(
    name: str, readouts: Path, out_root: Path, models: Models, *, readonly: bool = False
) -> Arm:
    items = bank.load_items()
    pos = bank.positions(items)
    cells, rep = load_readouts(readouts, positions=pos)
    d = arm_dir(out_root, models, name)
    if not readonly:
        d.mkdir(parents=True, exist_ok=True)
    cache = Cache(d / "cache.jsonl", readonly=readonly)
    return Arm(
        name=name,
        readouts=Path(readouts),
        dir=d,
        cells=cells,
        report=rep,
        file_layers=sorted({c.layer for c in cells}),
        cache=cache,
        items={it["id"]: it for it in items},
        sites=bank.sites_by_item(items),
    )


def cached_layers(arm: Arm) -> set[int]:
    return {int(m.group(1)) for k in arm.cache.keys() if (m := _LAYER_RE.search(k))}


# ---------------------------------------------------------------- one cached call


@dataclass
class CallOut:
    status: str  # ok | failed | pending
    result: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None  # an invalid answer (failed validation)


async def cached_call(
    arm: Arm,
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
    validate: Callable[[dict[str, Any]], bool] | None = None,
    dump: bool = False,
    invalid_retries: int = 0,
) -> CallOut:
    """One cached call. An answer that fails ``validate`` is retried within the run up to
    ``invalid_retries`` times (plan rev 14: 2 for the claim map) before it is cached as failed;
    the usage of every attempt is summed into the row's meta."""
    fp = fingerprint(version, model, effort, system, user)
    if dump and arm.dump is not None:
        arm.dump.write(f"===== {key} [{stage}] model={model} version={version}\n")
        arm.dump.write(f"----- system\n{system}\n----- user\n{user}\n\n")
        arm.dump.flush()
    row = arm.cache.get(key, fp)
    if row is not None:
        return CallOut("ok", row["result"])
    if llm is None:  # rebuild: cache only
        failed = arm.cache.get(key, fp, include_failed=True)
        if failed is None:
            return CallOut("pending")
        meta = failed.get("meta") or {}
        return CallOut("failed", raw=meta.get("raw") if isinstance(meta, dict) else None)
    usage = Usage()
    n_invalid = 0
    while True:
        out = await llm.call(model=model, system=system, user=user, schema=schema)
        arm.spend[stage].add(out.usage)
        usage.add(out.usage)
        r = out.result
        if r is None or validate is None or validate(r):
            break
        n_invalid += 1
        if n_invalid > invalid_retries:  # invalid after every in-run retry: cache as failed
            meta = _meta(stage, model, usage, n_invalid)
            arm.cache.put(key, fp, {"result": None, "meta": {**meta, "raw": r}})
            return CallOut("failed", raw=r)
    meta = _meta(stage, model, usage, n_invalid)
    arm.cache.put(key, fp, {"result": r, "meta": meta})
    return CallOut("ok" if r is not None else "failed", r)


def _meta(stage: str, model: str, usage: Usage, n_invalid: int) -> dict[str, Any]:
    meta: dict[str, Any] = {"stage": stage, "model": model, "usage": usage.to_dict()}
    if n_invalid:
        meta["n_invalid_attempts"] = n_invalid
    return meta


MAP_INVALID_RETRIES = 2  # plan rev 14


# ---------------------------------------------------------------- one cell


def _verdict_rows(verdicts: Sequence[v5c.ReadoutVerdict]) -> list[dict[str, Any]]:
    return [
        {
            "idx": v.idx,
            "kind": v.kind,
            "class": v.cls,
            "revoked": v.revoked,
            "wrong": v.wrong,
            "off_topic": v.off_topic,
            "n_unverified": v.n_unverified,
            "claims": v.claims,
            "quotes": None,
        }
        for v in verdicts
    ]


async def process_cell(
    arm: Arm,
    cell: Cell,
    llm: LLM | None,
    models: Models,
    *,
    classify: bool = True,
    dump: bool = False,
    stages: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Take one cell through every stage (judge mode) or read it back (rebuild mode); return
    its merged ``cells.jsonl`` record. In judge mode only the ``stages`` (default: plan 0001's)
    call the API; the others are read from the cache."""
    stages = (
        stages
        if stages is not None
        else expand_stages(None, classify=classify, anchor=models.map_anchor)
    )

    def via(stage: str) -> LLM | None:
        return llm if stage in stages else None

    item = arm.items[cell.id]
    site = arm.sites[cell.id][cell.pos]
    key = cell.key
    rec: dict[str, Any] = {
        "key": key,
        "id": cell.id,
        "layer": cell.layer,
        "pos": cell.pos,
        "site_kind": site["kind"],
        "status": "pending",
        "n_samples": 0,
        "samples": None,
        "stages": dict.fromkeys(STAGES, None),
        "verdict": None,
        "n_offtopic_overlap_quotes": 0,
        "spans": [],
        "classify_counts": None,
        "map": None,
    }
    if cell.empty:
        rec["status"] = "empty"
        return rec
    # ---- 1. samples (summarise token bags)
    if arm.kind == "tokens":
        decoded = [decode_token(t) for t in cell.tokens or ()]
        rec["tokens"] = decoded
        bag = summarizer.render_bag(decoded, cell.scores)
        out = await cached_call(
            arm,
            via("summary"),
            stage="summary",
            key=f"summ:{key}",
            version=summarizer.SUMMARIZER_PROMPT_VERSION,
            model=models.summ,
            effort=models.effort,
            system=summarizer.INTERP_SYSTEM,
            user=summarizer.render_user(bag),
            schema=summarizer.INTERP_SCHEMA,
            validate=lambda r: summarizer.interpretation_of(r) is not None,
            dump=dump,
        )
        rec["stages"]["summary"] = out.status
        text = summarizer.interpretation_of(out.result) if out.status == "ok" else None
        if text is None:
            rec["status"] = "failed_summary" if out.status == "failed" else "pending"
            return rec
        samples = [text]
        rec["n_samples"] = 1
    else:
        samples = [s for s in cell.samples or () if s.strip()][:HAL_K]
        rec["n_samples"] = len(samples)
    rec["samples"] = samples
    anchored = cm.check_anchor(models.map_anchor)
    # ---- claim map, single-call (default): needs only the samples
    if not anchored:
        await _map_stage(arm, cell, item, site, samples, None, rec, via("map"), models, dump)
    # ---- 2. span stage (v5c, verbatim); single-call map runs also call it on the fixed 10%
    # item subsample for the v5c comparison
    span_llm = via("span")
    if span_llm is None and not anchored and "map" in stages and in_v5c_subsample(cell.id):
        span_llm = llm
    system, user = v5c.judge_prompt(item, site, samples)
    out = await cached_call(
        arm,
        span_llm,
        stage="span",
        key=key,
        version=v5c.PROMPT_VERSION,
        model=models.judge,
        effort=models.effort,
        system=system,
        user=user,
        schema=v5c.JUDGE_SCHEMA,
        validate=lambda r: v5c.fully_judged(r, samples),
        dump=dump,
    )
    rec["stages"]["span"] = out.status
    verdicts = v5c.parse_verdict(out.result, samples) if out.status == "ok" else None
    if verdicts is None:
        rec["status"] = "failed_span" if out.status == "failed" else "pending"
        return rec
    rec["status"] = "judged"
    rows = _verdict_rows(verdicts)
    rec["verdict"] = rows
    # ---- 3. verify stage (v5c-chat-verify-v1, verbatim): every span-judged cell
    wrong = [list(v.wrong) for v in verdicts]
    system, user = v5c.verify_prompt(item, site, samples, wrong)
    vout = await cached_call(
        arm,
        via("verify"),
        stage="verify",
        key=f"{key}:V",
        version=v5c.VERIFY_PROMPT_VERSION,
        model=models.judge,
        effort=models.effort,
        system=system,
        user=user,
        schema=v5c.VERIFY_SCHEMA,
        validate=lambda r: v5c.tally_claims(r, samples, wrong) is not None,
        dump=dump,
    )
    rec["stages"]["verify"] = vout.status
    lists = v5c.checked_claim_lists(vout.result, samples, wrong) if vout.status == "ok" else None
    tallies = v5c.tally_claims(vout.result, samples, wrong) if lists is not None else None
    # ---- spans to classify (specific-class readouts only; class always from v5c)
    spans: list[tv1.SpanIn] = []
    n_off = 0
    for i, v in enumerate(verdicts):
        if tallies is not None and lists is not None:
            rows[i]["claims"] = tallies[i]
            v.claims = tallies[i]
        if v.cls not in v5c.SPECIFIC_CLASSES:
            if lists is not None:
                rows[i]["quotes"] = _quotes_row(lists[i], [], [])
            continue
        spans += [tv1.SpanIn(i, t, "contradicted") for t in v.wrong]
        if lists is not None:
            unsup, disp, n = v5c.classify_quotes(lists[i], v.off_topic)
            n_off += n
            rows[i]["quotes"] = _quotes_row(lists[i], unsup, disp)
            spans += [tv1.SpanIn(i, q, "disputed") for q in disp]
            spans += [tv1.SpanIn(i, q, "unsupported") for q in unsup]
    rec["n_offtopic_overlap_quotes"] = n_off
    # ---- 4. claim map anchored on v5c (--anchor v5c): every judged cell, whatever its class
    if anchored:
        await _map_stage(arm, cell, item, site, samples, verdicts, rec, via("map"), models, dump)
    # ---- 5. classify stage (taxonomy-v1): after verify tallied or failed for good
    if not spans:
        rec["stages"]["classify"] = "not_needed"
        return rec
    if vout.status == "pending" or not classify:
        rec["stages"]["classify"] = "pending"
        rec["spans"] = [lab.to_dict() for lab in tv1.unknown_labels(spans)]
        return rec
    readouts = [(i, samples[i]) for i, v in enumerate(verdicts) if v.cls in v5c.SPECIFIC_CLASSES]
    system, user = tv1.render_prompt(
        item["prompt"], bank.marked_response(item, site), readouts, spans
    )
    cout = await cached_call(
        arm,
        via("classify"),
        stage="classify",
        key=f"{key}:C",
        version=tv1.PROMPT_VERSION,
        model=models.clf,
        effort=models.effort,
        system=system,
        user=user,
        schema=tv1.SCHEMA,
        validate=lambda r: tv1.validate(r, spans),
        dump=dump,
    )
    rec["stages"]["classify"] = cout.status
    raw = cout.result if cout.status == "ok" else cout.raw  # an invalid answer: lenient read
    labels, counts = tv1.parse(raw, spans)
    rec["spans"] = [lab.to_dict() for lab in labels]
    rec["classify_counts"] = counts
    if isinstance(raw, dict):
        rec["classify_rationale"] = str(raw.get("rationale") or "")
    return rec


async def _map_stage(
    arm: Arm,
    cell: Cell,
    item: dict[str, Any],
    site: dict[str, Any],
    samples: list[str],
    verdicts: Sequence[v5c.ReadoutVerdict] | None,
    rec: dict[str, Any],
    llm: LLM | None,
    models: Models,
    dump: bool,
) -> None:
    """Split the readouts into units, call / read the ``:M`` stage and attach the parsed
    segments to ``rec["map"]``. Anchored (``verdicts`` given): established spans = v5c verified
    wrong strings, revoked or not. Single-call (``verdicts=None``): no established spans."""
    anchor = models.map_anchor
    wrong = [list(v.wrong) for v in verdicts] if verdicts is not None else [[] for _ in samples]
    units, unloc = split_cell(samples, wrong)
    if verdicts is None:
        units = cm.plain_units(units)
    system, user = cm.render_prompt(
        item["prompt"], bank.marked_response(item, site), units, kind=arm.kind, anchor=anchor
    )
    mout = await cached_call(
        arm,
        llm,
        stage="map",
        key=f"{cell.key}:M",
        version=cm.PROMPT_VERSION,
        model=models.mapper,
        effort=models.effort,
        system=system,
        user=user,
        schema=cm.schema_for(anchor),
        validate=lambda r: cm.validate(r, units),
        dump=dump,
        invalid_retries=MAP_INVALID_RETRIES,
    )
    rec["stages"]["map"] = mout.status
    raw = mout.result if mout.status == "ok" else None
    conv = cm.conversation_text(item["prompt"], bank.marked_response(item, site))
    segs, counts = cm.parse(raw, units, conv)
    counts["n_unlocated_established"] = sum(unloc)
    rec["map"] = {
        "anchor": anchor,
        "units": [u.to_dict() for u in units],
        "segments": [s.to_dict() for s in segs],
        "counts": counts,
        "rationale": str(raw.get("rationale") or "") if isinstance(raw, dict) else "",
    }


def _quotes_row(cl: v5c.ClaimLists, unsup: list[str], disp: list[str]) -> dict[str, Any]:
    return {
        "true": cl.true,
        "unverifiable": cl.unsupported,
        "disputed": cl.disputed,
        "off_topic": cl.off_topic,
        "classified_unsupported": unsup,
        "classified_disputed": disp,
    }


# ---------------------------------------------------------------- arms: selection and run


@dataclass
class Selection:
    arm: Arm
    layers: list[int]
    cells: list[Cell]  # the cells to judge this run (non-empty, sliced)
    n_missing: int


def select(arm: Arm, layers: Sequence[int] | None, limit: int | None) -> Selection:
    req = sorted(set(layers)) if layers else arm.file_layers
    sel_layers = [la for la in req if la in arm.file_layers]
    absent = [la for la in req if la not in arm.file_layers]
    if layers and absent:
        print(f"[{arm.name}] layers {absent} not in the readouts file (has {arm.file_layers})")
    items = list(arm.items.values())
    expected = bank.expected_grid(items, sel_layers)
    in_layers = [c for c in arm.cells if c.layer in sel_layers]
    missing = missing_cells(in_layers, expected)
    todo: list[Cell] = []
    for la in sel_layers:
        cl = [c for c in in_layers if c.layer == la and not c.empty]
        todo += cl[:limit] if limit else cl
    return Selection(arm, sel_layers, todo, len(missing))


def _iter_work(sels: Sequence[Selection]) -> Iterator[tuple[Selection, int, Cell]]:
    """Round-robin over arms so every arm progresses together."""
    iters = [iter(enumerate(s.cells)) for s in sels]
    live = list(range(len(sels)))
    while live:
        nxt = []
        for i in live:
            try:
                j, c = next(iters[i])
            except StopIteration:
                continue
            nxt.append(i)
            yield sels[i], j, c
        live = nxt


async def judge_cells(
    sels: Sequence[Selection],
    llm: LLM,
    models: Models,
    *,
    classify: bool = True,
    show_prompts: int = 0,
    workers: int = 128,
    log_every: float = 60.0,
    stages: frozenset[str] | None = None,
) -> None:
    total = sum(len(s.cells) for s in sels)
    done = 0
    t0 = time.monotonic()
    last = t0
    work = _iter_work(sels)

    def spend_line() -> str:
        tot = Usage()
        for s in sels:
            for u in s.arm.spend.values():
                tot.add(u)
        return f"calls={tot.calls} errors={tot.errors} spend=${tot.usd:.2f}"

    async def worker() -> None:
        nonlocal done, last
        for sel, j, cell in work:
            await process_cell(
                sel.arm,
                cell,
                llm,
                models,
                classify=classify,
                dump=j < show_prompts,
                stages=stages,
            )
            done += 1
            now = time.monotonic()
            if now - last >= log_every or done == total:
                last = now
                rate = done / max(now - t0, 1e-9)
                eta = (total - done) / rate if rate else float("inf")
                print(
                    f"  [{done}/{total} cells] {rate * 60:.0f} cells/min eta={eta / 60:.1f} min "
                    f"| {spend_line()}",
                    flush=True,
                )

    async with asyncio.TaskGroup() as tg:
        for _ in range(max(1, min(workers, total))):
            tg.create_task(worker())


async def run_arms_async(
    arms: Sequence[tuple[str, Path]],
    out_root: Path,
    models: Models,
    *,
    layers: Sequence[int] | None,
    limit: int | None,
    rpm: float,
    concurrency: int,
    allow_missing: bool,
    classify: bool = True,
    show_prompts: int = 0,
    llm: LLM | None = None,
    stages: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    run_stages = expand_stages(stages, classify=classify, anchor=models.map_anchor)
    opened = [open_arm(n, p, out_root, models) for n, p in arms]
    try:
        sels = []
        for a in opened:
            s = select(a, layers, limit)
            print(
                f"[{a.name}] kind={a.kind} layers={s.layers} cells_to_judge={len(s.cells)} "
                f"missing={s.n_missing} empty={a.report.n_empty} skipped={a.report.skipped}",
                flush=True,
            )
            if s.n_missing and not allow_missing:
                raise SystemExit(f"[{a.name}] {s.n_missing} missing cells (use --allow-missing)")
            if not s.layers:
                print(f"[{a.name}] no selected layer in this file: skipped")
                continue
            sels.append(s)
            _write_arm_json(a, models)
            if show_prompts:
                a.dump = (a.dir / "prompts_sample.txt").open("w", encoding="utf-8")
        own = llm is None
        llm = llm or LLM(rpm=rpm, concurrency=concurrency, effort=models.effort)
        try:
            needed = {models.judge}
            if any(s.arm.kind == "tokens" for s in sels):
                needed.add(models.summ)
            if "classify" in run_stages:
                needed.add(models.clf)
            if "map" in run_stages:
                needed.add(models.mapper)
            for m in sorted(needed):
                await llm.preflight(m)
                print(f"preflight ok: {m} (structured output, effort={models.effort})", flush=True)
            await judge_cells(
                sels,
                llm,
                models,
                classify=classify,
                show_prompts=show_prompts,
                workers=2 * concurrency,
                stages=run_stages,
            )
        finally:
            if own:
                await llm.aclose()
        for s in sels:
            print(f"[{s.arm.name}] this run:")
            for st in STAGES:
                print(f"    {st:9s} {s.arm.spend[st].report()}")
    finally:
        for a in opened:
            if a.dump is not None:
                a.dump.close()
            a.cache.close()
    summaries = {}
    for n, p in arms:
        summaries[n] = await rebuild_arm_async(n, p, out_root, models)
    return summaries


def run_arms(*args: Any, **kwargs: Any) -> dict[str, dict[str, Any]]:
    try:
        return asyncio.run(run_arms_async(*args, **kwargs))
    except* JudgeConfigError as eg:
        raise eg.exceptions[0] from None


def _write_arm_json(arm: Arm, models: Models) -> None:
    (arm.dir / "arm.json").write_text(
        json.dumps(
            {"arm": arm.name, "readouts": str(arm.readouts), "kind": arm.kind, **models.to_dict()},
            indent=1,
        )
        + "\n"
    )


# ---------------------------------------------------------------- rebuild (no API calls)


def rebuild_arm(name: str, readouts: Path, out_root: Path, models: Models) -> dict[str, Any]:
    return asyncio.run(rebuild_arm_async(name, readouts, out_root, models))


async def rebuild_arm_async(
    name: str, readouts: Path, out_root: Path, models: Models
) -> dict[str, Any]:
    """Regenerate ``cells.jsonl`` and ``summary.json`` from the cache over every judged layer
    (layers with any cache row). Streams records to disk; keeps slim score records only."""
    from . import score, score_v2

    arm = open_arm(name, readouts, out_root, models, readonly=True)
    layers = sorted(cached_layers(arm) & set(arm.file_layers))
    items = list(arm.items.values())
    expected = bank.expected_grid(items, layers)
    cells = [c for c in arm.cells if c.layer in layers]
    missing = missing_cells(cells, expected)
    acc = score.Accumulator(k=arm.k)
    acc2 = score_v2.AccumulatorV2()
    map_layers = {
        int(m.group(1)) for k in arm.cache.keys() if k.endswith(":M") and (m := _LAYER_RE.search(k))
    }
    tmp = arm.dir / f"cells.jsonl.{os.getpid()}.tmp"
    arm.dir.mkdir(parents=True, exist_ok=True)

    with tmp.open("w", encoding="utf-8") as fh:
        for c in cells:
            rec = await process_cell(arm, c, None, models)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            acc.add(rec)
            if c.layer in map_layers:
                acc2.add(rec)
    tmp.replace(arm.dir / "cells.jsonl")
    spend = {s: Usage() for s in STAGES}
    for row in arm.cache.rows():
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        spend[stage_of_key(row["key"])].add(meta.get("usage"))
    arm.cache.close()
    config = {
        "eval": NAME,
        "arm": name,
        "kind": arm.kind,
        "k": arm.k,
        "models": models.to_dict(),
        "prompt_versions": {
            "summary": summarizer.SUMMARIZER_PROMPT_VERSION if arm.kind == "tokens" else None,
            "span": v5c.PROMPT_VERSION,
            "verify": v5c.VERIFY_PROMPT_VERSION,
            "classify": tv1.PROMPT_VERSION,
            "map": cm.PROMPT_VERSION,
        },
        "bank_source": bank.BANK_SOURCE,
        "bank_commit": WSBENCH_PUB_COMMIT,
        "readouts": str(arm.readouts),
        "readouts_sha256": sha256_file(arm.readouts),
        "judged_layers": layers,
    }
    from hallucination_bench.calibrate_v2 import gate_passed

    config_v2 = {
        **config,
        "map_anchor": models.map_anchor,
        "map_layers": sorted(map_layers),
        "pre_gate": not gate_passed(out_root, models.mapper, models.map_anchor),
    }
    summary = score.summarize(
        acc,
        config=config,
        n_missing=len(missing),
        n_expected=len(expected),
        skipped_rows=arm.report.skipped,
        spend={k: v.to_dict() for k, v in spend.items()},
    )
    (arm.dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    p2 = arm.dir / "summary_v2.json"
    if any(c.map_status in ("ok", "failed") for c in acc2.cells):
        s2 = score_v2.summarize(acc2, config=config_v2, spend=spend["map"].to_dict())
        p2.write_text(json.dumps(s2, indent=1) + "\n")
    elif p2.exists():
        p2.unlink()
    return summary
