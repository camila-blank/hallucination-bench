"""wsbench_chat pipeline: cells -> (summarise) -> span -> verify -> classify -> cells.jsonl.

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
``{key}:V`` (verify), ``{key}:C`` (classify), with ``key = {id}__L{layer:03d}__p{pos}``.
Fingerprints = (prompt version, model, effort, system, user).
"""

import asyncio
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

from . import NAME, bank

STAGES: tuple[str, ...] = ("summary", "span", "verify", "classify")
HAL_K = v5c.HAL_K
_LAYER_RE = re.compile(r"__L(\d{3})__p")


@dataclass
class Models:
    judge: str = "claude-opus-5-5"
    summarizer: str | None = None
    classifier: str | None = None
    effort: str | None = "low"

    @property
    def summ(self) -> str:
        return self.summarizer or self.judge

    @property
    def clf(self) -> str:
        return self.classifier or self.judge

    def to_dict(self) -> dict[str, Any]:
        return {
            "judge": self.judge,
            "summarizer": self.summ,
            "classifier": self.clf,
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
) -> CallOut:
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
    out = await llm.call(model=model, system=system, user=user, schema=schema)
    arm.spend[stage].add(out.usage)
    meta: dict[str, Any] = {"stage": stage, "model": model, "usage": out.usage.to_dict()}
    r = out.result
    if r is not None and validate is not None and not validate(r):
        arm.cache.put(key, fp, {"result": None, "meta": {**meta, "raw": r}})
        return CallOut("failed", raw=r)
    arm.cache.put(key, fp, {"result": r, "meta": meta})
    return CallOut("ok" if r is not None else "failed", r)


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
) -> dict[str, Any]:
    """Take one cell through every stage (judge mode) or read it back (rebuild mode); return
    its merged ``cells.jsonl`` record."""
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
            llm,
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
    # ---- 2. span stage (v5c, verbatim)
    system, user = v5c.judge_prompt(item, site, samples)
    out = await cached_call(
        arm,
        llm,
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
        llm,
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
    # ---- 4. classify stage (taxonomy-v1): after verify tallied or failed for good
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
        llm,
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
            await process_cell(sel.arm, cell, llm, models, classify=classify, dump=j < show_prompts)
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
) -> dict[str, dict[str, Any]]:
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
            if classify:
                needed.add(models.clf)
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
    from . import score

    arm = open_arm(name, readouts, out_root, models, readonly=True)
    layers = sorted(cached_layers(arm) & set(arm.file_layers))
    items = list(arm.items.values())
    expected = bank.expected_grid(items, layers)
    cells = [c for c in arm.cells if c.layer in layers]
    missing = missing_cells(cells, expected)
    acc = score.Accumulator(k=arm.k)
    tmp = arm.dir / f"cells.jsonl.{os.getpid()}.tmp"
    arm.dir.mkdir(parents=True, exist_ok=True)

    with tmp.open("w", encoding="utf-8") as fh:
        for c in cells:
            rec = await process_cell(arm, c, None, models)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            acc.add(rec)
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
        },
        "bank_source": bank.BANK_SOURCE,
        "bank_commit": WSBENCH_PUB_COMMIT,
        "readouts": str(arm.readouts),
        "readouts_sha256": sha256_file(arm.readouts),
        "judged_layers": layers,
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
    return summary
