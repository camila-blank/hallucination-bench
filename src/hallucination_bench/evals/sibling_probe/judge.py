"""Sibling-probe judge (plan 0003 §Phase D): prefilter, coverage sample, claim-map judging,
mention -> segment attribution, scope and coverage calls.

Nothing here carries prompt text of the claim map or the summariser: the claim-map prompt,
schema, validation and parse are :mod:`hallucination_bench.judges.claimmap_v2`'s, the token-bag
summary is :mod:`hallucination_bench.summarizer`'s (``interp-v1``, cached ``summ:``), and every
call goes through :func:`hallucination_bench.evals.wsbench_chat.run.cached_call` (the same cache
rows, fingerprints and failure handling). The only new prompts are the two cheap structured
calls below (``sibscope-v1``, ``sibcover-v1``).

Per cell (one ``(id, layer, pos)`` of one arm):

* **prefilter**: prose arms, any alias hit of a concept of the item's category in the readout
  text; J-lens, a raw top-10 token equal to a token alias (no summary needed);
* **coverage sample**: a seeded 5% of the non-prefiltered cells per (arm, category) is judged
  too (inventory coverage);
* **claim map** (judged cells): prompt = the item's user turn, marked response = ``READ_MARK``
  at character 0 + the item's response (every read position is in the prompt), units = the
  production split of the readouts (prose) or of the summary (J-lens, ``kind="tokens"``);
* **J-lens summaries** are made for the judged cells and, by default, for every J-lens cell
  (``summarize_all``): the unmentioned rule needs "no summary alias hit" (plan rev 9); a J-lens
  cell without a summary has no unmentioned concepts (score.py);
* **mentions**: every alias hit (readout text, or J-lens summary) is attributed to the segment
  that contains it (largest overlap) and carries its leaf and verdict
  (``taxonomy_v2.VERDICT_OF``); a hit in an unjudged cell has verdict ``None``; a hit that no
  segment overlaps, or in a cell whose map failed, has verdict ``failed``;
* **scope** (``sibscope-v1``): one call per segment holding a sibling hallucination-candidate
  mention; **coverage** (``sibcover-v1``): one call per probe-candidate ``probe_target`` that
  matches no alias of the category ("is X an instance of <category>?"), cached per target.

Outputs: ``outputs/sibling/judge/<arm>/{cache.jsonl, cells.jsonl, run.json}``.
"""

import asyncio
import hashlib
import json
import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hallucination_bench import REPO_ROOT, summarizer
from hallucination_bench import taxonomy_v2 as tx
from hallucination_bench.cache import Cache
from hallucination_bench.evals.sibling_probe import bank as B
from hallucination_bench.evals.sibling_probe.concepts import Matcher, matcher
from hallucination_bench.evals.sibling_probe.probe_io import git_sha
from hallucination_bench.evals.wsbench_chat.run import HAL_K, cached_call
from hallucination_bench.judges import claimmap_v2 as cm
from hallucination_bench.judges.v5c import READ_MARK
from hallucination_bench.llm import LLM, Usage, schema_block
from hallucination_bench.readouts import Cell, decode_token, load_readouts
from hallucination_bench.units import split_cell

SIB = REPO_ROOT / "outputs" / "sibling"
READOUTS_ROOT = SIB / "readouts"
JUDGE_ROOT = SIB / "judge"
META_DIR = SIB / "render_eval" / "meta"
ARMS: tuple[str, ...] = ("s3d-rl600", "nla-rl-L42", "jlens")
LAYERS11: tuple[int, ...] = tuple(range(20, 61, 4))  # plan 0004: every lens layer
ARM_LAYERS: dict[str, tuple[int, ...]] = {
    "s3d-rl600": (28, 36, 44, 52),
    "jlens": (28, 36, 44, 52),
    "nla-rl-L42": (42,),
    # plan 0004 set A at all 11 layers (caches seeded from the 4-layer arms: same cell keys)
    "s3d-rl600-all11": LAYERS11,
    "jlens-all11": LAYERS11,
    "mlnla-rl600-all11": LAYERS11,
}
TOP_K = 10
COVERAGE_FRACTION = 0.05
COVERAGE_SEED = 0
M_VERDICTS: frozenset[str] = frozenset({tx.V_HALLUCINATION, tx.V_PROBE, tx.V_UNRESOLVED, tx.V_JUNK})

# ------------------------------------------------------------------ the two new cheap calls

SCOPE_VERSION = "sibscope-v1"
SCOPE_SYSTEM = (
    "You classify one segment of an interpretability-lens readout. The lens read a language "
    "model's internal activation while the model was reading a user's message, and the readout "
    "describes what the model was representing. Decide what the segment is about:\n"
    '- "user": it characterises the user or the user\'s own situation (who they are, where they '
    "are from, what they do, feel, keep, visit or celebrate, ...), correctly or not;\n"
    '- "other": it describes something else: a reconstructed or imagined prompt, a hypothetical '
    "scenario, a story, or a list of examples or options that is not a claim about this user.\n"
    "Output only the JSON verdict."
)
SCOPE_USER = (
    "The user's message:\n<message>\n{message}\n</message>\n\n"
    "The whole readout (context):\n<readout>\n{readout}\n</readout>\n\n"
    "The segment to classify:\n<segment>\n{segment}\n</segment>"
)
SCOPE_SCHEMA = schema_block(
    "sibscope",
    {"scope": {"type": "string", "enum": ["user", "other"]}, "why": {"type": "string"}},
    ["scope", "why"],
)

COVER_VERSION = "sibcover-v1"
COVER_SYSTEM = "You answer one yes/no question about a category. Output only the JSON verdict."
COVER_USER = (
    "Category: {category}.\n"
    'Is "{target}" a specific instance of this category (not a generic word, a broader group, '
    "or a different kind of thing)?"
)
COVER_SCHEMA = schema_block("sibcover", {"instance": {"type": "boolean"}}, ["instance"])
CATEGORY_NOUN: dict[str, str] = {
    "country": "a country",
    "occupation": "an occupation (a job or profession)",
    "emotion": "an emotion (a feeling)",
    "farm_animal": "a farm animal (a kind of livestock or poultry)",
    "capital_city": "a capital city of a country",
    "tree": "a kind of tree",
    "holiday": "a holiday or festival",
}
PROMPTS: dict[str, str] = {
    "SCOPE_SYSTEM": SCOPE_SYSTEM,
    "SCOPE_USER": SCOPE_USER,
    "COVER_SYSTEM": COVER_SYSTEM,
    "COVER_USER": COVER_USER,
}


def render_scope(message: str, readout: str, segment: str) -> tuple[str, str]:
    vals = {"message": message, "readout": readout, "segment": segment}
    user = SCOPE_USER
    for k in ("message", "readout", "segment"):  # one pass per placeholder, values last
        user = user.replace("{" + k + "}", "\0" + k + "\0")
    for k, v in vals.items():
        user = user.replace("\0" + k + "\0", v)
    return SCOPE_SYSTEM, user


def render_cover(category: str, target: str) -> tuple[str, str]:
    noun = CATEGORY_NOUN.get(category, category.replace("_", " "))
    return COVER_SYSTEM, COVER_USER.replace("{category}", noun).replace("{target}", target)


def norm_target(t: str) -> str:
    return " ".join(t.lower().split())


# ------------------------------------------------------------------ bank / readouts


def load_meta(meta_dir: Path = META_DIR) -> dict[str, dict[str, Any]]:
    """``{id: {n, positions_reported, ...}}`` from the ``prompts`` render meta."""
    meta: dict[str, dict[str, Any]] = {}
    for p in sorted(Path(meta_dir).glob("*.json")):
        meta.update(json.loads(p.read_text()))
    return meta


def item_prompt(item: Mapping[str, Any]) -> str:
    """The prompt shown to the judge: the item's user turn (as wsbench_chat's ``prompt``)."""
    return B.user_text(item["messages"])


def marked_response(item: Mapping[str, Any]) -> str:
    """Every read position is in the prompt, so the marker sits at character 0 (unless the item
    carries its own ``marked_response``, e.g. plan 0004 set B's prefilled answer stem)."""
    if item.get("marked_response") is not None:
        return item["marked_response"]
    return READ_MARK + (item.get("response") or "")


def render_map_prompt(
    item: Mapping[str, Any], samples: Sequence[str], kind: str
) -> tuple[str, str, list]:
    """(system, user, units) of the claim-map call for one cell (claimmap_v2, ``anchor="none"``)."""
    units, _ = split_cell(list(samples))
    units = cm.plain_units(units)
    system, user = cm.render_prompt(
        item_prompt(item), marked_response(item), units, kind=kind, anchor="none"
    )
    return system, user, units


def readout_files(arm: str, root: Path = READOUTS_ROOT) -> list[Path]:
    return sorted((Path(root) / arm).glob("*.jsonl"))


def load_arm_cells(
    arm: str, meta: Mapping[str, Mapping[str, Any]], root: Path = READOUTS_ROOT
) -> tuple[list[Cell], str]:
    """Every readout cell of the arm at its layers, restricted to the meta's ids / positions."""
    pos = {k: [int(p) for p in v["positions_reported"]] for k, v in meta.items()}
    cells: list[Cell] = []
    kind = None
    for f in readout_files(arm, root):
        cs, rep = load_readouts(f, positions=pos, layers=ARM_LAYERS.get(arm))
        cells += cs
        kind = kind or rep.kind
    seen: set[tuple[str, int, int]] = set()
    uniq = []
    for c in cells:
        k = (c.id, c.layer, c.pos)
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    return uniq, kind or "prose"


def coverage_sample(
    keys_by_cat: Mapping[str, Sequence[str]],
    *,
    fraction: float = COVERAGE_FRACTION,
    seed: int = COVERAGE_SEED,
    arm: str = "",
) -> set[str]:
    """Seeded ``fraction`` of the given (non-prefiltered) cell keys per category:
    ``ceil(fraction * n)`` keys with the smallest ``sha256(seed:arm:key)``."""
    out: set[str] = set()
    for _cat, keys in keys_by_cat.items():
        n = math.ceil(fraction * len(keys))
        ranked = sorted(
            keys, key=lambda k: hashlib.sha256(f"sibcov:{seed}:{arm}:{k}".encode()).hexdigest()
        )
        out.update(ranked[:n])
    return out


# ------------------------------------------------------------------ attribution


def attribute(
    readout: int, start: int, end: int, segments: Sequence[Mapping[str, Any]]
) -> int | None:
    """Index of the segment of ``readout`` with the largest character overlap with
    ``[start, end)``; None if no segment overlaps."""
    best, best_ov = None, 0
    for i, s in enumerate(segments):
        if int(s["readout"]) != readout:
            continue
        ov = min(end, int(s["end"])) - max(start, int(s["start"]))
        if ov > best_ov:
            best, best_ov = i, ov
    return best


def token_hits(mt: Matcher, tokens: Sequence[str], k: int = TOP_K) -> dict[str, list[str]]:
    """Concept -> raw top-k tokens that equal one of its token aliases."""
    out: dict[str, list[str]] = {}
    for t in list(tokens)[:k]:
        cid = mt.token_concept(t, raw=True)
        if cid is not None:
            out.setdefault(cid, []).append(t)
    return out


def text_hits(mt: Matcher, texts: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {"concept": h.concept, "alias": h.alias, "readout": i, "start": h.start, "end": h.end}
        for i, t in enumerate(texts)
        for h in mt.hits(t)
    ]


def is_sibling(cid: str, item: Mapping[str, Any]) -> bool:
    return cid != item["gold"] and cid not in set(item.get("named") or ())


# ------------------------------------------------------------------ per-arm context


@dataclass
class JudgeArm:
    """Duck-types ``wsbench_chat.run.Arm`` for :func:`cached_call` (cache, spend, dump)."""

    name: str
    dir: Path
    cache: Cache
    kind: str = "prose"
    spend: dict[str, Usage] = field(default_factory=lambda: defaultdict(Usage))
    dump: Any = None


@dataclass
class JudgeModels:
    judge: str = "claude-opus-5-5"
    effort: str | None = "low"


async def process_cell(
    ja: JudgeArm,
    cell: Cell,
    item: Mapping[str, Any],
    meta: Mapping[str, Any],
    llm: LLM | None,
    models: JudgeModels,
    *,
    in_sample: bool,
    summarize_all: bool,
    mt: Matcher | None = None,
    judge_all: bool = False,
    extras: bool = True,
) -> dict[str, Any]:
    """One cell -> its ``cells.jsonl`` record (judge mode with ``llm``, cache-only without).

    Plan 0004 set B: ``mt`` is the set-B category matcher, ``judge_all`` judges every non-empty
    cell (no prefilter), ``extras=False`` skips the scope / coverage calls."""
    cat = item["category"]
    mt = mt or matcher(cat)
    response_named = sorted(mt.mentions(item.get("response") or ""))
    rec: dict[str, Any] = {
        "key": cell.key,
        "id": cell.id,
        "layer": cell.layer,
        "pos": cell.pos,
        "offset": int(meta["n"]) - 1 - cell.pos,
        "category": cat,
        "gold": item["gold"],
        "named": list(item.get("named") or ()),
        "response_named": response_named,
        "kind": ja.kind,
        "empty": cell.empty,
        "prefiltered": False,
        "coverage_sample": False,
        "judged": False,
        "status": "unjudged",
        "token_hits": {},
        "summary": None,
        "summary_status": None,
        "mentions": [],
        "segments": [],
    }
    if cell.empty:
        rec["status"] = "empty"
        return rec
    if ja.kind == "tokens":
        th = token_hits(mt, cell.tokens or ())
        rec["token_hits"] = th
        rec["prefiltered"] = bool(th)
    else:
        samples = [s for s in cell.samples or () if s.strip()][:HAL_K]
        rec["prefiltered"] = bool(text_hits(mt, samples))
    rec["coverage_sample"] = (not rec["prefiltered"]) and in_sample
    to_judge = rec["prefiltered"] or rec["coverage_sample"] or judge_all
    # ---- J-lens: summary
    if ja.kind == "tokens":
        if not (to_judge or summarize_all):
            return rec
        decoded = [decode_token(t) for t in cell.tokens or ()]
        bag = summarizer.render_bag(decoded, cell.scores)
        out = await cached_call(
            ja,
            llm,
            stage="summary",
            key=f"summ:{cell.key}",
            version=summarizer.SUMMARIZER_PROMPT_VERSION,
            model=models.judge,
            effort=models.effort,
            system=summarizer.INTERP_SYSTEM,
            user=summarizer.render_user(bag),
            schema=summarizer.INTERP_SCHEMA,
            validate=lambda r: summarizer.interpretation_of(r) is not None,
        )
        rec["summary_status"] = out.status
        text = summarizer.interpretation_of(out.result) if out.status == "ok" else None
        if text is None:
            rec["status"] = "failed_summary" if out.status == "failed" else "pending_summary"
            return rec
        rec["summary"] = text
        samples = [text]
    hits = text_hits(mt, samples)
    for h in hits:
        h["source"] = "summary" if ja.kind == "tokens" else "readout"
        h["verdict"] = None
        h["leaf"] = None
        h["segment"] = None
    rec["mentions"] = hits
    if not to_judge:
        return rec
    # ---- claim map
    rec["judged"] = True
    system, user, units = render_map_prompt(item, samples, ja.kind)
    mout = await cached_call(
        ja,
        llm,
        stage="map",
        key=f"{cell.key}:M",
        version=cm.PROMPT_VERSION,
        model=models.judge,
        effort=models.effort,
        system=system,
        user=user,
        schema=cm.schema_for("none"),
        validate=lambda r: cm.validate(r, units),
    )
    rec["status"] = {"ok": "judged", "failed": "failed_map"}.get(mout.status, "pending")
    raw = mout.result if mout.status == "ok" else None
    segs, _counts = cm.parse(raw, units)
    seg_rows = []
    for s in segs:
        targets = sorted(mt.mentions(s.probe_target)) if s.probe_target else []
        seg_rows.append(
            {
                "readout": s.readout,
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "leaf": s.leaf,
                "verdict": tx.VERDICT_OF.get(s.leaf, tx.FAILED),
                "probe_target": s.probe_target,
                "target_concepts": targets,
                "scope": None,
                "cover": None,
            }
        )
    rec["segments"] = seg_rows
    for h in hits:
        if rec["status"] != "judged":
            h["verdict"], h["leaf"] = tx.FAILED, tx.FAILED
            continue
        i = attribute(h["readout"], h["start"], h["end"], seg_rows)
        if i is None:
            h["verdict"], h["leaf"] = tx.FAILED, tx.FAILED
            continue
        h["segment"] = i
        h["leaf"] = seg_rows[i]["leaf"]
        h["verdict"] = seg_rows[i]["verdict"]
    # ---- scope call per segment holding a sibling hallucination-candidate mention
    if not extras:
        return rec
    scope_segs = sorted(
        {
            h["segment"]
            for h in hits
            if h["segment"] is not None
            and h["verdict"] in M_VERDICTS
            and is_sibling(h["concept"], item)
            and (ja.kind != "tokens" or h["concept"] in rec["token_hits"])
        }
    )
    message = item_prompt(item)
    for i in scope_segs:
        s = seg_rows[i]
        system, user = render_scope(message, samples[s["readout"]], s["text"])
        out = await cached_call(
            ja,
            llm,
            stage="scope",
            key=f"{cell.key}:S:{s['readout']}:{s['start']}-{s['end']}",
            version=SCOPE_VERSION,
            model=models.judge,
            effort=models.effort,
            system=system,
            user=user,
            schema=SCOPE_SCHEMA,
            validate=lambda r: r.get("scope") in ("user", "other"),
        )
        s["scope"] = out.result.get("scope") if out.status == "ok" and out.result else None
    for h in hits:
        if h["segment"] is not None:
            h["scope"] = seg_rows[h["segment"]]["scope"]
    # ---- coverage call per out-of-inventory probe_target
    for s in seg_rows:
        if s["verdict"] != tx.V_PROBE or not s["probe_target"] or s["target_concepts"]:
            continue
        system, user = render_cover(cat, s["probe_target"])
        out = await cached_call(
            ja,
            llm,
            stage="cover",
            key=f"cover:{cat}:{norm_target(s['probe_target'])}",
            version=COVER_VERSION,
            model=models.judge,
            effort=models.effort,
            system=system,
            user=user,
            schema=COVER_SCHEMA,
            validate=lambda r: isinstance(r.get("instance"), bool),
        )
        s["cover"] = out.result.get("instance") if out.status == "ok" and out.result else None
    return rec


# ------------------------------------------------------------------ one arm


async def judge_arm(
    arm: str,
    llm: LLM | None,
    models: JudgeModels,
    *,
    readouts_root: Path = READOUTS_ROOT,
    out_root: Path = JUDGE_ROOT,
    meta_dir: Path = META_DIR,
    items: Sequence[Mapping[str, Any]] | None = None,
    limit: int | None = None,
    summarize_all: bool = True,
    workers: int = 128,
    log_every: float = 60.0,
    extras: bool = True,
) -> dict[str, Any]:
    """Judge (``llm``) or rebuild (``llm=None``) one arm; always rewrites ``cells.jsonl`` and
    ``run.json`` from the cache."""
    items_by_id = {it["id"]: it for it in (items if items is not None else B.load_items())}
    meta = {k: v for k, v in load_meta(meta_dir).items() if k in items_by_id}
    cells, kind = load_arm_cells(arm, meta, readouts_root)
    cells = sorted(cells, key=lambda c: (c.id, c.layer, c.pos))
    if limit:
        keep_ids = sorted({c.id for c in cells})[:limit]
        cells = [c for c in cells if c.id in set(keep_ids)]
    d = Path(out_root) / arm
    d.mkdir(parents=True, exist_ok=True)
    ja = JudgeArm(arm, d, Cache(d / "cache.jsonl", readonly=False), kind)
    # coverage sample over the non-prefiltered cells, per category
    nonpre: dict[str, list[str]] = defaultdict(list)
    for c in cells:
        if c.empty:
            continue
        it = items_by_id[c.id]
        mt = matcher(it["category"])
        if kind == "tokens":
            pre = bool(token_hits(mt, c.tokens or ()))
        else:
            pre = bool(text_hits(mt, [s for s in c.samples or () if s.strip()][:HAL_K]))
        if not pre:
            nonpre[it["category"]].append(c.key)
    sample = coverage_sample(nonpre, arm=arm)
    recs: list[dict[str, Any] | None] = [None] * len(cells)
    t0 = last = time.monotonic()
    done = 0
    queue = list(enumerate(cells))

    async def worker() -> None:
        nonlocal done, last
        while queue:
            j, c = queue.pop()
            recs[j] = await process_cell(
                ja,
                c,
                items_by_id[c.id],
                meta[c.id],
                llm,
                models,
                in_sample=c.key in sample,
                summarize_all=summarize_all,
                extras=extras,
            )
            done += 1
            now = time.monotonic()
            if llm is not None and (now - last >= log_every or done == len(cells)):
                last = now
                tot = Usage()
                for u in ja.spend.values():
                    tot.add(u)
                print(
                    f"  [{arm} {done}/{len(cells)}] calls={tot.calls} errors={tot.errors} "
                    f"spend=${tot.usd:.2f} ({(now - t0) / 60:.1f} min)",
                    flush=True,
                )

    try:
        n_workers = max(1, min(workers if llm is not None else 1, len(cells)))
        await asyncio.gather(*(worker() for _ in range(n_workers)))
    finally:
        ja.cache.close()
    with (d / "cells.jsonl").open("w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts: dict[str, int] = defaultdict(int)
    for r in recs:
        assert r is not None
        counts[f"status:{r['status']}"] += 1
        counts["prefiltered"] += r["prefiltered"]
        counts["coverage_sample"] += r["coverage_sample"]
        counts["judged"] += r["judged"]
    spend = {k: v.to_dict() for k, v in ja.spend.items()}
    run = {
        "arm": arm,
        "kind": kind,
        "n_cells": len(cells),
        "counts": dict(counts),
        "claimmap_prompt_version": cm.PROMPT_VERSION,
        "taxonomy_version": tx.TAXONOMY_VERSION,
        "summarizer_prompt_version": summarizer.SUMMARIZER_PROMPT_VERSION,
        "scope_prompt_version": SCOPE_VERSION,
        "cover_prompt_version": COVER_VERSION,
        "anchor": "none",
        "model": models.judge,
        "effort": models.effort,
        "coverage_fraction": COVERAGE_FRACTION,
        "coverage_seed": COVERAGE_SEED,
        "summarize_all": summarize_all,
        "extras": extras,
        "git_sha": git_sha(),
        "spend_this_run": spend,
        "spend_this_run_usd": round(sum(v.usd for v in ja.spend.values()), 4),
    }
    (d / "run.json").write_text(json.dumps(run, indent=1) + "\n")
    return run


async def judge_arms(
    arms: Sequence[str],
    *,
    rebuild: bool = False,
    rpm: float = 1500.0,
    concurrency: int = 64,
    models: JudgeModels | None = None,
    **kw: Any,
) -> dict[str, dict[str, Any]]:
    models = models or JudgeModels()
    llm = None if rebuild else LLM(rpm=rpm, concurrency=concurrency, effort=models.effort)
    try:
        if llm is not None:
            await llm.preflight(models.judge)
        runs = await asyncio.gather(
            *(judge_arm(a, llm, models, workers=2 * concurrency, **kw) for a in arms)
        )
        out = dict(zip(arms, runs, strict=True))
        for a, r in out.items():
            print(f"[{a}] {json.dumps(r['counts'])} spend=${r['spend_this_run_usd']}")
        return out
    finally:
        if llm is not None:
            await llm.aclose()


def load_cells(arm: str, root: Path = JUDGE_ROOT) -> list[dict[str, Any]]:
    path = Path(root) / arm / "cells.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
