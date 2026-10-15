"""Plan 0009 §2b-c: the slot-fill judge ``sibfill-v1`` and the pre-registered rung-1 rule.

One call per unit (``claude-opus-5-5``, effort low, no temperature, json_schema output), cached in
``outputs/sibling_gate/fill/cache.jsonl`` under ``fill:{arm}:{cellkey}:{said}:{actual}`` with the
standard fingerprint (prompt version, model, effort, system, user). An invalid answer is re-asked
up to 2 times; after that the unit is ``resample_failed`` (not a hallucination).

Per block the judge answers what fills **the same slot** as the said object in the original
claim: ``said`` (aliases count), ``actual`` (the item's right answer, any form), ``other`` (a
different object, named) or ``none``. A multi-line block gets one answer, by precedence
said > other (every distinct other object listed) > actual > none. Every non-``none`` answer
quotes the block verbatim; a quote that fails ``readouts.verify_quote`` turns the block into
``none`` (counted).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from hallucination_bench.cache import Cache, fingerprint
from hallucination_bench.llm import LLM, Usage, schema_block
from hallucination_bench.readouts import verify_quote

from . import units as U

PROMPT_VERSION = "sibfill-v1"
MODEL = "claude-opus-5-5"
EFFORT = "low"
INVALID_RETRIES = 2
WINDOW = 400
ANSWERS = ("said", "actual", "other", "none")
CACHE_PATH = U.OUT / "fill" / "cache.jsonl"


# ---------------------------------------------------------------- a cached structured call


async def cached_json_call(
    cache: Cache,
    llm: LLM | None,
    *,
    key: str,
    version: str,
    model: str,
    effort: str | None,
    system: str,
    user: str,
    schema: dict[str, Any],
    validate: Callable[[dict[str, Any]], bool] | None = None,
    invalid_retries: int = INVALID_RETRIES,
    spend: Usage | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """The ``run.cached_call`` contract without an ``Arm``: ``("ok", result)``, ``("failed",
    raw|None)`` or ``("pending", None)`` (cache only, ``llm is None``). Failed answers are cached
    with ``result: None`` (+ ``meta.raw``) and re-asked on the next run."""
    fp = fingerprint(version, model, effort, system, user)
    row = cache.get(key, fp)
    if row is not None:
        return "ok", row["result"]
    if llm is None:
        failed = cache.get(key, fp, include_failed=True)
        if failed is None:
            return "pending", None
        return "failed", (failed.get("meta") or {}).get("raw")
    usage = Usage()
    n_invalid = 0
    while True:
        out = await llm.call(model=model, system=system, user=user, schema=schema)
        usage.add(out.usage)
        if spend is not None:
            spend.add(out.usage)
        r = out.result
        if r is None or validate is None or validate(r):
            break
        n_invalid += 1
        if n_invalid > invalid_retries:
            meta = {"model": model, "usage": usage.to_dict(), "n_invalid_attempts": n_invalid}
            cache.put(key, fp, {"result": None, "meta": {**meta, "raw": r}})
            return "failed", r
    meta: dict[str, Any] = {"model": model, "usage": usage.to_dict()}
    if n_invalid:
        meta["n_invalid_attempts"] = n_invalid
    cache.put(key, fp, {"result": r, "meta": meta})
    return ("ok", r) if r is not None else ("failed", None)


async def gather_limited(coros: Sequence[Any], limit: int = 256) -> list[Any]:
    sem = asyncio.Semaphore(limit)

    async def one(c: Any) -> Any:
        async with sem:
            return await c

    return await asyncio.gather(*(one(c) for c in coros))


# ---------------------------------------------------------------- prompt

SYSTEM = """You audit resampled readouts of an interpretability lens. A lens reads a language \
model's activation at one token of a chat and writes a short description of what the model is \
representing there. A claim-map judge already found that one original readout made a \
SUBSTITUTION: it named a sibling object (SAID) where the conversation gives a different, right \
object (ACTUAL). You are shown the original claim, the evidence, the conversation around that \
evidence, and N fresh readouts of the same lens at the same activation (numbered blocks).

For EACH block, decide what fills THE SAME SLOT: the role the said object plays in the original \
claim (for example "the company that built me", "the country the user is from").
- "said": the block fills the slot with the said object. Aliases and longer names count \
("OpenAI's ChatGPT" for ChatGPT).
- "actual": the block fills the slot with the right answer as the conversation gives it, in any \
form, including a description ("the team that created me").
- "other": the block fills the slot with any different object, whatever its kind. List it in \
"objects", normalised (a short canonical name).
- "none": the block does not fill the slot. A mention of the said object elsewhere in the block, \
in a different role, is "none".

A block is a multi-line "- " list and gets ONE answer, by precedence: "said" if any line fills \
the slot with the said object; else "other" if any line fills it with a different object (list \
every distinct other object); else "actual"; else "none".

Every answer except "none" carries "quote": a verbatim span copied from THAT block (at least 3 \
characters) showing the filler. For "none" give an empty quote and no objects. Answer every \
block exactly once, by its number. Return JSON only."""


def _schema() -> dict[str, Any]:
    item = {
        "type": "object",
        "additionalProperties": False,
        "required": ["block", "answer", "objects", "quote"],
        "properties": {
            "block": {"type": "integer"},
            "answer": {"type": "string", "enum": list(ANSWERS)},
            "objects": {"type": "array", "items": {"type": "string"}},
            "quote": {"type": "string"},
        },
    }
    return schema_block(
        "sibfill",
        {"blocks": {"type": "array", "items": item}, "note": {"type": "string"}},
        ["blocks", "note"],
    )


SCHEMA = _schema()


def window(conversation: str, quote: str | None, width: int = WINDOW) -> str:
    """±``width`` characters of the conversation around ``quote`` (case-insensitive find; the
    start of the conversation if the quote is not found)."""
    if not quote:
        return conversation[: 2 * width]
    i = conversation.find(quote)
    if i < 0:
        i = conversation.lower().find(quote.lower())
    if i < 0:
        return conversation[: 2 * width]
    return conversation[max(0, i - width) : i + len(quote) + width]


def render_user(unit: Mapping[str, Any], conversation: str, blocks: Sequence[str]) -> str:
    seg = unit["segments"][0]
    cb = seg.get("contradicted_by") or ""
    lines = [
        "ORIGINAL CLAIM (segment of one original readout): "
        + json.dumps(seg.get("text") or "", ensure_ascii=False),
        f"SAID (the sibling the readout named): {json.dumps(unit['said'], ensure_ascii=False)}",
        f"ACTUAL (the right object): {json.dumps(unit['actual'], ensure_ascii=False)}",
        f"EVIDENCE (contradicted_by, from the conversation): {json.dumps(cb, ensure_ascii=False)}",
        "",
        "CONVERSATION AROUND THE EVIDENCE:",
        window(conversation, cb),
        "",
        f"{len(blocks)} FRESH READOUT BLOCKS:",
    ]
    for i, b in enumerate(blocks, 1):
        lines += ["", f"=== block {i} ===", b]
    return "\n".join(lines)


# ---------------------------------------------------------------- validate / parse


def validate(raw: Any, n_blocks: int) -> bool:
    if not isinstance(raw, dict) or not isinstance(raw.get("blocks"), list):
        return False
    seen = set()
    for b in raw["blocks"]:
        if not isinstance(b, dict) or b.get("answer") not in ANSWERS:
            return False
        i = b.get("block")
        if not isinstance(i, int) or not 1 <= i <= n_blocks or i in seen:
            return False
        seen.add(i)
        objs = b.get("objects")
        if not isinstance(objs, list) or not all(isinstance(o, str) for o in objs):
            return False
        if b["answer"] == "other" and not [o for o in objs if o.strip()]:
            return False
        if b["answer"] != "none" and not isinstance(b.get("quote"), str):
            return False
    return len(seen) == n_blocks


def parse(raw: Mapping[str, Any], blocks: Sequence[str]) -> dict[str, Any]:
    """Validated answer -> per-block answers after the quote check, plus the failed-quote count."""
    by = {int(b["block"]): b for b in raw["blocks"]}
    answers, n_failed = [], 0
    for i, text in enumerate(blocks, 1):
        b = by[i]
        a = b["answer"]
        objs = [o.strip() for o in b.get("objects") or [] if o.strip()] if a == "other" else []
        if a != "none" and not verify_quote(b.get("quote") or "", [text]):
            n_failed += 1
            a, objs = "none", []
        answers.append({"block": i, "answer": a, "objects": objs})
    return {"answers": answers, "n_failed_quotes": n_failed}


# ---------------------------------------------------------------- rung-1 rule (pre-registered)


def counts(
    answers: Sequence[Mapping[str, Any]], said: str | None = None, actual: str | None = None
) -> dict[str, int]:
    """Rung-1 counts. ``other`` objects that normalise to the said / actual object (an alias the
    judge filed as other) never count as distinct others (``n_other_dropped``)."""
    same = {U.norm(x) for x in (said, actual) if x}
    raw = {U.norm(o) for a in answers if a["answer"] == "other" for o in a["objects"]}
    others = raw - same
    return {
        "n_said": sum(a["answer"] == "said" for a in answers),
        "n_actual": sum(a["answer"] == "actual" for a in answers),
        "n_other": len(others),  # distinct other objects over the blocks
        "n_other_blocks": sum(a["answer"] == "other" for a in answers),
        "n_none": sum(a["answer"] == "none" for a in answers),
        "n_other_dropped": len(raw & same),
    }


def _cond(c: Mapping[str, int]) -> tuple[bool, bool]:
    return c["n_other"] >= 2, c["n_actual"] >= 4


def rung1(c: Mapping[str, int]) -> dict[str, Any]:
    """§2c: unstable iff n_said <= 2 and (n_other >= 2 [branch A] or n_actual >= 4 [branch B]);
    stable iff n_said >= 3; else sparse."""
    a, b = _cond(c)
    if c["n_said"] <= 2 and (a or b):
        status = "unstable"
    elif c["n_said"] >= 3:
        status = "stable"
    else:
        status = "sparse"
    branch = ("both" if a and b else "A" if a else "B") if status == "unstable" else None
    return {"status": status, "branch": branch, "diverse": a}


def sensitivity_unstable(c: Mapping[str, int]) -> dict[str, bool]:
    """Each sensitivity row's rung-1 decision (True = unstable; False = goes to rung 2)."""
    a, b = _cond(c)
    s = c["n_said"]
    return {
        # (i) rev-1: n_said <= 2 and n_said + n_actual + #other blocks >= 4
        "rung1_rev1": s <= 2 and s + c["n_actual"] + c["n_other_blocks"] >= 4,
        "rung1_branch_a_only": s <= 2 and a,  # (ii)
        "rung1_said_le1": s <= 1 and (a or b),  # (iii)
        "rung1_said_le3": s <= 3 and (a or b),  # (iv) stable at >= 4
    }


def needs_rung2(r1: Mapping[str, Any]) -> bool:
    """Rung 2 runs on stable / sparse / deterministic units, plus the sensitivity-only units
    (2 <= n_said <= 3, and branch-B-only unstable units)."""
    st = r1.get("rung1")
    if st in ("stable", "sparse", "deterministic"):
        return True
    c = r1.get("counts") or {}
    if st == "unstable" and (2 <= c.get("n_said", -1) <= 3 or r1.get("branch") == "B"):
        return True
    return False


def rung1_row(unit: Mapping[str, Any], status: str, parsed: Mapping[str, Any] | None):
    """The rung-1 record of one unit (``status``: ok | failed | pending | deterministic |
    resample_failed)."""
    base = {"unit": unit["unit"], "arm": unit["arm"]}
    if status in ("deterministic", "resample_failed", "pending"):
        r1 = None if status == "pending" else status
        return {**base, "rung1": r1, "counts": None, "branch": None, "diverse": False}
    if status != "ok" or parsed is None:
        return {
            **base,
            "rung1": "resample_failed",
            "counts": None,
            "branch": None,
            "diverse": False,
            "fill_failed": True,
        }
    c = counts(parsed["answers"], unit.get("said"), unit.get("actual"))
    r = rung1(c)
    return {
        **base,
        "rung1": r["status"],
        "counts": c,
        "branch": r["branch"],
        "diverse": r["diverse"],
        "sens_unstable": sensitivity_unstable(c),
        "n_failed_quotes": parsed["n_failed_quotes"],
        "answers": parsed["answers"],
    }


# ---------------------------------------------------------------- the stage


def conversation_for(item: Mapping[str, Any], pos: int) -> str:
    from hallucination_bench.evals.wsbench_chat import bank
    from hallucination_bench.judges.claimmap_v2 import conversation_text

    site = next(s for s in item["sites"] if int(s["pos"]) == int(pos))
    return conversation_text(item["prompt"], bank.marked_response(item, site))


async def run_fill(
    units: Sequence[Mapping[str, Any]],
    resampled: Mapping[str, Mapping[tuple[str, int, int], list[str]]],
    items: Mapping[str, Mapping[str, Any]],
    llm: LLM | None,
    cache: Cache,
    *,
    failed_pools: frozenset[str] = frozenset(),
    model: str = MODEL,
    effort: str | None = EFFORT,
) -> tuple[list[dict[str, Any]], Usage]:
    """Rung 1 for every unit. ``resampled``: pool -> (id, layer, pos) -> 10 blocks. Token arms
    are ``deterministic``; units of a failed pool, or with no resample row, ``resample_failed``.
    ``llm=None`` replays the cache only."""
    from . import resample as R

    spend = Usage()

    async def one(u: Mapping[str, Any]) -> dict[str, Any]:
        if u["arm"] in U.DETERMINISTIC_ARMS:
            return rung1_row(u, "deterministic", None)
        pool = R.pool_of(u["arm"])
        blocks = (resampled.get(pool) or {}).get((u["id"], int(u["layer"]), int(u["pos"])))
        if pool is None or pool in failed_pools or not blocks:
            return rung1_row(u, "resample_failed", None)
        user = render_user(u, conversation_for(items[u["id"]], u["pos"]), blocks)
        st, res = await cached_json_call(
            cache,
            llm,
            key=U.fill_cache_key(u),
            version=PROMPT_VERSION,
            model=model,
            effort=effort,
            system=SYSTEM,
            user=user,
            schema=SCHEMA,
            validate=lambda r: validate(r, len(blocks)),
            spend=spend,
        )
        if st == "pending":
            return rung1_row(u, "pending", None)
        parsed = parse(res, blocks) if st == "ok" and res is not None else None
        return rung1_row(u, st, parsed)

    rows = await gather_limited([one(u) for u in units])
    return rows, spend


def rung1_path(arm: str, out: Path = U.OUT) -> Path:
    return Path(out) / "fill" / "rung1" / f"{arm}.jsonl"


# ---------------------------------------------------------------- calibration (plan 0006 protocol)


def agreement(judge: Mapping[str, Any], gold: Mapping[str, Any]) -> dict[str, Any]:
    """Per-block fill agreement and unit-level rung-1 agreement between judge rung-1 rows and gold
    rows (``{unit, answers: [{block, answer, objects}]}``), on their shared units."""
    nb = ab = nu = au = 0
    for k, g in gold.items():
        j = judge.get(k)
        if j is None or not j.get("answers"):
            continue
        ja = {a["block"]: a["answer"] for a in j["answers"]}
        for a in g["answers"]:
            nb += 1
            ab += ja.get(a["block"]) == a["answer"]
        nu += 1
        au += rung1(counts(g["answers"], g.get("said"), g.get("actual")))["status"] == j["rung1"]
    return {
        "n_blocks": nb,
        "block_agreement": ab / nb if nb else None,
        "n_units": nu,
        "unit_agreement": au / nu if nu else None,
        "targets": {"block": 0.90, "unit": 0.85},
    }
