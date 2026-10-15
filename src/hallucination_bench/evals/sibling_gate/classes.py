"""Plan 0009 §3a: classes and concepts for rung 2.

* ``sibclass-v1`` (Opus 5.5, cached, one call per distinct (normalised said, item)): the said
  object's enumerable ``class`` (or ``no_class``: fewer than 9 well-known members),
  ``said_canonical``, the ``gold`` (the canonical same-class entity the item text makes right for
  the slot, or ``no_gold``) and ``compatible`` concepts (true together with the gold: parent
  company, alias, the gold's own product).
* **Class dedupe**: one call maps the full list of class names to canonical classes.
* **Concept set C per class**: every ``said_canonical`` and ``gold`` of the class, plus well-known
  fillers from one ``sibconcepts-v1`` call per class until C has at least 16 concepts. A filler
  is never a gold or said object of the class. The **compatible stratum** of a gold is its
  ``compatible`` list (never a distractor for that gold, §3d).

``said_canonical`` is the join key for cross-method and the distractor exclusion.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from hallucination_bench.cache import Cache
from hallucination_bench.llm import LLM, Usage, schema_block

from . import units as U
from .fill_judge import EFFORT, MODEL, cached_json_call, gather_limited, window

CLASS_VERSION = "sibclass-v1"
DEDUPE_VERSION = "sibclass-dedupe-v2"
CONCEPTS_VERSION = "sibconcepts-v1"
NO_CLASS = "no_class"
NO_GOLD = "no_gold"
MIN_CLASS_SIZE = 9  # §3d: a class with fewer concepts is n/a
MIN_CONCEPTS = 16
CACHE_PATH = U.OUT / "classes" / "cache.jsonl"

CLASS_SYSTEM = """You classify one object a lens readout named in place of the right one. You \
get the said object, the readout segment, and the conversation window that gives the right \
answer. Return:
- "class": a short plural class name for the said object ("AI assistants", "countries", \
"Chinese tech companies"), or "no_class" when the said object has no enumerable class with at \
least 9 well-known members;
- "said_canonical": the said object's canonical name (expand aliases: "OpenAI's ChatGPT" -> \
"ChatGPT");
- "gold": the canonical entity OF THE SAME CLASS that the conversation makes right for the slot \
the said object fills (e.g. "Alibaba" for "the engineers who created me" when the text says Qwen \
is by Alibaba), or "no_gold" when the conversation names no such entity;
- "compatible": concepts that are true together with the gold (its parent company, an alias, the \
gold's own product); empty if none.
Return JSON only."""

DEDUPE_SYSTEM = """You merge near-duplicate class names. You get a list of short class names \
(e.g. "AI assistants", "AI chatbots", "countries", "nations"). Names that denote the same set of \
objects share one canonical class name (pick the most standard input name); distinct sets stay \
distinct. Return ONLY the names that must be renamed, as {name, canonical} pairs with canonical \
different from name; a name that is already canonical is omitted. Return JSON only."""

CONCEPTS_SYSTEM = """You list well-known members of a class. Return well-known, unambiguous \
members of the class that are NOT in the exclusion list, by canonical name, most familiar first. \
Return JSON only."""

CLASS_SCHEMA = schema_block(
    "sibclass",
    {
        "class": {"type": "string"},
        "said_canonical": {"type": "string"},
        "gold": {"type": "string"},
        "compatible": {"type": "array", "items": {"type": "string"}},
    },
    ["class", "said_canonical", "gold", "compatible"],
)
DEDUPE_SCHEMA = schema_block(
    "sibclass_dedupe",
    {
        "mapping": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "canonical"],
                "properties": {"name": {"type": "string"}, "canonical": {"type": "string"}},
            },
        }
    },
    ["mapping"],
)
CONCEPTS_SCHEMA = schema_block(
    "sibconcepts", {"members": {"type": "array", "items": {"type": "string"}}}, ["members"]
)


def valid_class(r: Any) -> bool:
    return (
        isinstance(r, dict)
        and all(isinstance(r.get(k), str) and r[k].strip() for k in ("class", "said_canonical"))
        and isinstance(r.get("gold"), str)
        and isinstance(r.get("compatible"), list)
    )


def class_user(said: str, segment: str, conv_window: str) -> str:
    return "\n".join(
        [
            f"SAID OBJECT: {json.dumps(said, ensure_ascii=False)}",
            f"READOUT SEGMENT: {json.dumps(segment, ensure_ascii=False)}",
            "CONVERSATION WINDOW:",
            conv_window,
        ]
    )


def class_requests(units: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """One request per distinct (normalised said, item): the first unit's said / segment."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for u in units:
        k = (u["said_norm"], u["id"])
        if k not in out:
            seg = u["segments"][0]
            out[k] = {
                "said": u["said"],
                "segment": seg.get("text") or "",
                "contradicted_by": seg.get("contradicted_by"),
                "pos": u["pos"],
            }
    return out


async def run_classes(
    reqs: Mapping[tuple[str, str], Mapping[str, Any]],
    items: Mapping[str, Mapping[str, Any]],
    llm: LLM | None,
    cache: Cache,
) -> tuple[dict[tuple[str, str], dict[str, Any] | None], Usage]:
    from .fill_judge import conversation_for

    spend = Usage()

    async def one(k: tuple[str, str], r: Mapping[str, Any]):
        conv = conversation_for(items[k[1]], r["pos"])
        st, res = await cached_json_call(
            cache,
            llm,
            key=f"class:{k[0]}:{k[1]}",
            version=CLASS_VERSION,
            model=MODEL,
            effort=EFFORT,
            system=CLASS_SYSTEM,
            user=class_user(r["said"], r["segment"], window(conv, r.get("contradicted_by"))),
            schema=CLASS_SCHEMA,
            validate=valid_class,
            spend=spend,
        )
        return k, (res if st == "ok" else None)

    pairs = await gather_limited([one(k, r) for k, r in reqs.items()])
    return dict(pairs), spend


async def run_dedupe(names: Sequence[str], llm: LLM | None, cache: Cache) -> dict[str, str]:
    names = sorted({n for n in names if n and n != NO_CLASS})
    if not names:
        return {}
    st, res = await cached_json_call(
        cache,
        llm,
        key="dedupe:all",
        version=DEDUPE_VERSION,
        model=MODEL,
        effort=EFFORT,
        system=DEDUPE_SYSTEM,
        user="CLASS NAMES:\n" + "\n".join(f"- {n}" for n in names),
        schema=DEDUPE_SCHEMA,
        validate=lambda r: {m.get("name") for m in r.get("mapping") or []} <= set(names),
    )
    if st != "ok" or res is None:
        return {n: n for n in names}
    return apply_merges(names, res["mapping"])


def apply_merges(names: Sequence[str], mapping: Sequence[Mapping[str, str]]) -> dict[str, str]:
    """Rev-8 full run: the dedupe call returns only the renamed classes (a 1,100-name echo
    overran max_tokens); every other name maps to itself. A chain a -> b -> c is resolved."""
    ren = {m["name"]: (m["canonical"].strip() or m["name"]) for m in mapping}
    out = {}
    for n in names:
        c, seen = n, {n}
        while c in ren and ren[c] not in seen:
            c = ren[c]
            seen.add(c)
        out[n] = c
    return out


def assemble(
    raw: Mapping[tuple[str, str], Mapping[str, Any] | None], dedupe: Mapping[str, str]
) -> dict[str, dict[str, Any]]:
    """``{"<said_norm>|<item>": {class, said_canonical, gold, compatible, no_class, no_gold}}``
    with classes mapped to their canonical names. A failed call counts as ``no_class``."""
    out = {}
    for (sn, item), r in raw.items():
        if r is None:
            out[f"{sn}|{item}"] = {
                "class": NO_CLASS,
                "said_canonical": sn,
                "gold": NO_GOLD,
                "compatible": [],
                "no_class": True,
                "no_gold": True,
                "failed": True,
            }
            continue
        cls = r["class"].strip()
        nc = cls == NO_CLASS
        gold = r["gold"].strip()
        out[f"{sn}|{item}"] = {
            "class": NO_CLASS if nc else dedupe.get(cls, cls),
            "said_canonical": r["said_canonical"].strip(),
            "gold": gold or NO_GOLD,
            "compatible": sorted({c.strip() for c in r["compatible"] if c.strip()}),
            "no_class": nc,
            "no_gold": gold in ("", NO_GOLD),
        }
    return out


def canonical_map(classes: Mapping[str, Mapping[str, Any]]) -> dict[tuple[str, str], str]:
    """(said_norm, item) -> said_canonical (the §3a rebuild key of item_sibling_names)."""
    out = {}
    for k, v in classes.items():
        sn, item = k.split("|", 1)
        out[(sn, item)] = U.norm(v["said_canonical"])
    return out


def class_members(classes: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, set[str]]]:
    """class -> {"said": said_canonicals, "gold": golds} (canonical display names)."""
    out: dict[str, dict[str, set[str]]] = defaultdict(lambda: {"said": set(), "gold": set()})
    for v in classes.values():
        if v["class"] == NO_CLASS:
            continue
        out[v["class"]]["said"].add(v["said_canonical"])
        if not v["no_gold"]:
            out[v["class"]]["gold"].add(v["gold"])
    return dict(out)


async def run_concepts(
    members: Mapping[str, Mapping[str, set[str]]], llm: LLM | None, cache: Cache
) -> dict[str, dict[str, Any]]:
    """Per class: C = said + gold (+ fillers until >= 16). Fillers never repeat a said / gold."""

    async def one(cls: str, m: Mapping[str, set[str]]):
        core = sorted(m["said"] | m["gold"], key=str.casefold)
        core_n = {U.norm(c) for c in core}
        fillers: list[str] = []
        need = MIN_CONCEPTS - len(core_n)
        if need > 0:
            st, res = await cached_json_call(
                cache,
                llm,
                key=f"concepts:{cls}",
                version=CONCEPTS_VERSION,
                model=MODEL,
                effort=EFFORT,
                system=CONCEPTS_SYSTEM,
                user=f"CLASS: {cls}\nNUMBER WANTED: {need + 8}\nEXCLUDE:\n"
                + "\n".join(f"- {c}" for c in core),
                schema=CONCEPTS_SCHEMA,
                validate=lambda r: isinstance(r.get("members"), list),
            )
            seen = set(core_n)
            for f in (res or {}).get("members") or [] if st == "ok" else []:
                fn = U.norm(f)
                if fn and fn not in seen and len(fillers) < need:
                    fillers.append(f.strip())
                    seen.add(fn)
        return cls, {
            "concepts": [*core, *fillers],
            "said": sorted(m["said"]),
            "gold": sorted(m["gold"]),
            "fillers": fillers,
        }

    pairs = await gather_limited([one(c, m) for c, m in sorted(members.items())])
    return dict(pairs)


def compatible_of(classes: Mapping[str, Mapping[str, Any]]) -> dict[str, set[str]]:
    """gold (normalised) -> its compatible stratum (normalised), pooled over items."""
    out: dict[str, set[str]] = defaultdict(set)
    for v in classes.values():
        if not v["no_gold"]:
            out[U.norm(v["gold"])] |= {U.norm(c) for c in v["compatible"]}
    return dict(out)
