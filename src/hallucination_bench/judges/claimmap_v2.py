"""The claim-map judge (``claimmap-v2``): every unit of every readout gets a taxonomy-v2 leaf.

One call per non-empty cell, whatever the v5c class of its readouts (plan 0002, §Inputs). The
system prompt is rendered from :mod:`hallucination_bench.taxonomy_v2`. The user prompt shows the
prompt, the marked response and each readout as its code-split units (:mod:`units`), with the
units that are located v5c verified wrong spans marked ``ESTABLISHED``. Verify outputs are never
shown.

:func:`parse` turns the answer into **segments** (a unit, or a located sub-span of a split unit,
or a piece of a split unit's unquoted remainder); segments never overlap and cover every unit.
It applies the established-span rules (forced / disputed -> ``unknown_incorrect``) and fails
closed: a missing unit, an unknown leaf or a failed call gives ``failed`` segments.
:func:`validate` rejects (caches as failed, re-queues) an answer with a wrong readout count or
more than 10% missing units.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from hallucination_bench import taxonomy_v2 as tx
from hallucination_bench.judges.v5c import READ_MARK
from hallucination_bench.llm import schema_block
from hallucination_bench.units import Normalised, Unit

PROMPT_VERSION = "claimmap-v2.1"
MAX_MISSING_SHARE = 0.10

SYSTEM_INTRO = (
    "You map the readouts of an interpretability lens. The lens reads a language model's "
    "internal activation while the model writes a response, and its readouts describe what the "
    "model is representing at that point. Code has split every readout into units. Your job is "
    "to put every piece of every readout into exactly one leaf of the claim-map taxonomy below: "
    "label each unit, and split a unit into verbatim sub-spans when it carries more than one "
    "claim or mixes claim and non-claim text. Some units are ESTABLISHED: an earlier audit "
    "verified them as wrong about this conversation (rule 2). Judge everything against the "
    "WHOLE response and world knowledge, following the precedence rules in order. Output only "
    "the JSON verdict."
)

SYSTEM = SYSTEM_INTRO + "\n\n" + tx.render()

TOKENS_NOTE = (
    "This lens is a top-k token lens: each readout below is a summariser's one- or two-sentence "
    "interpretation of the lens's top-10 token bag (rule 1, token-bag summaries).\n\n"
)

USER_TEMPLATE = (
    "A language model was asked:\n\n<prompt>\n{prompt}\n</prompt>\n\n"
    "and wrote the response below. The lens read the model's activation at the token immediately "
    "before the marker {mark}; everything before it is what the model had already written, "
    "everything after it is what it went on to write. The WHOLE response is the context.\n\n"
    "<response>\n{marked_response}\n</response>\n\n"
    "{kind_note}The lens readouts of that activation, pre-split into {n} units. Each line is "
    "`unit id | flag | unit text` (the text JSON-quoted); the flag ESTABLISHED marks a span an "
    "earlier audit verified as wrong (rule 2), `-` marks an ordinary unit.\n\n{units_block}\n\n"
    "For EACH unit report: id (exactly as given), leaf, hedged, world_false, cruxy, dispute, "
    "alt_leaf, pair, why (one short clause), splits (an empty list unless the unit needs "
    "splitting; each split quotes its text verbatim from the unit, in left-to-right order) and "
    "rest_leaf (the leaf of the unquoted remainder when you split, else null). Label every unit "
    "exactly once, in order, and no others. Then give a one-sentence rationale."
)

_LEAF_ENUM = {"type": "string", "enum": list(tx.LEAF_NAMES)}
_NULLABLE_LEAF = {"anyOf": [_LEAF_ENUM, {"type": "null"}]}
_PAIR = {
    "anyOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["said", "actual"],
            "properties": {"said": {"type": "string"}, "actual": {"type": "string"}},
        },
        {"type": "null"},
    ]
}
_SEG_PROPS: dict[str, Any] = {
    "leaf": _LEAF_ENUM,
    "hedged": {"type": "boolean"},
    "world_false": {"type": "boolean"},
    "cruxy": {"type": "boolean"},
    "dispute": {"type": "boolean"},
    "alt_leaf": _NULLABLE_LEAF,
    "pair": _PAIR,
    "why": {"type": "string"},
}
_SEG_REQ = list(_SEG_PROPS)
SPLIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["text", *_SEG_REQ],
    "properties": {"text": {"type": "string"}, **_SEG_PROPS},
}
UNIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["id", *_SEG_REQ, "splits", "rest_leaf"],
    "properties": {
        "id": {"type": "string"},
        **_SEG_PROPS,
        "splits": {"type": "array", "items": SPLIT_SCHEMA},
        "rest_leaf": _NULLABLE_LEAF,
    },
}
SCHEMA: dict[str, Any] = schema_block(
    "claimmap_v2",
    {"units": {"type": "array", "items": UNIT_SCHEMA}, "rationale": {"type": "string"}},
    ["units", "rationale"],
)

PROMPTS: dict[str, str] = {"SYSTEM": SYSTEM, "USER_TEMPLATE": USER_TEMPLATE}
_PLACEHOLDER = re.compile(r"\{(prompt|mark|marked_response|kind_note|n|units_block)\}")


def units_block(units: Sequence[Unit]) -> str:
    out: list[str] = []
    for r in sorted({u.readout for u in units}):
        out.append(f'<readout idx="{r}">')
        for u in units:
            if u.readout == r:
                flag = "ESTABLISHED" if u.established else "-"
                out.append(f"{u.id} | {flag} | {json.dumps(u.text, ensure_ascii=False)}")
        out.append("</readout>")
        out.append("")
    return "\n".join(out).rstrip()


def render_prompt(
    prompt: str, marked_response: str, units: Sequence[Unit], *, kind: str = "prose"
) -> tuple[str, str]:
    """(system, user) for one claim-map call; ``marked_response`` carries ``⟦READ⟧``."""
    if not units:
        raise ValueError("render_prompt needs at least one unit")
    values = {
        "prompt": prompt,
        "mark": READ_MARK,
        "marked_response": marked_response,
        "kind_note": TOKENS_NOTE if kind == "tokens" else "",
        "n": str(len(units)),
        "units_block": units_block(units),
    }
    # one pass: braces inside the substituted text are never re-expanded
    return SYSTEM, _PLACEHOLDER.sub(lambda m: values[m.group(1)], USER_TEMPLATE)


# ---------------------------------------------------------------- ids / validation

_ID = re.compile(r"^\s*R(\d+)\s*\.\s*U(\d+)\s*$", re.IGNORECASE)


def norm_id(x: Any) -> tuple[int, int] | None:
    if not isinstance(x, str):
        return None
    m = _ID.match(x)
    return (int(m.group(1)), int(m.group(2))) if m else None


def validate(raw: dict[str, Any] | None, units: Sequence[Unit]) -> bool:
    """Well-shaped, the answer's readouts are exactly the cell's, and at most 10% of the units
    are missing. Duplicates, extra ids and bad leaves do not invalidate (parse counts them)."""
    if not isinstance(raw, dict) or not isinstance(raw.get("units"), list):
        return False
    if not all(isinstance(e, dict) for e in raw["units"]):
        return False
    want = {(u.readout, u.n) for u in units}
    got = {k for e in raw["units"] if (k := norm_id(e.get("id"))) is not None}
    if {r for r, _ in got} != {u.readout for u in units}:
        return False
    missing = len(want - got)
    return missing <= MAX_MISSING_SHARE * len(want)


# ---------------------------------------------------------------- segments


@dataclass
class Segment:
    unit: str
    readout: int
    start: int  # raw offsets in the readout
    end: int
    text: str
    leaf: str  # the scored leaf (forced / disputed -> unknown_incorrect; failed)
    raw_leaf: str  # the judge's leaf (``failed`` when there is none)
    status: str = "ok"  # ok | forced | disputed | missing | failed | bad_leaf
    established: bool = False
    kind: str = "unit"  # unit | split | rest
    alt_leaf: str | None = None
    hedged: bool = False
    world_false: bool = False
    cruxy: bool = False
    dispute: bool = False
    pair: dict[str, str] | None = None
    why: str = ""

    @property
    def chars(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PARSE_COUNTS: tuple[str, ...] = (
    "n_units",
    "n_established_units",
    "n_segments",
    "n_missing_units",
    "n_duplicate_units",
    "n_extra_units",
    "n_bad_leaves",
    "n_splits",
    "n_unlocated_subspans",
    "n_missing_rest",
    "n_forced_established",
    "n_disputed_established",
    "n_missing_pairs",
    "n_dispute_ignored",
)


def _b(x: Any) -> bool:
    return x is True


def _pair(x: Any) -> dict[str, str] | None:
    if isinstance(x, dict) and isinstance(x.get("said"), str) and isinstance(x.get("actual"), str):
        return {"said": x["said"], "actual": x["actual"]}
    return None


def _leaf(x: Any) -> str | None:
    return x if isinstance(x, str) and x in tx.LEAVES else None


@dataclass
class _Piece:
    start: int
    end: int
    entry: dict[str, Any] | None  # split entry (tags) or None for the rest
    leaf: str | None
    kind: str


def _pieces(u: Unit, e: dict[str, Any], c: dict[str, int]) -> list[_Piece]:
    """Located splits plus rest pieces (unit-relative offsets), or the whole unit."""
    splits = [s for s in (e.get("splits") or []) if isinstance(s, dict)]
    if not splits:
        return [_Piece(0, len(u.text), e, _leaf(e.get("leaf")), "unit")]
    norm = Normalised(u.text)
    located: list[_Piece] = []
    pos = 0
    for s in splits:
        c["n_splits"] += 1
        t = s.get("text")
        m = norm.find(t, pos) if isinstance(t, str) else None
        if m is None:
            c["n_unlocated_subspans"] += 1
            continue
        located.append(_Piece(m[0], m[1], s, _leaf(s.get("leaf")), "split"))
        pos = m[1]
    rest_leaf = _leaf(e.get("rest_leaf"))
    prev = 0
    gaps: list[tuple[int, int]] = []
    for p in located:
        gaps.append((prev, p.start))
        prev = p.end
    gaps.append((prev, len(u.text)))
    rest: list[_Piece] = []
    for a, b in gaps:
        while a < b and u.text[a].isspace():
            a += 1
        while b > a and u.text[b - 1].isspace():
            b -= 1
        if a < b:
            rest.append(_Piece(a, b, None, rest_leaf, "rest"))
    if rest and rest_leaf is None:
        c["n_missing_rest"] += 1
        fallback = _leaf(e.get("leaf"))
        for p in rest:
            p.leaf = fallback
    return sorted(located + rest, key=lambda p: p.start)


def parse(
    raw: dict[str, Any] | None, units: Sequence[Unit]
) -> tuple[list[Segment], dict[str, int]]:
    """Answer -> segments covering every unit, fail closed (see the module docstring)."""
    c = dict.fromkeys(PARSE_COUNTS, 0)
    c["n_units"] = len(units)
    c["n_established_units"] = sum(u.established for u in units)
    entries: dict[tuple[int, int], dict[str, Any]] = {}
    known = {(u.readout, u.n) for u in units}
    if isinstance(raw, dict) and isinstance(raw.get("units"), list):
        for e in raw["units"]:
            if not isinstance(e, dict):
                continue
            k = norm_id(e.get("id"))
            if k is None or k not in known:
                c["n_extra_units"] += 1
                continue
            if k in entries:
                c["n_duplicate_units"] += 1
                continue
            entries[k] = e
    segs: list[Segment] = []
    for u in units:
        e = entries.get((u.readout, u.n))
        if e is None:
            status = "failed" if raw is None else "missing"
            if raw is not None:
                c["n_missing_units"] += 1
            segs.append(
                Segment(
                    u.id,
                    u.readout,
                    u.start,
                    u.end,
                    u.text,
                    tx.FAILED,
                    tx.FAILED,
                    status,
                    u.established,
                )
            )
            continue
        useg: list[Segment] = []
        for p in _pieces(u, e, c):
            src = p.entry if p.entry is not None else {}
            leaf = p.leaf
            seg = Segment(
                u.id,
                u.readout,
                u.start + p.start,
                u.start + p.end,
                u.text[p.start : p.end],
                leaf or tx.FAILED,
                leaf or tx.FAILED,
                "ok" if leaf else "bad_leaf",
                u.established,
                p.kind,
                alt_leaf=_leaf(src.get("alt_leaf")),
                hedged=_b(src.get("hedged")),
                world_false=_b(src.get("world_false")),
                cruxy=_b(src.get("cruxy")),
                dispute=_b(src.get("dispute")),
                pair=_pair(src.get("pair")),
                why=str(src.get("why") or e.get("why") or ""),
            )
            if leaf is None:
                c["n_bad_leaves"] += 1
            useg.append(seg)
        _established_rules(u, useg, c)
        for s in useg:
            if s.raw_leaf not in tx.INCORRECT_LEAVES:
                s.cruxy = False  # cruxy is an incorrect-only tag
            if s.raw_leaf not in tx.SUBSTITUTION_LEAVES:
                s.pair = None  # pair is logged for substitutions only
            elif s.status == "ok" and s.pair is None:
                c["n_missing_pairs"] += 1
        segs += useg
    c["n_segments"] = len(segs)
    return segs, c


def _established_rules(u: Unit, segs: list[Segment], c: dict[str, int]) -> None:
    """Rule 2 in code. Non-established units: dispute / alt_leaf are ignored. Established
    units: a segment with ``dispute`` and a non-incorrect ``alt_leaf`` is disputed; if no
    undisputed segment has an incorrect leaf, every other non-incorrect segment is forced. Both
    score as ``unknown_incorrect``."""
    if not u.established:
        for s in segs:
            if s.dispute or s.alt_leaf is not None:
                c["n_dispute_ignored"] += 1
            s.dispute, s.alt_leaf = False, None
        return
    for s in segs:
        if s.status != "ok":
            continue
        if s.dispute and s.alt_leaf is not None and s.alt_leaf not in tx.INCORRECT_LEAVES:
            s.status = "disputed"
            s.leaf = tx.UNKNOWN_INCORRECT
            c["n_disputed_established"] += 1
        else:
            s.dispute, s.alt_leaf = False, None
    if any(s.status == "ok" and s.raw_leaf in tx.INCORRECT_LEAVES for s in segs):
        return
    for s in segs:
        if s.status == "ok" and s.raw_leaf not in tx.INCORRECT_LEAVES:
            s.status = "forced"
            s.leaf = tx.UNKNOWN_INCORRECT
            c["n_forced_established"] += 1


def failed_segments(units: Sequence[Unit]) -> list[Segment]:
    """Every unit ``failed``: a failed, pending-forever or invalid call."""
    segs, _ = parse(None, units)
    return segs


def resolved_leaf(seg: dict[str, Any] | Segment) -> str:
    """The leaf calibration scores: a disputed segment by its ``alt_leaf``, a forced one by the
    judge's raw leaf; every other segment by its leaf."""
    d = seg.to_dict() if isinstance(seg, Segment) else seg
    if d["status"] == "disputed" and d.get("alt_leaf"):
        return d["alt_leaf"]
    if d["status"] == "forced":
        return d["raw_leaf"]
    return d["leaf"]


def alt_leaf_view(seg: dict[str, Any] | Segment) -> str:
    """The report's sensitivity view: disputed segments re-scored with their alternative."""
    d = seg.to_dict() if isinstance(seg, Segment) else seg
    if d["status"] == "disputed" and d.get("alt_leaf"):
        return d["alt_leaf"]
    return d["leaf"]
