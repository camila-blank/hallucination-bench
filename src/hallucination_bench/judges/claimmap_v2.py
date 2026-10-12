"""The claim-map judge (``claimmap-v2``): every unit of every readout gets a taxonomy-v2 leaf.

One call per non-empty cell, whatever the v5c class of its readouts (plan 0002, §Inputs). The
system prompt is rendered from :mod:`hallucination_bench.taxonomy_v2`. Two modes (plan rev 13):

* **single-call** (``anchor="none"``, the default): code split -> one ``:M`` call -> code
  roll-up. No v5c spans are passed, so there are no established units, no rule 2, no
  ``dispute`` / ``alt_leaf`` in the schema, and the unit lines carry no flag column;
* **anchored** (``anchor="v5c"``, plan rev 12): the units that are located v5c verified wrong
  spans are marked ``ESTABLISHED`` and the forced / disputed rules apply.

Verify outputs are never shown. Every probe-candidate segment carries ``probe_target``.

Rev 17 (quoted evidence, fail closed): every incorrect segment must carry ``contradicted_by``
and every ``correct`` segment (except ``source = entailed``) ``supported_by``, a verbatim span
of the conversation. :func:`parse`, given the conversation (:func:`conversation_text`: the
prompt plus the response without the marker), locates each quote; a missing or unfindable one
re-labels the segment ``unverifiable`` (tag ``unquoted_contradiction`` / ``unquoted_support``,
the judge's leaf kept in ``judge_leaf``) and counts it (``n_unquoted_contradictions`` /
``n_unquoted_support``).

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
from dataclasses import asdict, dataclass, replace
from typing import Any

from hallucination_bench import taxonomy_v2 as tx
from hallucination_bench.judges.v5c import READ_MARK
from hallucination_bench.llm import schema_block
from hallucination_bench.units import Normalised, Unit

PROMPT_VERSION = "claimmap-v2.7"
ANCHORS: tuple[str, ...] = ("none", "v5c")
MAX_MISSING_SHARE = 0.10

_INTRO_HEAD = (
    "You map the readouts of an interpretability lens. The lens reads a language model's "
    "internal activation while the model writes a response, and its readouts describe what the "
    "model is representing at that point. Code has split every readout into units. Your job is "
    "to put every piece of every readout into exactly one leaf of the claim-map taxonomy below: "
    "label each unit, and split a unit into verbatim sub-spans when it carries more than one "
    "claim or mixes claim and non-claim text. "
)
_INTRO_TAIL = (
    "Judge everything against the WHOLE response and world knowledge, following the precedence "
    "rules in order. Output only the JSON verdict."
)
SYSTEM_INTRO = _INTRO_HEAD + _INTRO_TAIL
SYSTEM_INTRO_V5C = (
    _INTRO_HEAD + "Some units are ESTABLISHED: an earlier audit verified them as wrong about "
    "this conversation (rule 2). " + _INTRO_TAIL
)

SYSTEM = SYSTEM_INTRO + "\n\n" + tx.render()  # single-call (default)
SYSTEM_V5C = SYSTEM_INTRO_V5C + "\n\n" + tx.render(anchored=True)  # --anchor v5c

TOKENS_NOTE = (
    "This lens is a top-k token lens: each readout below is a summariser's one- or two-sentence "
    "interpretation of the lens's top-10 token bag (rule 1, token-bag summaries).\n\n"
)

_USER_HEAD = (
    "A language model was asked:\n\n<prompt>\n{prompt}\n</prompt>\n\n"
    "and wrote the response below. The lens read the model's activation at the token immediately "
    "before the marker {mark}; everything before it is what the model had already written, "
    "everything after it is what it went on to write. The WHOLE response is the context.\n\n"
    "<response>\n{marked_response}\n</response>\n\n"
    "{kind_note}The lens readouts of that activation, pre-split into {n} units. "
)
_USER_TAIL = (
    "splitting; each split quotes its text verbatim from the unit, in left-to-right order) and "
    "rest_leaf (the leaf of the unquoted remainder when you split, else null). Label every unit "
    "exactly once, in order, and no others. Then give a one-sentence rationale."
)
USER_TEMPLATE = (  # single-call (default)
    _USER_HEAD + "Each line is `unit id | unit text` (the text JSON-quoted).\n\n{units_block}\n\n"
    "For EACH unit report: id (exactly as given), leaf, the leaf tags (reason, source, "
    "same_slot_alternative, world, prediction), the evidence quotes (contradicted_by for "
    "incorrect leaves, supported_by for correct), hedged, world_false, cruxy, pair, "
    "probe_target, why (one short clause), splits (an empty list unless the unit needs "
    + _USER_TAIL
)
USER_TEMPLATE_V5C = (
    _USER_HEAD + "Each line is `unit id | flag | unit text` (the text JSON-quoted); the flag "
    "ESTABLISHED marks a span an earlier audit verified as wrong (rule 2), `-` marks an "
    "ordinary unit.\n\n{units_block}\n\n"
    "For EACH unit report: id (exactly as given), leaf, the leaf tags (reason, source, "
    "same_slot_alternative, world, prediction), the evidence quotes (contradicted_by for "
    "incorrect leaves, supported_by for correct), hedged, world_false, cruxy, dispute, "
    "alt_leaf, pair, probe_target, why (one short clause), splits (an empty list unless the "
    "unit needs " + _USER_TAIL
)

_LEAF_ENUM = {"type": "string", "enum": list(tx.JUDGE_LEAF_NAMES)}  # the judge's enum
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
_NULLABLE_STR = {"anyOf": [{"type": "string"}, {"type": "null"}]}


def _seg_props(anchored: bool) -> dict[str, Any]:
    props: dict[str, Any] = {
        "leaf": _LEAF_ENUM,
        "hedged": {"type": "boolean"},
        "world_false": {"type": "boolean"},
        "cruxy": {"type": "boolean"},
    }
    if anchored:
        props |= {"dispute": {"type": "boolean"}, "alt_leaf": _NULLABLE_LEAF}
    return props | {
        "pair": _PAIR,
        "probe_target": _NULLABLE_STR,
        "reason": {"anyOf": [{"type": "string", "enum": list(tx.REASONS)}, {"type": "null"}]},
        "source": {"anyOf": [{"type": "string", "enum": list(tx.SOURCES)}, {"type": "null"}]},
        **{f: {"type": "boolean"} for f in tx.UNVERIFIABLE_FLAGS},
        "contradicted_by": _NULLABLE_STR,
        "supported_by": _NULLABLE_STR,
        "why": {"type": "string"},
    }


def _schemas(anchored: bool) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    props = _seg_props(anchored)
    split: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["text", *props],
        "properties": {"text": {"type": "string"}, **props},
    }
    unit: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["id", *props, "splits", "rest_leaf"],
        "properties": {
            "id": {"type": "string"},
            **props,
            "splits": {"type": "array", "items": split},
            "rest_leaf": _NULLABLE_LEAF,
        },
    }
    schema = schema_block(
        "claimmap_v2",
        {"units": {"type": "array", "items": unit}, "rationale": {"type": "string"}},
        ["units", "rationale"],
    )
    return split, unit, schema


SPLIT_SCHEMA, UNIT_SCHEMA, SCHEMA = _schemas(False)  # single-call (default)
SPLIT_SCHEMA_V5C, UNIT_SCHEMA_V5C, SCHEMA_V5C = _schemas(True)

PROMPTS: dict[str, str] = {
    "SYSTEM": SYSTEM,
    "USER_TEMPLATE": USER_TEMPLATE,
    "SYSTEM_V5C": SYSTEM_V5C,
    "USER_TEMPLATE_V5C": USER_TEMPLATE_V5C,
}


def check_anchor(anchor: str) -> bool:
    """``True`` for the anchored mode (``v5c``); raises on an unknown anchor."""
    if anchor not in ANCHORS:
        raise ValueError(f"unknown anchor {anchor!r} (known: {', '.join(ANCHORS)})")
    return anchor == "v5c"


def schema_for(anchor: str = "none") -> dict[str, Any]:
    return SCHEMA_V5C if check_anchor(anchor) else SCHEMA


def plain_units(units: Sequence[Unit]) -> list[Unit]:
    """Single-call mode: the same units (ids, offsets) with every established flag dropped, so
    they are shown and parsed as ordinary text (no forcing, no disputes)."""
    return [replace(u, established=False) if u.established else u for u in units]


_PLACEHOLDER = re.compile(r"\{(prompt|mark|marked_response|kind_note|n|units_block)\}")


def units_block(units: Sequence[Unit], *, flags: bool = True) -> str:
    out: list[str] = []
    for r in sorted({u.readout for u in units}):
        out.append(f'<readout idx="{r}">')
        for u in units:
            if u.readout == r:
                text = json.dumps(u.text, ensure_ascii=False)
                if flags:
                    flag = "ESTABLISHED" if u.established else "-"
                    out.append(f"{u.id} | {flag} | {text}")
                else:
                    out.append(f"{u.id} | {text}")
        out.append("</readout>")
        out.append("")
    return "\n".join(out).rstrip()


def render_prompt(
    prompt: str,
    marked_response: str,
    units: Sequence[Unit],
    *,
    kind: str = "prose",
    anchor: str = "none",
) -> tuple[str, str]:
    """(system, user) for one claim-map call; ``marked_response`` carries ``⟦READ⟧``. In
    single-call mode (``anchor="none"``) established flags are never shown."""
    anchored = check_anchor(anchor)
    if not units:
        raise ValueError("render_prompt needs at least one unit")
    values = {
        "prompt": prompt,
        "mark": READ_MARK,
        "marked_response": marked_response,
        "kind_note": TOKENS_NOTE if kind == "tokens" else "",
        "n": str(len(units)),
        "units_block": units_block(units, flags=anchored),
    }
    system, template = (SYSTEM_V5C, USER_TEMPLATE_V5C) if anchored else (SYSTEM, USER_TEMPLATE)
    # one pass: braces inside the substituted text are never re-expanded
    return system, _PLACEHOLDER.sub(lambda m: values[m.group(1)], template)


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
    probe_target: str | None = None
    reason: str | None = None  # non_claim only
    source: str | None = None  # correct only
    same_slot_alternative: bool = False  # unverifiable only
    world: bool = False  # unverifiable only
    prediction: bool = False  # unverifiable only
    contradicted_by: str | None = None  # incorrect leaves only (rev 17)
    supported_by: str | None = None  # correct only (rev 17)
    unquoted_contradiction: bool = False  # re-labelled unverifiable: no locatable quote
    unquoted_support: bool = False
    judge_leaf: str | None = None  # the judge's leaf when a quote check re-labelled it
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
    "n_probe_candidates",
    "n_missing_probe_targets",
    "n_contradiction_quotes_checked",
    "n_unquoted_contradictions",
    "n_support_quotes_checked",
    "n_unquoted_support",
)


def _b(x: Any) -> bool:
    return x is True


def _pair(x: Any) -> dict[str, str] | None:
    if isinstance(x, dict) and isinstance(x.get("said"), str) and isinstance(x.get("actual"), str):
        return {"said": x["said"], "actual": x["actual"]}
    return None


def _probe_target(x: Any) -> str | None:
    return x.strip() if isinstance(x, str) and x.strip() else None


def _enum(x: Any, allowed: Sequence[str]) -> str | None:
    return x if isinstance(x, str) and x in allowed else None


def _leaf_tags(s: "Segment") -> None:
    """Rev 16 leaf tags are kept only on their leaf: ``reason`` on ``non_claim``, ``source`` on
    ``correct``, the three flags on ``unverifiable``."""
    if s.raw_leaf not in ("non_claim", "junk"):
        s.reason = None
    if s.raw_leaf != "correct":
        s.source = None
    if s.raw_leaf != "unverifiable":
        s.same_slot_alternative = s.world = s.prediction = False
    if s.raw_leaf not in tx.INCORRECT_LEAVES and not s.unquoted_contradiction:
        s.contradicted_by = None
    if s.raw_leaf != "correct" and not s.unquoted_support:
        s.supported_by = None


_QUOTE_TRIM = " \t\n\"'“”‘’«»「」…."


def _quote(x: Any) -> str | None:
    if not isinstance(x, str):
        return None
    q = x.strip()
    while q.startswith("..."):
        q = q[3:].strip()
    while q.endswith("..."):
        q = q[:-3].strip()
    q = q.strip(_QUOTE_TRIM)
    return q or None


def _check_quotes(s: "Segment", conv: Normalised, c: dict[str, int]) -> None:
    """Rev 17, fail closed: an incorrect segment needs a locatable ``contradicted_by``; a
    ``correct`` one (unless ``source = entailed``) a locatable ``supported_by``. Otherwise it is
    re-labelled ``unverifiable`` (the judge's leaf kept in ``judge_leaf``)."""
    if s.status != "ok":
        return
    if s.raw_leaf in tx.INCORRECT_LEAVES:
        c["n_contradiction_quotes_checked"] += 1
        q = s.contradicted_by
        if q is None or conv.find(q, 0) is None:
            c["n_unquoted_contradictions"] += 1
            s.judge_leaf, s.leaf, s.raw_leaf = s.raw_leaf, "unverifiable", "unverifiable"
            s.unquoted_contradiction = True
    elif s.raw_leaf == "correct" and s.source != "entailed":
        c["n_support_quotes_checked"] += 1
        q = s.supported_by
        if q is None or conv.find(q, 0) is None:
            c["n_unquoted_support"] += 1
            s.judge_leaf, s.leaf, s.raw_leaf = s.raw_leaf, "unverifiable", "unverifiable"
            s.unquoted_support = True


def _leaf(x: Any) -> str | None:
    # judge leaves (answers) and rev-18 scoring leaves (gold) both parse
    return x if isinstance(x, str) and (x in tx.JUDGE_LEAVES or x in tx.LEAVES) else None


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


def conversation_text(prompt: str, marked_response: str) -> str:
    """What evidence quotes are checked against: the prompt and the response (marker removed,
    so a quote may run across it)."""
    return f"{prompt}\n{marked_response.replace(READ_MARK, '')}"


def parse(
    raw: dict[str, Any] | None,
    units: Sequence[Unit],
    conversation: str | None = None,
) -> tuple[list[Segment], dict[str, int]]:
    """Answer -> segments covering every unit, fail closed (see the module docstring). With
    ``conversation`` (production, calibration, consistency) the rev-17 evidence quotes are
    checked; ``None`` skips the check (unit tests of the other mechanics only)."""
    conv = Normalised(conversation) if conversation is not None else None
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
            tsrc = e if p.kind == "rest" else src  # leaf tags of a rest piece: the unit's
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
                probe_target=_probe_target(tsrc.get("probe_target")),
                reason=_enum(tsrc.get("reason"), tx.REASONS),
                source=_enum(tsrc.get("source"), tx.SOURCES),
                same_slot_alternative=_b(tsrc.get("same_slot_alternative")),
                world=_b(tsrc.get("world")),
                prediction=_b(tsrc.get("prediction")),
                contradicted_by=_quote(tsrc.get("contradicted_by")),
                supported_by=_quote(tsrc.get("supported_by")),
                why=str(src.get("why") or e.get("why") or ""),
            )
            if leaf is None:
                c["n_bad_leaves"] += 1
            useg.append(seg)
        if conv is not None:
            for s in useg:
                _check_quotes(s, conv, c)
        _established_rules(u, useg, c)
        for s in useg:
            if s.raw_leaf not in tx.INCORRECT_LEAVES:
                s.cruxy = False  # cruxy is an incorrect-only tag
            if s.raw_leaf not in tx.SUBSTITUTION_LEAVES:
                s.pair = None  # pair is logged for substitutions only
            elif s.status == "ok" and s.pair is None:
                c["n_missing_pairs"] += 1
            _leaf_tags(s)
            if s.raw_leaf not in tx.PROBE_CANDIDATE_LEAVES or s.status != "ok":
                s.probe_target = None  # probe_target is kept for probe candidates only
            else:
                c["n_probe_candidates"] += 1
                if s.probe_target is None:
                    c["n_missing_probe_targets"] += 1  # counted; the leaf is kept
        segs += useg
    for s in segs:
        _merge_junk(s)
    c["n_segments"] = len(segs)
    return segs, c


def _merge_junk(s: "Segment") -> None:
    """Rev 18, after parsing: the judge's ``non_claim`` (reason kept) and ``off_topic`` (reason
    ``off_topic``) become the scoring leaf ``junk``; the prompt and cached answers are unchanged."""
    if s.raw_leaf == "off_topic" or (s.leaf == "off_topic" and s.raw_leaf != "non_claim"):
        s.reason = "off_topic"
    for f in ("leaf", "raw_leaf", "alt_leaf"):
        v = getattr(s, f)
        if v in tx.JUDGE_TO_LEAF:
            setattr(s, f, tx.JUDGE_TO_LEAF[v])


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
