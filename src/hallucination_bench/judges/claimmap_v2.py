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
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
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


def _seg_props(anchored: bool, v28: bool = False, v29: bool = False) -> dict[str, Any]:
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
        **({"continuation_text": {"type": "boolean"}} if v28 else {}),
        **(
            {"contradicted_before_marker": {"type": "boolean"}, "prefix_clash": _NULLABLE_STR}
            if v29
            else {}
        ),
        "contradicted_by": _NULLABLE_STR,
        "supported_by": _NULLABLE_STR,
        "why": {"type": "string"},
    }


def _schemas(
    anchored: bool, v28: bool = False, v29: bool = False
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    props = _seg_props(anchored, v28, v29)
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


# ---------------------------------------------------------------- versions (plan 0029 §3.1)

# claimmap-v2.8: the v2.7 user templates with the `{joint_note}` placeholder (empty for
# per-layer cells; the 11-layer note for joint cells) and `continuation_text` among the leaf tags
_JOINT_SLOT = ("{kind_note}The lens readouts", "{kind_note}{joint_note}The lens readouts")
_TAG_LIST = (
    "world, prediction), the evidence",
    "world, prediction, continuation_text), the evidence",
)


def _v28_template(t: str) -> str:
    assert t.count(_JOINT_SLOT[0]) == 1 and t.count(_TAG_LIST[0]) == 1
    return t.replace(*_JOINT_SLOT).replace(*_TAG_LIST)


SYSTEM_V28 = SYSTEM_INTRO + "\n\n" + tx.render(version=tx.V28)
SYSTEM_V5C_V28 = SYSTEM_INTRO_V5C + "\n\n" + tx.render(anchored=True, version=tx.V28)
USER_TEMPLATE_V28 = _v28_template(USER_TEMPLATE)
USER_TEMPLATE_V5C_V28 = _v28_template(USER_TEMPLATE_V5C)
SPLIT_SCHEMA_V28, UNIT_SCHEMA_V28, SCHEMA_V28 = _schemas(False, v28=True)
SPLIT_SCHEMA_V5C_V28, UNIT_SCHEMA_V5C_V28, SCHEMA_V5C_V28 = _schemas(True, v28=True)

# claimmap-v2.9 (plan 0029 amendment): the v2.8 templates with the two prefix-check fields among
# the leaf tags; the v2.9 taxonomy (slot rule, prefix check, EXAMPLES_V29); schemas with both
# fields required.
_TAG_LIST_V29 = (
    _TAG_LIST[1],
    "world, prediction, continuation_text, contradicted_before_marker, prefix_clash), the evidence",
)


def _v29_template(t: str) -> str:
    t8 = _v28_template(t)
    assert t8.count(_TAG_LIST_V29[0]) == 1
    return t8.replace(*_TAG_LIST_V29)


SYSTEM_V29 = SYSTEM_INTRO + "\n\n" + tx.render(version=tx.V29)
SYSTEM_V5C_V29 = SYSTEM_INTRO_V5C + "\n\n" + tx.render(anchored=True, version=tx.V29)
USER_TEMPLATE_V29 = _v29_template(USER_TEMPLATE)
USER_TEMPLATE_V5C_V29 = _v29_template(USER_TEMPLATE_V5C)
SPLIT_SCHEMA_V29, UNIT_SCHEMA_V29, SCHEMA_V29 = _schemas(False, v28=True, v29=True)
SPLIT_SCHEMA_V5C_V29, UNIT_SCHEMA_V5C_V29, SCHEMA_V5C_V29 = _schemas(True, v28=True, v29=True)


@dataclass(frozen=True)
class Spec:
    """One registered claim-map prompt version: system / user templates and schemas for both
    modes, and the taxonomy / scoring versions it renders and is scored under."""

    version: str
    system: str
    user_template: str
    system_v5c: str
    user_template_v5c: str
    schema: dict[str, Any]
    schema_v5c: dict[str, Any]
    taxonomy_version: str
    scoring_version: str

    @property
    def v28(self) -> bool:
        return tx.is_v28(self.version)

    @property
    def v29(self) -> bool:
        return tx.is_v29(self.version)

    def definitions(self, anchored: bool = False) -> list[str]:
        return tx.definitions(anchored, self.version)


VERSIONS: dict[str, Spec] = {
    "claimmap-v2.7": Spec(
        "claimmap-v2.7",
        SYSTEM,
        USER_TEMPLATE,
        SYSTEM_V5C,
        USER_TEMPLATE_V5C,
        SCHEMA,
        SCHEMA_V5C,
        tx.TAXONOMY_VERSION,
        tx.SCORING_VERSION,
    ),
    "claimmap-v2.8": Spec(
        "claimmap-v2.8",
        SYSTEM_V28,
        USER_TEMPLATE_V28,
        SYSTEM_V5C_V28,
        USER_TEMPLATE_V5C_V28,
        SCHEMA_V28,
        SCHEMA_V5C_V28,
        tx.TAXONOMY_VERSION_V28,
        tx.SCORING_VERSION_V28,
    ),
    "claimmap-v2.9": Spec(
        "claimmap-v2.9",
        SYSTEM_V29,
        USER_TEMPLATE_V29,
        SYSTEM_V5C_V29,
        USER_TEMPLATE_V5C_V29,
        SCHEMA_V29,
        SCHEMA_V5C_V29,
        tx.TAXONOMY_VERSION_V29,
        tx.SCORING_VERSION_V29,
    ),
}
assert VERSIONS[PROMPT_VERSION].system is SYSTEM and set(VERSIONS) == set(tx.PROMPT_VERSIONS)


def spec(version: str | None = None) -> Spec:
    """The registered spec (``None`` = claimmap-v2.7, what a root without a record used)."""
    v = version or PROMPT_VERSION
    if v not in VERSIONS:
        raise ValueError(f"unknown claim-map prompt {v!r} (known: {', '.join(VERSIONS)})")
    return VERSIONS[v]


def is_v28(version: str | None) -> bool:
    """Every registered version >= claimmap-v2.8 (they get ``read_offset`` and the new fields)."""
    return spec(version).v28


def is_v29(version: str | None) -> bool:
    """Every registered version >= claimmap-v2.9 (the prefix check: ``contradicted_before_marker``
    / ``prefix_clash``)."""
    return spec(version).v29


def check_anchor(anchor: str) -> bool:
    """``True`` for the anchored mode (``v5c``); raises on an unknown anchor."""
    if anchor not in ANCHORS:
        raise ValueError(f"unknown anchor {anchor!r} (known: {', '.join(ANCHORS)})")
    return anchor == "v5c"


def schema_for(anchor: str = "none", version: str | None = None) -> dict[str, Any]:
    sp = spec(version)
    return sp.schema_v5c if check_anchor(anchor) else sp.schema


JOINT_LAYERS: tuple[int, ...] = tuple(range(20, 61, 4))  # R0 = L20 ... R10 = L60 (plan §3.7)


def joint_note(layers: Mapping[int, int]) -> str:
    """The user-prompt sentence of a joint 11-layer cell (plan 0029 §3.4): ``layers`` maps the
    readout index ``i`` to its layer."""
    if not layers:
        return ""
    parts = ", ".join(f"R{i} = layer {la}" for i, la in sorted(layers.items()))
    return (
        f"The readouts below come from {len(layers)} layers at the same position; {parts} (one "
        "sample of each layer's readout).\n\n"
    )


def plain_units(units: Sequence[Unit]) -> list[Unit]:
    """Single-call mode: the same units (ids, offsets) with every established flag dropped, so
    they are shown and parsed as ordinary text (no forcing, no disputes)."""
    return [replace(u, established=False) if u.established else u for u in units]


_PLACEHOLDER = re.compile(r"\{(prompt|mark|marked_response|kind_note|joint_note|n|units_block)\}")


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
    version: str | None = None,
    joint_note: str = "",
) -> tuple[str, str]:
    """(system, user) for one claim-map call; ``marked_response`` carries ``⟦READ⟧``. In
    single-call mode (``anchor="none"``) established flags are never shown. ``version`` picks
    the registered prompt (default claimmap-v2.7); ``joint_note`` (v2.8 only) fills the joint
    cell's ``{joint_note}``."""
    anchored = check_anchor(anchor)
    sp = spec(version)
    if not units:
        raise ValueError("render_prompt needs at least one unit")
    if joint_note and not sp.v28:
        raise ValueError(f"{sp.version} has no joint unit")
    values = {
        "prompt": prompt,
        "mark": READ_MARK,
        "marked_response": marked_response,
        "kind_note": TOKENS_NOTE if kind == "tokens" else "",
        "joint_note": joint_note,
        "n": str(len(units)),
        "units_block": units_block(units, flags=anchored),
    }
    system, template = (
        (sp.system_v5c, sp.user_template_v5c) if anchored else (sp.system, sp.user_template)
    )
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
    # plan 0029 (claimmap-v2.8 roots only; never emitted for v2.7, so v7 rebuilds are unchanged)
    continuation_text: bool = False  # incorrect only: written as the text's own continuation
    contradiction_region: str | None = None  # incorrect only: prefix | after (needs read_offset)
    # plan 0029 amendment (claimmap-v2.9 roots only; never emitted for v2.7 / v2.8)
    contradicted_before_marker: bool | None = None  # incorrect only; None = field missing
    prefix_clash: str | None = None  # the judge's quote from the text before the marker
    prefix_clash_region: str | None = None  # prefix | after | None (not located / no quote)
    # plan 0029 Amendment 2 (v2.9 roots with the continuation verifier only): a candidate's
    # verdict (yes | no | failed | pending; None = not a candidate) and its three checks
    excuse_verdict: str | None = None
    excuse_checks: dict[str, bool] | None = None
    v28: bool = field(default=False, repr=False)  # emit the v2.8 fields (never emitted itself)
    v29: bool = field(default=False, repr=False)  # emit the v2.9 fields (never emitted itself)
    cv: bool = field(default=False, repr=False)  # emit the verifier fields (never emitted itself)

    @property
    def chars(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        del d["v28"], d["v29"], d["cv"]
        if not self.v28:
            for f in V28_SEGMENT_FIELDS:
                del d[f]
        if not self.v29:
            for f in V29_SEGMENT_FIELDS:
                del d[f]
        if not self.cv:
            for f in VERIFIER_SEGMENT_FIELDS:
                del d[f]
        return d


V28_SEGMENT_FIELDS: tuple[str, ...] = ("continuation_text", "contradiction_region")
V29_SEGMENT_FIELDS: tuple[str, ...] = (
    "contradicted_before_marker",
    "prefix_clash",
    "prefix_clash_region",
)
VERIFIER_SEGMENT_FIELDS: tuple[str, ...] = ("excuse_verdict", "excuse_checks")


@dataclass(frozen=True)
class ExcuseVerdict:
    """Plan 0029 Amendment 2: the continuation verifier's answer for one candidate segment
    (``status`` ok | failed | pending; the three checks only when ok)."""

    status: str
    c1: bool = False
    c2: bool = False
    c3: bool = False

    @property
    def yes(self) -> bool:
        return self.status == "ok" and self.c1 and self.c2 and self.c3

    @property
    def label(self) -> str:
        return ("yes" if self.yes else "no") if self.status == "ok" else self.status


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
    "n_hedged_remapped",
)
# plan 0029: the v2.8 counters, emitted only for v2.8 roots (after PARSE_COUNTS)
PARSE_COUNTS_V28_EXTRA: tuple[str, ...] = (
    "n_continuation_miss",
    "n_contradiction_prefix",
    "n_contradiction_after",
    "n_short_quotes",
)
# plan 0029 amendment: the v2.9 counters, emitted only for v2.9 roots (after the v2.8 ones)
PARSE_COUNTS_V29_EXTRA: tuple[str, ...] = (
    # inconsistent answers only: `continuation_text` AND `contradicted_before_marker` both true
    # (an excusable leaf quoted `after`); the excuse is refused. A consistent judge never sets
    # both, so this is NOT the number of excuses the prefix check blocks (see the last counter)
    "n_continuation_refused_prefix",
    "n_continuation_refused_missing",  # excuse refused: the field is missing (safe default)
    "n_contradicted_before_marker",  # incorrect segments the judge flagged (any tag)
    "n_prefix_clash_unlocated",  # flagged with a quote not located before the marker
    # an EXCUSABLE_LEAVES segment quoted `after` with `contradicted_before_marker` true, whatever
    # `continuation_text` says: every segment the prefix check keeps out of the excuse
    "n_excuse_blocked_prefix",
)
# plan 0029 Amendment 2: the continuation-verifier counters, emitted only for v2.9 roots with
# the verifier on (after the v2.9 ones). Candidates = ok, EXCUSABLE_LEAVES, contradiction
# `after`; `n_continuation_miss` = `n_verifier_yes` there. The judge-tag agreement counts cover
# candidates with an ok verdict ("judge" = the v2.9 tag rule: continuation_text and
# contradicted_before_marker false; it no longer decides anything)
PARSE_COUNTS_VERIFIER_EXTRA: tuple[str, ...] = (
    "n_excuse_candidates",
    "n_verifier_yes",
    "n_verifier_no",
    "n_verifier_failed",
    "n_verifier_pending",
    "n_judge_would_excuse",
    "n_judge_yes_verifier_yes",
    "n_judge_yes_verifier_no",
    "n_judge_no_verifier_yes",
    "n_judge_no_verifier_no",
)
SHORT_QUOTE_WORDS = 3


def parse_counts(version: str | None = None, verifier: str | None = None) -> tuple[str, ...]:
    """The parse counters of ``version`` (v2.7: ``PARSE_COUNTS`` exactly)."""
    return (*PARSE_COUNTS, *extra_parse_counts(version, verifier))


def extra_parse_counts(version: str | None = None, verifier: str | None = None) -> tuple[str, ...]:
    """The counters ``version`` adds after ``PARSE_COUNTS`` (v2.7: none; v2.8: the v2.8 ones;
    v2.9: + the prefix-check ones; v2.9 with a continuation ``verifier``: + the verifier ones)."""
    if not is_v28(version):
        return ()
    if not is_v29(version):
        return PARSE_COUNTS_V28_EXTRA
    extra = PARSE_COUNTS_VERIFIER_EXTRA if verifier else ()
    return (*PARSE_COUNTS_V28_EXTRA, *PARSE_COUNTS_V29_EXTRA, *extra)


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
    if s.raw_leaf not in tx.INCORRECT_LEAVES:
        s.continuation_text = False  # v2.8: an incorrect-leaf tag (never overlaps prediction)
        if s.v29:  # v2.9: incorrect-leaf tags too
            s.contradicted_before_marker, s.prefix_clash = False, None
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
            s.continuation_text = False  # plan 0029: an incorrect-leaf tag (explicit here)
            if s.v29:
                s.contradicted_before_marker, s.prefix_clash = False, None
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


def read_offset_chat(prompt: str, response: str, char: int) -> int:
    """Plan 0029 §1.1: the marker's offset in :func:`conversation_text` (main / story):
    ``len(prompt) + 1 + clamp(char, 0, len(response))`` (the v5c clamp)."""
    return len(prompt) + 1 + max(0, min(int(char), len(response)))


def parse(
    raw: dict[str, Any] | None,
    units: Sequence[Unit],
    conversation: str | None = None,
    *,
    read_offset: int | None = None,
    version: str | None = None,
    excuse_verdicts: Mapping[int, ExcuseVerdict] | None = None,
) -> tuple[list[Segment], dict[str, int]]:
    """Answer -> segments covering every unit, fail closed (see the module docstring). With
    ``conversation`` (production, calibration, consistency) the rev-17 evidence quotes are
    checked; ``None`` skips the check (unit tests of the other mechanics only).

    Plan 0029 (claimmap-v2.8, ``version`` >= v2.8; ``read_offset`` needs ``version``): segments
    carry ``continuation_text`` / ``contradiction_region`` and the v2.8 counters are added. With
    ``read_offset`` (the marker's offset in ``conversation``; single-call mode only, never
    anchored) every incorrect segment's quote is placed (``prefix`` iff it ends at or before
    the marker) and the §1.2 excuse runs last, after the rev 17 / 18 / 19 steps.

    Plan 0029 amendment (claimmap-v2.9, ``version`` >= v2.9): segments also carry
    ``contradicted_before_marker`` / ``prefix_clash`` / ``prefix_clash_region`` and the excuse
    is refused when the judge says the text before the marker rules the segment out (or the
    field is missing: the safe default is no excuse); a contradiction quote straddling the
    marker is ``prefix``.

    Plan 0029 Amendment 2 (``excuse_verdicts`` given; v2.9 with ``read_offset`` only): the
    continuation verifier decides the excuse, not the judge's tags. A **candidate** is an ``ok``
    segment with a leaf in ``EXCUSABLE_LEAVES`` and ``contradiction_region == "after"``; it gets
    ``excuse_verdict`` (from ``excuse_verdicts[segment index]``; a missing entry is
    ``pending``) and is excused iff the verdict is c1 & c2 & c3. A failed or pending verdict
    never excuses. The judge's ``continuation_text`` / ``contradicted_before_marker`` are kept
    and counted (the v2.9 counters, + judge-vs-verifier agreement). ``parse`` stays pure: the
    caller passes ``{}`` first to learn the candidates, then the verdicts.

    Guards: ``read_offset`` without ``version`` raises (no implicit v2.8), and so does an answer
    carrying ``contradicted_before_marker`` when ``version`` is not v2.9 or later, and
    ``excuse_verdicts`` without a v2.9 ``version`` and ``read_offset``."""
    if read_offset is not None and version is None:
        raise ValueError("read_offset needs an explicit version (claimmap-v2.8 or later)")
    v29 = version is not None and is_v29(version)
    if not v29 and _has_key(raw, "contradicted_before_marker"):
        raise ValueError(
            f"the answer carries contradicted_before_marker but version {version!r} is not "
            "claimmap-v2.9 or later"
        )
    cv = excuse_verdicts is not None
    if cv and (not v29 or read_offset is None):
        raise ValueError("excuse_verdicts need a claimmap-v2.9 version and read_offset")
    v28 = v29 or read_offset is not None or (version is not None and is_v28(version))
    if read_offset is not None and conversation is None:
        raise ValueError("read_offset needs the conversation")
    conv = Normalised(conversation) if conversation is not None else None
    c = dict.fromkeys(
        parse_counts(tx.V29 if v29 else tx.V28, "on" if cv else None) if v28 else PARSE_COUNTS, 0
    )
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
                continuation_text=v28 and _b(tsrc.get("continuation_text")),
                contradicted_before_marker=_opt_bool(tsrc.get("contradicted_before_marker"))
                if v29
                else None,
                prefix_clash=_quote(tsrc.get("prefix_clash")) if v29 else None,
                v28=v28,
                v29=v29,
                cv=cv,
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
        _remap_hedged(s, c)
    if read_offset is not None and conv is not None:
        for i, s in enumerate(segs):
            _place_contradiction(s, conv, read_offset, c, straddle_prefix=v29)
            if v29:
                _place_prefix_clash(s, conv, read_offset, c)
                if (
                    s.status == "ok"
                    and s.leaf in tx.EXCUSABLE_LEAVES
                    and s.contradiction_region == "after"
                    and s.contradicted_before_marker is True
                ):
                    c["n_excuse_blocked_prefix"] += 1
            if excuse_verdicts is not None:
                _verify_continuation(s, excuse_verdicts.get(i), c)
            else:
                _excuse_continuation(s, c)
    c["n_segments"] = len(segs)
    return segs, c


def _has_key(x: Any, key: str) -> bool:
    """``key`` in any unit entry or split of a judge answer (a malformed answer: ``False``)."""
    if not isinstance(x, dict) or not isinstance(x.get("units"), list):
        return False
    for e in x["units"]:
        if not isinstance(e, dict):
            continue
        splits = e.get("splits") if isinstance(e.get("splits"), list) else []
        if key in e or any(isinstance(p, dict) and key in p for p in splits):
            return True
    return False


def contradiction_region(
    q: str, conv: Normalised, read_offset: int, *, straddle_prefix: bool = False
) -> str | None:
    """Plan 0029 §1.1: ``prefix`` iff the earliest match of the quote ends at or before the
    marker, else ``after`` (a crossing quote is ``after``); ``None`` when it is not found.

    ``straddle_prefix`` (claimmap-v2.9 contradiction quotes, plan 0029 amendment): a crossing
    quote -- earliest match starting before the marker and ending after it -- is ``prefix``
    (fail closed: it is never excused)."""
    m = conv.find(q, 0)
    if m is None:
        return None
    if m[1] <= read_offset or (straddle_prefix and m[0] < read_offset):
        return "prefix"
    return "after"


def _place_contradiction(
    s: "Segment",
    conv: Normalised,
    read_offset: int,
    c: dict[str, int],
    *,
    straddle_prefix: bool = False,
) -> None:
    if s.status != "ok" or s.leaf not in tx.INCORRECT_LEAVES or s.contradicted_by is None:
        return
    region = contradiction_region(
        s.contradicted_by, conv, read_offset, straddle_prefix=straddle_prefix
    )
    s.contradiction_region = region
    if region is not None:
        c[f"n_contradiction_{region}"] += 1
        if len(s.contradicted_by.split()) <= SHORT_QUOTE_WORDS:
            c["n_short_quotes"] += 1


def _opt_bool(x: Any) -> bool | None:
    """A judge boolean, ``None`` when the field is missing or not a boolean."""
    return x if isinstance(x, bool) else None


def _place_prefix_clash(
    s: "Segment", conv: Normalised, read_offset: int, c: dict[str, int]
) -> None:
    """Plan 0029 amendment (v2.9): an ``ok`` incorrect segment the judge flagged
    ``contradicted_before_marker`` is counted, and its ``prefix_clash`` quote is placed
    (``prefix`` iff its earliest match ends at or before the marker). A missing quote, or one
    not located before the marker, is recorded (``n_prefix_clash_unlocated``); the boolean
    alone still refuses the excuse."""
    if s.status != "ok" or s.leaf not in tx.INCORRECT_LEAVES or not s.contradicted_before_marker:
        return
    c["n_contradicted_before_marker"] += 1
    if s.prefix_clash is not None:
        s.prefix_clash_region = contradiction_region(s.prefix_clash, conv, read_offset)
    if s.prefix_clash_region != "prefix":
        c["n_prefix_clash_unlocated"] += 1


def _excuse_continuation(s: "Segment", c: dict[str, int]) -> None:
    """Plan 0029 §1.2, last (after rev 17 quotes, rev 18 junk, rev 19 hedges): an ``ok``
    segment whose leaf is incorrect (``tx.EXCUSABLE_LEAVES``: any size of change), that the
    judge tagged ``continuation_text`` (a literal, on-topic continuation possible given the text
    up to the marker) and whose quoted contradiction is located ``after`` the read point (a
    prefix contradiction is never excused) becomes ``continuation_miss`` (``judge_leaf`` = the
    judge's leaf, ``raw_leaf`` unchanged). It is no longer a probe candidate.

    v2.9 (plan 0029 amendment): the excuse is also refused when the judge says the text before
    the marker already rules the segment out (``contradicted_before_marker``, whatever its
    ``prefix_clash`` quote locates to; reached only when ``continuation_text`` is also true, an
    inconsistent answer, counted ``n_continuation_refused_prefix``; every segment the check
    blocks is counted in :func:`parse` as ``n_excuse_blocked_prefix``) or did not give the
    field (``n_continuation_refused_missing``: the safe default is no excuse). A v2.9
    contradiction quote that straddles the marker is ``prefix`` (never excused)."""
    if _judge_would_excuse(s, c):
        _apply_excuse(s, c)


def _judge_would_excuse(s: "Segment", c: dict[str, int]) -> bool:
    """The judge-tag excuse rule (v2.8 / v2.9; see :func:`_excuse_continuation`), counting the
    v2.9 refusals (``n_continuation_refused_prefix`` / ``_missing``)."""
    if (
        s.status != "ok"
        or s.leaf not in tx.EXCUSABLE_LEAVES
        or not s.continuation_text
        or s.contradiction_region != "after"
    ):
        return False
    if s.v29 and s.contradicted_before_marker is not False:
        refused = s.contradicted_before_marker  # True, or None (missing field)
        c["n_continuation_refused_prefix" if refused else "n_continuation_refused_missing"] += 1
        return False
    return True


def is_excuse_candidate(s: "Segment") -> bool:
    """Plan 0029 Amendment 2: a segment the continuation verifier is asked about (``ok``, an
    excusable leaf, its contradiction located only after the marker; v2.9's straddle rule kept
    by :func:`contradiction_region`), whatever the judge's continuation tags say."""
    return s.status == "ok" and s.leaf in tx.EXCUSABLE_LEAVES and s.contradiction_region == "after"


def _verify_continuation(s: "Segment", v: ExcuseVerdict | None, c: dict[str, int]) -> None:
    """Plan 0029 Amendment 2: a candidate is excused iff its verifier verdict is c1 & c2 & c3; a
    missing (``pending``) or failed verdict never excuses. The judge's tag rule is evaluated
    for reporting only (its refusal counters, ``n_judge_would_excuse``, agreement counts)."""
    if not is_excuse_candidate(s):
        return
    v = v if v is not None else ExcuseVerdict("pending")
    judge = _judge_would_excuse(s, c)
    c["n_excuse_candidates"] += 1
    c["n_judge_would_excuse"] += judge
    s.excuse_verdict = v.label
    c[f"n_verifier_{v.label}"] += 1
    if v.status == "ok":
        s.excuse_checks = {"c1": v.c1, "c2": v.c2, "c3": v.c3}
        j, y = "yes" if judge else "no", "yes" if v.yes else "no"
        c[f"n_judge_{j}_verifier_{y}"] += 1
    if v.yes:
        _apply_excuse(s, c)


def _apply_excuse(s: "Segment", c: dict[str, int]) -> None:
    if s.raw_leaf in tx.PROBE_CANDIDATE_LEAVES:  # as rev 19
        c["n_probe_candidates"] -= 1
        if s.probe_target is None:
            c["n_missing_probe_targets"] -= 1
    s.judge_leaf = s.leaf
    s.leaf = tx.CONTINUATION_MISS
    s.probe_target = None
    c["n_continuation_miss"] += 1


def _remap_hedged(s: "Segment", c: dict[str, int]) -> None:
    """Rev 19, after parsing: a ``hedged`` segment (a list or disjunction of options) is scored
    ``unverifiable`` with ``same_slot_alternative = hedged = true``, whatever the judge's leaf
    (kept in ``raw_leaf``; a quote-check re-label keeps it in ``judge_leaf``). A single wrong
    entity is not hedged and stays a substitution."""
    if not s.hedged or s.status != "ok":
        return
    if s.leaf == "unverifiable":  # already unverifiable: tag only, not counted as a remap
        s.same_slot_alternative = True
        return
    if s.raw_leaf in tx.PROBE_CANDIDATE_LEAVES:  # no longer a probe candidate
        c["n_probe_candidates"] -= 1
        if s.probe_target is None:
            c["n_missing_probe_targets"] -= 1
    c["n_hedged_remapped"] += 1
    s.leaf = "unverifiable"
    s.continuation_text = False  # plan 0029: an incorrect-leaf tag; no longer incorrect
    s.same_slot_alternative = True
    s.cruxy = False
    s.probe_target = None
    s.reason = s.source = None


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
