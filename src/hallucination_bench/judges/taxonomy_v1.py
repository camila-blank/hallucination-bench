"""The taxonomy classifier (``taxonomy-v1``): labels already-found hallucinated spans.

It never decides whether a span is a hallucination and never adds, removes or re-judges spans.
The system prompt is rendered from :mod:`hallucination_bench.taxonomy`. It sees the prompt, the
marked response, the specific-class readouts and a numbered span list ``S1..Sn`` (readout index +
grounding label); it never sees the earlier stages' ``why`` text.

Parsing fails closed (see :func:`parse`); enum values are validated in code (:func:`validate`),
and an invalid answer is cached as a failure and re-queued, as ``validate`` does in wsbench.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from hallucination_bench import taxonomy as tx
from hallucination_bench.judges.v5c import READ_MARK
from hallucination_bench.llm import schema_block

PROMPT_VERSION = "taxonomy-v1"
GROUNDING_ORDER: tuple[str, ...] = ("contradicted", "disputed", "unsupported")

SYSTEM_INTRO = (
    "You classify hallucinated spans in the readouts of an interpretability lens. The lens reads "
    "a language model's internal activation while the model writes a response, and its readouts "
    "describe what the model is representing at that point. Earlier audits have ALREADY decided "
    "that every span you are given is a hallucination about this conversation, and how it is "
    "grounded (contradicted, disputed or unsupported). That is settled: do not re-judge it, and "
    "do not add, drop or merge spans. Your only job is to label each span with its type, "
    "target, severity and, for unsupported spans, its unsupported_kind, using the taxonomy "
    "below. Output only the JSON verdict."
)

SYSTEM = SYSTEM_INTRO + "\n\n" + tx.render()

USER_TEMPLATE = (
    "A language model was asked:\n\n<prompt>\n{prompt}\n</prompt>\n\n"
    "and wrote the response below. The lens read the model's activation at the token immediately "
    "before the marker {mark}; everything before it is what the model had already written, "
    "everything after it is what it went on to write. The WHOLE response is the context.\n\n"
    "<response>\n{marked_response}\n</response>\n\n"
    "The lens readouts of that activation:\n\n{readouts_block}\n\n"
    "Spans to label ({n}), each quoted verbatim from the readout it names, with its grounding:\n\n"
    "{spans_block}\n\n"
    "For EACH span report: span_id (exactly as given), type, target, severity (1, 2 or 3), "
    "unsupported_kind (one of prediction, inference, fabricated, world_knowledge for an "
    "unsupported span; null for a contradicted or disputed span), and why (one short clause). "
    "Label every span exactly once and no others. Then give a one-sentence rationale."
)

SCHEMA: dict[str, Any] = schema_block(
    "taxonomy_v1",
    {
        "spans": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["span_id", "type", "target", "severity", "unsupported_kind", "why"],
                "properties": {
                    "span_id": {"type": "string"},
                    "type": {"type": "string", "enum": list(tx.TYPES)},
                    "target": {"type": "string", "enum": list(tx.TARGETS)},
                    "severity": {"type": "integer", "enum": list(tx.SEVERITIES)},
                    "unsupported_kind": {
                        "anyOf": [
                            {"type": "string", "enum": list(tx.UNSUPPORTED_KINDS)},
                            {"type": "null"},
                        ]
                    },
                    "why": {"type": "string"},
                },
            },
        },
        "rationale": {"type": "string"},
    },
    ["spans", "rationale"],
)

PROMPTS: dict[str, str] = {"SYSTEM": SYSTEM, "USER_TEMPLATE": USER_TEMPLATE}
_PLACEHOLDER = re.compile(r"\{(prompt|mark|marked_response|readouts_block|n|spans_block)\}")


@dataclass(frozen=True)
class SpanIn:
    """One span to classify: its readout index, verbatim text and grounding label."""

    readout_idx: int
    text: str
    grounding: str


@dataclass
class SpanLabel:
    span_id: str
    readout_idx: int
    text: str
    grounding: str
    type: str = tx.UNKNOWN
    target: str = tx.UNKNOWN
    severity: int | None = None
    unsupported_kind: str | None = None
    why: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def span_id(i: int) -> str:
    return f"S{i + 1}"


def readouts_block(readouts: Sequence[tuple[int, str]]) -> str:
    return "\n\n".join(f'<readout idx="{i}">\n{s}\n</readout>' for i, s in readouts)


def spans_block(spans: Sequence[SpanIn]) -> str:
    return "\n".join(
        f"{span_id(i)} | readout {s.readout_idx} | grounding: {s.grounding} | "
        f"{json.dumps(s.text, ensure_ascii=False)}"
        for i, s in enumerate(spans)
    )


def render_prompt(
    prompt: str,
    marked_response: str,
    readouts: Sequence[tuple[int, str]],
    spans: Sequence[SpanIn],
) -> tuple[str, str]:
    """(system, user) for one classify call. ``readouts`` = ``(idx, text)`` of the cell's
    specific-class readouts; ``marked_response`` already carries the ``⟦READ⟧`` marker."""
    if not spans:
        raise ValueError("render_prompt needs at least one span")
    for s in spans:
        if s.grounding not in GROUNDING_ORDER:
            raise ValueError(f"bad grounding {s.grounding!r}")
    values = {
        "prompt": prompt,
        "mark": READ_MARK,
        "marked_response": marked_response,
        "readouts_block": readouts_block(readouts),
        "n": str(len(spans)),
        "spans_block": spans_block(spans),
    }
    # one pass: braces inside the substituted text are never re-expanded
    user = _PLACEHOLDER.sub(lambda m: values[m.group(1)], USER_TEMPLATE)
    return SYSTEM, user


# ---------------------------------------------------------------- validation / parsing


def _norm_id(x: Any, n: int) -> str | None:
    s = str(x).strip().upper() if isinstance(x, str | int) and not isinstance(x, bool) else ""
    if s.isdigit():
        s = f"S{s}"
    if s.startswith("S") and s[1:].isdigit() and 1 <= int(s[1:]) <= n:
        return f"S{int(s[1:])}"
    return None


def _valid_severity(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v in tx.SEVERITIES


def validate(raw: dict[str, Any] | None, spans: Sequence[SpanIn]) -> bool:
    """True when the answer is well-shaped and every label on a known span id is a valid enum
    value. Missing / duplicated / extra ids do NOT invalidate (they fail closed in parse)."""
    if not isinstance(raw, dict) or not isinstance(raw.get("spans"), list):
        return False
    for e in raw["spans"]:
        if not isinstance(e, dict):
            return False
        if _norm_id(e.get("span_id"), len(spans)) is None:
            continue  # an extra id: ignored and counted by parse
        if e.get("type") not in tx.TYPES or e.get("target") not in tx.TARGETS:
            return False
        if not _valid_severity(e.get("severity")):
            return False
        uk = e.get("unsupported_kind")
        if uk is not None and uk not in tx.UNSUPPORTED_KINDS:
            return False
    return True


PARSE_COUNTS: tuple[str, ...] = (
    "n_unclassified",
    "n_missing_ids",
    "n_duplicate_ids",
    "n_extra_ids",
    "n_bad_labels",
    "n_kind_ignored",
)


def parse(
    raw: dict[str, Any] | None, spans: Sequence[SpanIn]
) -> tuple[list[SpanLabel], dict[str, int]]:
    """Answer -> one label per input span, fail closed:

    * no answer, or a span id missing or duplicated -> that span is ``type = unknown`` (all its
      labels unknown), counted in ``n_unclassified``;
    * an extra id is ignored and counted (``n_extra_ids``);
    * an ``unsupported_kind`` on a contradicted / disputed span is ignored (set to null);
    * a missing ``unsupported_kind`` on an unsupported span becomes ``unknown``;
    * a bad enum value (only possible for an answer that failed :func:`validate`, read at report
      time) makes that label ``unknown`` (``n_bad_labels``)."""
    counts = dict.fromkeys(PARSE_COUNTS, 0)
    labels = [
        SpanLabel(
            span_id(i),
            s.readout_idx,
            s.text,
            s.grounding,
            unsupported_kind=tx.UNKNOWN if s.grounding == "unsupported" else None,
        )
        for i, s in enumerate(spans)
    ]
    entries: dict[str, list[dict[str, Any]]] = {}
    if isinstance(raw, dict) and isinstance(raw.get("spans"), list):
        for e in raw["spans"]:
            if not isinstance(e, dict):
                continue
            sid = _norm_id(e.get("span_id"), len(spans))
            if sid is None:
                counts["n_extra_ids"] += 1
                continue
            entries.setdefault(sid, []).append(e)
    by_id = {lab.span_id: lab for lab in labels}
    for sid, lab in by_id.items():
        got = entries.get(sid, [])
        if len(got) != 1:
            if raw is not None:
                counts["n_missing_ids" if not got else "n_duplicate_ids"] += 1
            continue
        e = got[0]
        lab.why = str(e.get("why") or "")
        t, tg, sev, uk = (
            e.get("type"),
            e.get("target"),
            e.get("severity"),
            e.get("unsupported_kind"),
        )
        if t in tx.TYPES:
            lab.type = t
        else:
            counts["n_bad_labels"] += 1
        if tg in tx.TARGETS:
            lab.target = tg
        else:
            counts["n_bad_labels"] += 1
        if _valid_severity(sev):
            lab.severity = sev
        else:
            counts["n_bad_labels"] += 1
        if lab.grounding == "unsupported":
            if uk in tx.UNSUPPORTED_KINDS:
                lab.unsupported_kind = uk
            elif uk is not None:
                counts["n_bad_labels"] += 1
        elif uk is not None:
            counts["n_kind_ignored"] += 1
    counts["n_unclassified"] = sum(lab.type == tx.UNKNOWN for lab in labels)
    return labels, counts


def unknown_labels(spans: Sequence[SpanIn]) -> list[SpanLabel]:
    """Every span unknown: a failed, missing or stale (fingerprint-mismatched) classify call."""
    labels, _ = parse(None, spans)
    return labels
