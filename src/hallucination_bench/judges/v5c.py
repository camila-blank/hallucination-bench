"""The v5c judge stages, verbatim: span detection (``v5c-chat``) and claim verify
(``v5c-chat-verify-v1``).

Ported from WorkspaceBench ``src/wsbench/evals/hallucination/{prompts,judge}.py`` @ 92d763e
(MIT): prompt literals, schemas, rendering, ``parse_verdict``, ``tally_claims`` and
``fully_judged`` are copied without change. The span literals are asserted at import to equal the
frozen ``data/wsbench_chat/judge_prompt_v5c_chat.json``; ``tests/test_v5c_frozen.py`` also checks
the verify literals against the wsbench-pub source and the data README.

New here (not in the source): :func:`claim_lists`, a parallel re-implementation of the
``tally_claims`` loop that returns the verified quotes themselves, disjoint by status, for the
taxonomy classifier; it is checked count-for-count against ``tally_claims`` at runtime and in tests.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from hallucination_bench import DATA
from hallucination_bench.llm import schema_block
from hallucination_bench.readouts import normalize_ws, verify_quote

PROMPT_VERSION = "v5c-chat"
READ_MARK = "⟦READ⟧"
FROZEN_PATH = DATA / "wsbench_chat" / "judge_prompt_v5c_chat.json"
FROZEN: dict[str, Any] = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))

JUDGE_SYSTEM = (
    "You audit an interpretability lens that reads a language model's internal activation "
    "while the model writes a response. You are shown the model's full response and the "
    "lens's raw readouts of one activation inside it. Your job is to point at the parts of "
    "a readout that are WRONG about this response, and separately at the parts that are "
    "JUNK. You quote the readout's own words and never paraphrase. Output only the JSON "
    "verdict."
)

JUDGE_USER = (
    "A language model was asked:\n\n<prompt>\n{prompt}\n</prompt>\n\nand wrote the "
    "response below. The lens read the model's activation at the token immediately before "
    "the marker {mark}; everything before it is what the model had already written, "
    "everything after it is what it went on to write. The WHOLE response is the context — "
    "a readout may legitimately refer to any part of it.\n\n<response>\n{marked_response}\n"
    "</response>\n{facts_block}\nBelow are {k} independent readouts of that one "
    'activation.\n\n{samples_block}\n\nFor EACH readout, report:\n\n- kind: "specific" if '
    'it says anything concrete; "generic" if it only says generic things ("a narrative '
    'about a character", "an account of a historical event"); "empty" for noise, '
    "fragments, formatting, a bare restatement of the prompt, or a garbled string that "
    "asserts nothing.\n- spans: the parts of THIS readout that are WRONG or OFF-TOPIC, "
    "each copied character-for-character from the readout (never your own words, never a "
    'span from the response), each with a type:\n\n  type "wrong" — the span is on topic '
    "(about THIS subject, THIS response, or what it says next) and the response or the "
    "facts show it is wrong:\n    - it contradicts the response or the facts: a wrong "
    "date, number or quantity; a wrong name, place or label; the wrong person doing "
    'something; a wrong cause, outcome or detail ("October 24" where the response says '
    'October 29; "over 700 died" where it says over 140; Johnson where it says Kennedy; '
    "uracil among the bases of DNA);\n    - it quotes or names something in the response "
    "and gets the words wrong: a heading, section title, list item or quotation the "
    'response does not have ("the \'Quick Facts\' section" where the heading is "Basic '
    'Facts"). Reordering or reformatting the same words is NOT wrong;\n    - it misstates '
    "what the response has ALREADY written before the marker: what a passage says, how the "
    "response is structured so far, or what it has already claimed;\n    - it states "
    "something about this subject that conflicts with the response (an event, outcome or "
    'detail the response rules out).\n\n  type "off_topic" — the span is JUNK: garbled or '
    "broken tokens, stray code or programming terms, or content with no connection at all "
    "to this conversation (a Mars probe in an answer about English idioms; TimeSpan values "
    "in a recipe). This is uncommon. Coherent content that connects to the conversation is "
    "NOT off_topic, even when it brings up a new topic, a side example or a different "
    "scenario (a hostile manager or a breakup in an answer about kindness to family): if "
    "it is a reasonable direction for the response, leave it unmarked; if it says "
    'something about this conversation that the text rules out, it is "wrong". Content '
    "from this subject or a closely related one is never off_topic. Off-topic spans are "
    "reported but do not count.\n\nDo NOT mark (neither type):\n  - a prediction of what "
    "comes next that is a reasonable continuation of the text before the marker, even when "
    "the response actually went a different way. The response after the marker is ONE "
    "sample of what the model could have written, not the only right answer: a readout "
    'that expects the model to call its "better half" its codebase, where the response '
    "went on to say the engineers who built it, is consistent. Judge a prediction against "
    'the text BEFORE the marker, and mark it "wrong" only when that text rules it out;\n  '
    "- anything the response simply does not mention: absence is not wrongness. Mark "
    '"wrong" only when the response or the facts rule the span out;\n  - content that is '
    "right, compatible, or a reasonable inference; a generic statement;\n  - hedging, "
    "repetition, formatting, or a fragment that breaks off.\nWhen a span could reasonably "
    "be right, leave it.\n\nReport, as JSON: samples = one entry per readout in order, "
    "each with idx, kind, spans (list of {{text, type, why}} where `text` is the verbatim "
    'readout span, `type` is "wrong" or "off_topic", and `why` is one short clause), and a '
    "one-sentence rationale."
)

JUDGE_SCHEMA: dict[str, Any] = schema_block(
    "hallucination_ontopic_spans",
    {
        "samples": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["idx", "kind", "spans", "rationale"],
                "properties": {
                    "idx": {"type": "integer"},
                    "kind": {"type": "string"},
                    "spans": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["text", "type", "why"],
                            "properties": {
                                "text": {"type": "string"},
                                "type": {"type": "string", "enum": ["wrong", "off_topic"]},
                                "why": {"type": "string"},
                            },
                        },
                    },
                    "rationale": {"type": "string"},
                },
            },
        }
    },
    ["samples"],
)

# ------------------------------------------------------------------ stage 2: claim verification

VERIFY_PROMPT_VERSION = "v5c-chat-verify-v1"
VERIFY_STATUSES: tuple[str, ...] = ("true", "unverifiable", "disputed", "off_topic")

VERIFY_SYSTEM = (
    "You audit an interpretability lens that reads a language model's internal activation while "
    "the model writes a response. You are shown the model's full response and the lens's raw "
    "readouts of one activation inside it. Some spans of each readout have ALREADY been "
    "established as false by a separate audit; those are given to you and are not yours to "
    "re-judge. Your job is to list every OTHER specific claim each readout makes about this "
    "conversation and say whether the response settles it as true, or cannot settle it. You "
    "quote the readout's own words for every claim and never paraphrase. Output only the JSON "
    "verdict."
)

VERIFY_USER = (
    "A language model was asked:\n\n<prompt>\n{prompt}\n</prompt>\n\n"
    "and wrote the response below. The lens read the model's activation at the token immediately "
    "before the marker {mark}; everything before it is what the model had already written, "
    "everything after it is what it went on to write. The WHOLE response is the context — a "
    "readout may legitimately refer to any part of it.\n\n"
    "<response>\n{marked_response}\n</response>\n\n"
    "Below are {k} independent readouts of that one activation. Under each readout is the list "
    "of its spans already established as FALSE (possibly empty).\n\n{samples_block}\n\n"
    "For EACH readout, report:\n\n"
    '- kind: "specific" if it says anything concrete; "generic" if it only says generic things '
    '("a narrative about a character", "an account of a historical event"); "empty" for noise, '
    "fragments, formatting, a bare restatement of the prompt, or a garbled string that asserts "
    "nothing.\n"
    "- claims: EVERY separate specific claim the readout makes about this conversation that is "
    "NOT one of its already-established false spans. A claim is a specific assertion the "
    "readout commits to — a fact, entity, quantity, name, place, quotation, heading, structure, "
    "intent, or a prediction of what the response says next. Generic phrases, task "
    "descriptions, hedging, repetition, formatting and broken fragments are NOT claims; a "
    "readout may have zero claims. Do not list the established-false spans again. For each:\n\n"
    "  - quote: the readout's own words for this claim, copied character-for-character from THIS "
    "readout (never your own words, never a span of the response). Quote the shortest span that "
    "carries the claim.\n"
    "  - status, exactly one of:\n"
    '    "true" — the response states this claim or clearly entails it. A prediction of what '
    "comes next counts as true when the response after the marker does say it.\n"
    '    "unverifiable" — the response neither states it nor rules it out: an unstated detail, a '
    "plausible inference, a claim about what the model is thinking or intending, or a "
    "prediction the response does not go on to make but that nothing in the text excludes. "
    "Absence is not confirmation: if the response simply does not mention it, it is "
    "unverifiable, not true.\n"
    '    "disputed" — you believe the response actually RULES THIS OUT, but it was not in the '
    "established-false list. Use this rather than inventing a false verdict; it is reported "
    "separately.\n"
    '    "off_topic" — junk: garbled or broken tokens, stray code or programming terms, or '
    "content with no connection at all to this conversation.\n"
    "  - why: one short clause.\n\n"
    "Rules:\n"
    "  - Use ONLY this conversation to settle a claim, never world knowledge. A claim that is "
    "true in the world but unmentioned here is unverifiable.\n"
    '  - "true" requires the response to actually say or entail it. Compatible is not the same '
    "as stated: a claim that merely fits the response is unverifiable.\n\n"
    "Report, as JSON: samples = one entry per readout in order, each with idx, kind, claims "
    "(list of {{quote, status, why}}), and a one-sentence rationale."
)

VERIFY_SCHEMA: dict[str, Any] = schema_block(
    "hallucination_chat_verify",
    {
        "samples": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["idx", "kind", "claims", "rationale"],
                "properties": {
                    "idx": {"type": "integer"},
                    "kind": {"type": "string"},
                    "claims": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["quote", "status", "why"],
                            "properties": {
                                "quote": {"type": "string"},
                                "status": {"type": "string", "enum": list(VERIFY_STATUSES)},
                                "why": {"type": "string"},
                            },
                        },
                    },
                    "rationale": {"type": "string"},
                },
            },
        }
    },
    ["samples"],
)

PROMPTS: dict[str, str] = {
    "JUDGE_SYSTEM": JUDGE_SYSTEM,
    "JUDGE_USER": JUDGE_USER,
    "VERIFY_SYSTEM": VERIFY_SYSTEM,
    "VERIFY_USER": VERIFY_USER,
}


def render_marked_response(response: str, char: int) -> str:
    """The response with :data:`READ_MARK` inserted at ``char`` — the frozen offset just after
    the read token (``sites[].char`` in the bank)."""
    char = max(0, min(int(char), len(response)))
    return response[:char] + READ_MARK + response[char:]


def samples_block(samples: Sequence[str]) -> str:
    return "\n\n".join(f'<readout idx="{i}">\n{s}\n</readout>' for i, s in enumerate(samples))


def judge_prompt(
    item: dict[str, Any], site: dict[str, Any], samples: Sequence[str]
) -> tuple[str, str]:
    """(system, user) for one (item, layer, site) cell. No facts block: chat items have none."""
    return (
        JUDGE_SYSTEM,
        JUDGE_USER.format(
            prompt=item["prompt"],
            mark=READ_MARK,
            marked_response=render_marked_response(item["response"], int(site["char"])),
            facts_block="",
            k=len(samples),
            samples_block=samples_block(samples),
        ),
    )


def verify_samples_block(samples: Sequence[str], wrong: Sequence[Sequence[str]]) -> str:
    """Each readout block followed by its established-false spans (one ``json.dumps`` line each,
    or ``(none)``), exactly as the source pass rendered them."""
    parts = []
    for i, (s, w) in enumerate(zip(samples, wrong, strict=True)):
        falses = "\n".join(f"  - {json.dumps(x, ensure_ascii=False)}" for x in w) or "  (none)"
        parts.append(
            f'<readout idx="{i}">\n{s}\n</readout>\n'
            f"Established FALSE spans of readout {i}:\n{falses}"
        )
    return "\n\n".join(parts)


def verify_prompt(
    item: dict[str, Any],
    site: dict[str, Any],
    samples: Sequence[str],
    wrong: Sequence[Sequence[str]],
) -> tuple[str, str]:
    """(system, user) for the claim-verification call of one cell; ``wrong[i]`` = the span
    judge's verified wrong spans of readout ``i`` (given to the judge as established false)."""
    return (
        VERIFY_SYSTEM,
        VERIFY_USER.format(
            prompt=item["prompt"],
            mark=READ_MARK,
            marked_response=render_marked_response(item["response"], int(site["char"])),
            k=len(samples),
            samples_block=verify_samples_block(samples, wrong),
        ),
    )


# ------------------------------------------------------------------ verdicts (judge.py)

HAL_K = 3  # readouts per cell (a maximum); k = 1 for a summarised top-k token lens
SITE_KINDS: tuple[str, ...] = ("sentence", "clause", "quote", "newline", "markup", "eos")
READOUT_KINDS: tuple[str, ...] = ("specific", "generic", "empty")
SPECIFIC_CLASSES = frozenset({"hallucinated", "off_topic", "consistent"})


@dataclass
class ReadoutVerdict:
    idx: int
    kind: str
    wrong: list[str]  # verified wrong spans (verbatim readout text)
    off_topic: list[str]  # verified junk spans
    n_unverified: int  # spans dropped because they are not verbatim in this readout
    cls: str  # hallucinated | off_topic | generic | consistent | unjudged
    revoked: bool = False  # every wrong span is also a verified wrong span of another readout
    claims: dict[str, int] | None = None  # stage-2 tally (see tally_claims); None = not verified


def _readout_class(kind: str, wrong: Sequence[str], off_topic: Sequence[str]) -> str:
    if wrong:
        return "hallucinated"  # whatever the kind: a made-up claim is never "generic"
    if kind in ("generic", "empty"):
        return "generic"  # junk on a non-specific readout does not enter the denominator
    return "off_topic" if off_topic else "consistent"


def parse_verdict(
    raw: dict[str, Any] | None, samples: Sequence[str]
) -> list[ReadoutVerdict] | None:
    """Judge output -> one verdict per readout, or None for an unjudged cell (never zeros).

    Every span must be verbatim in ITS OWN readout (`verify_quote` normalisation) or it is
    dropped and counted; an unknown span type counts as wrong (fail closed)."""
    if not isinstance(raw, dict) or not isinstance(raw.get("samples"), list):
        return None
    by_idx: dict[int, dict[str, Any]] = {}
    for s in raw["samples"]:
        if not isinstance(s, dict):
            continue
        idx = s.get("idx")
        if isinstance(idx, int | str) and str(idx).strip().lstrip("-").isdigit():
            by_idx[int(idx)] = s
    out: list[ReadoutVerdict] = []
    for i, readout in enumerate(samples):
        s = by_idx.get(i)
        if s is None:
            out.append(ReadoutVerdict(i, "specific", [], [], 0, "unjudged"))
            continue
        kind = str(s.get("kind") or "").strip().lower()
        kind = kind if kind in READOUT_KINDS else "specific"
        wrong: list[str] = []
        off: list[str] = []
        n_unverified = 0
        for sp in s.get("spans") or []:
            if not isinstance(sp, dict):
                continue
            text = str(sp.get("text") or "")
            if not verify_quote(text, [readout]):
                n_unverified += 1
                continue
            is_off = str(sp.get("type") or "").strip().lower() == "off_topic"
            (off if is_off else wrong).append(text)
        out.append(
            ReadoutVerdict(i, kind, wrong, off, n_unverified, _readout_class(kind, wrong, off))
        )
    for v in out:
        if v.cls != "hallucinated":
            continue
        others = {normalize_ws(t) for o in out if o.idx != v.idx for t in o.wrong}
        v.revoked = all(normalize_ws(t) in others for t in v.wrong)
    return out


CLAIM_KEYS: tuple[str, ...] = (
    "false",
    "true",
    "unverifiable",
    "disputed",
    "off_topic",
    "n_unverified",
)


def tally_claims(
    raw: dict[str, Any] | None, samples: Sequence[str], wrong: Sequence[Sequence[str]]
) -> list[dict[str, int]] | None:
    """Stage-2 output -> one tally per readout, or None for an unjudged cell (re-queued).

    Port of the source pass's ``tally``: ``false`` = the established wrong spans given to the
    judge; a quote must be verbatim in ITS OWN readout (``verify_quote``) or it is dropped and
    counted in ``n_unverified``; a verified quote overlapping an established-false span is
    skipped (already false, never double counted); ``disputed`` counts as ``disputed`` AND
    ``unverifiable``; any other or unknown status is ``unverifiable`` (never inflates ``true``).
    Every readout must be present with a ``claims`` list, else the whole cell is unjudged."""
    if not isinstance(raw, dict) or not isinstance(raw.get("samples"), list):
        return None
    by_idx: dict[int, dict[str, Any]] = {}
    for s in raw["samples"]:
        if isinstance(s, dict) and isinstance(s.get("idx"), int) and not isinstance(s["idx"], bool):
            by_idx[s["idx"]] = s
    if any(
        i not in by_idx or not isinstance(by_idx[i].get("claims"), list)
        for i in range(len(samples))
    ):
        return None
    out: list[dict[str, int]] = []
    for i, readout in enumerate(samples):
        w = [normalize_ws(x) for x in wrong[i]]
        c = dict.fromkeys(CLAIM_KEYS, 0)
        c["false"] = len(wrong[i])
        for cl in by_idx[i]["claims"]:
            if not isinstance(cl, dict):
                continue
            q = str(cl.get("quote") or "")
            if not q or not verify_quote(q, [readout]):
                c["n_unverified"] += 1
                continue
            nq = normalize_ws(q)
            if any(x in nq or nq in x for x in w):
                continue  # already counted as false
            st = str(cl.get("status") or "").strip().lower()
            if st == "true":
                c["true"] += 1
            elif st == "off_topic":
                c["off_topic"] += 1
            elif st == "disputed":
                c["disputed"] += 1
                c["unverifiable"] += 1
            else:  # unverifiable, or anything unexpected -- never inflates true
                c["unverifiable"] += 1
        out.append(c)
    return out


def fully_judged(raw: dict[str, Any] | None, samples: Sequence[str]) -> bool:
    """A verdict counts only if it judged EVERY readout (a partly judged cell is re-judged, never
    scored partially) — the ``validate`` hook of :func:`run_calls`."""
    v = parse_verdict(raw, samples)
    return v is not None and all(x.cls != "unjudged" for x in v)


# The span-stage literals must equal the frozen JSON byte for byte (import fails otherwise).
assert FROZEN["system"] == JUDGE_SYSTEM, "JUDGE_SYSTEM drifted from the frozen JSON"
assert FROZEN["user"] == JUDGE_USER, "JUDGE_USER drifted from the frozen JSON"
assert FROZEN["schema"] == JUDGE_SCHEMA, "JUDGE_SCHEMA drifted from the frozen JSON"
assert FROZEN["version"] == PROMPT_VERSION, "PROMPT_VERSION drifted from the frozen JSON"


# ------------------------------------------------------------------ verified quotes (new)

STATUS_LISTS: tuple[str, ...] = ("true", "unsupported", "disputed", "off_topic")


@dataclass
class ClaimLists:
    """One readout's verified verify-stage quotes, disjoint by status. ``unsupported`` holds the
    verify ``unverifiable`` claims (and unknown statuses); ``disputed`` holds ``disputed`` claims
    only (the frozen tally counts them as unverifiable too)."""

    false: int = 0
    true: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    disputed: list[str] = field(default_factory=list)
    off_topic: list[str] = field(default_factory=list)
    n_unverified: int = 0

    def as_tally(self) -> dict[str, int]:
        """The frozen ``tally_claims`` view of these lists."""
        return {
            "false": self.false,
            "true": len(self.true),
            "unverifiable": len(self.unsupported) + len(self.disputed),
            "disputed": len(self.disputed),
            "off_topic": len(self.off_topic),
            "n_unverified": self.n_unverified,
        }


def claim_lists(
    raw: dict[str, Any] | None, samples: Sequence[str], wrong: Sequence[Sequence[str]]
) -> list[ClaimLists] | None:
    """A parallel re-implementation of the :func:`tally_claims` loop that keeps the quotes.

    Same checks in the same order: the answer must carry every readout with a ``claims`` list
    (else ``None``); a quote must be verbatim in ITS OWN readout (``verify_quote``, incl. the
    under-3-character rule) or it is dropped and counted; a quote overlapping a ``normalize_ws``'d
    established-false span in either direction is skipped; ``true`` / ``off_topic`` /
    ``disputed`` go to their own lists and anything else (``unverifiable`` or an unknown status)
    to ``unsupported``."""
    if not isinstance(raw, dict) or not isinstance(raw.get("samples"), list):
        return None
    by_idx: dict[int, dict[str, Any]] = {}
    for s in raw["samples"]:
        if isinstance(s, dict) and isinstance(s.get("idx"), int) and not isinstance(s["idx"], bool):
            by_idx[s["idx"]] = s
    if any(
        i not in by_idx or not isinstance(by_idx[i].get("claims"), list)
        for i in range(len(samples))
    ):
        return None
    out: list[ClaimLists] = []
    for i, readout in enumerate(samples):
        w = [normalize_ws(x) for x in wrong[i]]
        c = ClaimLists(false=len(wrong[i]))
        for cl in by_idx[i]["claims"]:
            if not isinstance(cl, dict):
                continue
            q = str(cl.get("quote") or "")
            if not q or not verify_quote(q, [readout]):
                c.n_unverified += 1
                continue
            nq = normalize_ws(q)
            if any(x in nq or nq in x for x in w):
                continue  # already counted as false
            st = str(cl.get("status") or "").strip().lower()
            if st == "true":
                c.true.append(q)
            elif st == "off_topic":
                c.off_topic.append(q)
            elif st == "disputed":
                c.disputed.append(q)
            else:  # unverifiable, or anything unexpected -- never inflates true
                c.unsupported.append(q)
        out.append(c)
    return out


def checked_claim_lists(
    raw: dict[str, Any] | None, samples: Sequence[str], wrong: Sequence[Sequence[str]]
) -> list[ClaimLists] | None:
    """:func:`claim_lists`, asserted count-for-count against the frozen :func:`tally_claims`."""
    lists = claim_lists(raw, samples, wrong)
    frozen = tally_claims(raw, samples, wrong)
    if (lists is None) != (frozen is None):
        raise AssertionError("claim_lists and tally_claims disagree on validity")
    if lists is not None and frozen is not None:
        for cl, t in zip(lists, frozen, strict=True):
            if cl.as_tally() != t:
                raise AssertionError(f"claim_lists {cl.as_tally()} != tally_claims {t}")
    return lists


def _overlaps(nq: str, spans: Sequence[str]) -> bool:
    return any(x in nq or nq in x for x in spans)


def classify_quotes(
    cl: ClaimLists, off_topic_spans: Sequence[str]
) -> tuple[list[str], list[str], int]:
    """The quotes of one readout that go to the classifier: ``(unsupported, disputed,
    n_offtopic_overlap)``. Quotes are deduplicated within the readout (``normalize_ws``); a quote
    overlapping one of the readout's v5c ``off_topic`` spans (either direction) is dropped and
    counted; a quote listed as both ``disputed`` and ``unverifiable`` stays ``disputed`` only."""
    off = [normalize_ws(x) for x in off_topic_spans]
    n_off = 0

    def clean(quotes: Sequence[str]) -> list[str]:
        nonlocal n_off
        seen: set[str] = set()
        kept: list[str] = []
        for q in quotes:
            nq = normalize_ws(q)
            if nq in seen:
                continue
            seen.add(nq)
            if _overlaps(nq, off):
                n_off += 1
                continue
            kept.append(q)
        return kept

    disputed = clean(cl.disputed)
    dset = {normalize_ws(q) for q in cl.disputed}
    unsupported = clean([q for q in cl.unsupported if normalize_ws(q) not in dset])
    return unsupported, disputed, n_off
