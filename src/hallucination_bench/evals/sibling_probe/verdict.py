"""Plan 0004: the post-probe verdict judge (``sibverdict-v1``), depth / rule labels, categories.

Per claim unit (``claims.Unit``) the judge returns one **form** per claim segment, from
{assertion, alternatives, continuation, question, other}, each with a verbatim quote from the
readout (fail closed: a missing or non-verbatim quote -> ``other``). Code derives the unit's
``form_label`` (plan rev 4 item 4): any assertion -> ``assertion``; all of alternatives /
continuation / question -> ``option_form``; else ``other``.

**Category** (one per unit; precedence top to bottom, plan rev 4 + rev 2 item 1):

=====================  =========================================================  ==========
entity_substitution    the readout swaps the prompt's entity (set B); no probe      code
validated              the probe detects the concept (gates passing)               code
consensus              post hoc consensus (``claims.consensus_flags``; ML-NLA       code
                       units: ``consensus_mlnla``)
probe_na               no gate-passing probe for (concept, layer); the judge's      code
                       form is kept as ``form_label``
hallucination          form_label = assertion                                     judge
option                 form_label = option_form                                   judge
unclear                form_label = other (or no judge answer)                    judge
=====================  =========================================================  ==========

Every unit keeps its ``form_label`` (validated / consensus too: "hallucinated consensus" =
consensus with form_label assertion). An entity substitution asserted counts as a
hallucination in the rates.

Depth (rev 4 item 2): early <= 36, mid in {40, 42}, late >= 44. Default rule label: early ->
hallucination, late -> option, mid -> unclear. Sweep: binary cuts c in {32, 36, 40, 44}
(layer < c early -> hallucination, else option). The layer-blind run omits the layer AND the
depth label (rev 4 item 3).
"""

import re
from collections.abc import Mapping, Sequence
from typing import Any

from hallucination_bench.llm import schema_block
from hallucination_bench.readouts import normalize_ws

PROMPT_VERSION = "sibverdict-v2"  # plan 0004 rev 4: the one revision, from the 40-unit dev set
FORMS: tuple[str, ...] = ("assertion", "alternatives", "continuation", "question", "other")
OPTION_FORMS = frozenset({"alternatives", "continuation", "question"})
CATEGORIES: tuple[str, ...] = (
    "validated",
    "consensus",
    "option",
    "hallucination",
    "unclear",
    "probe_na",
    "entity_substitution",
)
JUDGE_CATEGORIES = ("option", "hallucination", "unclear")
CUTS: tuple[int, ...] = (32, 36, 40, 44)
SEG_MARK_OPEN, SEG_MARK_CLOSE = "⟦", "⟧"


# ------------------------------------------------------------------ depth / rule


def depth(layer: int) -> str:
    if layer <= 36:
        return "early"
    if layer >= 44:
        return "late"
    return "mid"


def rule_label(layer: int, cut: int | None = None) -> str:
    """Camila's rule of thumb. ``cut=None``: the default 3-way label (mid -> unclear);
    else the binary cut (layer < cut -> hallucination, else option)."""
    if cut is None:
        return {"early": "hallucination", "late": "option", "mid": "unclear"}[depth(layer)]
    return "hallucination" if layer < cut else "option"


# ------------------------------------------------------------------ form / category


def unit_form(seg_forms: Sequence[str]) -> str:
    fs = list(seg_forms)
    if not fs:
        return "other"
    if any(f == "assertion" for f in fs):
        return "assertion"
    if all(f in OPTION_FORMS for f in fs):
        return "option_form"
    return "other"


def judge_category(form_label: str | None) -> str:
    return {"assertion": "hallucination", "option_form": "option"}.get(form_label or "", "unclear")


def category(
    *,
    kind: str,
    probe_status: str,
    detected: bool,
    consensus: bool,
    form_label: str | None,
) -> str:
    """Precedence: entity_substitution > validated > consensus > probe_na > judge.
    ``probe_status``: ``ok`` (a gate-passing probe scored the cell) or ``na``."""
    if kind == "entity_substitution":
        return "entity_substitution"
    if probe_status == "ok" and detected:
        return "validated"
    if consensus:
        return "consensus"
    if probe_status != "ok":
        return "probe_na"
    return judge_category(form_label)


def is_hallucination(cat: str, form_label: str | None) -> bool:
    """Counted as a hallucination in the rates: the judge category, or an asserted entity
    substitution."""
    return cat == "hallucination" or (cat == "entity_substitution" and form_label == "assertion")


# ------------------------------------------------------------------ the judge prompt

SYSTEM = (
    "You label claims in interpretability-lens readouts. A lens read a language model's internal "
    "activation at one point of a prompt (the READ SITE) and produced a readout: text describing "
    "what the model was representing there. One concept named in the readout is under review: "
    "it is NOT the expected answer and is not named in the prompt or in what the model actually "
    "went on to write. Some segments of the readout that name it are marked "
    f"{SEG_MARK_OPEN}S1: ...{SEG_MARK_CLOSE}, {SEG_MARK_OPEN}S2: ...{SEG_MARK_CLOSE}, ...\n\n"
    "For each marked segment decide the FORM in which it presents the concept:\n"
    '- "assertion": it states the concept as a fact about this context: the answer, what the '
    'person / thing in the prompt is or does, what the user is, has or feels (e.g. "The sport '
    'he practices is Cricket", "she is a swimmer", "the answer is **Swimming**", "the user is '
    'from Japan"). This includes: a single candidate that is only hedged ("probably X", '
    '"likely X", "possibly X", "an occasion like X"); restating the prompt or the user\'s '
    "situation with the concept substituted, even as a question or request that presupposes it "
    '("day one of exploring London?" for a Prague trip, "you mentioned being a historian"); '
    "and describing the requested output as being about the concept when that output is about "
    'the user or the prompt\'s subject ("a biography of a Canadian musician" when the user asked '
    "for their own biography).\n"
    '- "alternatives": it offers the concept together with other named candidates in the same '
    'segment or sentence ("Boxing or football", "such as X, Y", "e.g. X, Y", "X, with Y and Z as '
    'other candidates", a list of choices, a multiple-choice line). Different bullets or samples '
    "that each name a different answer are separate claims, not a list of alternatives: label "
    "each on its own wording.\n"
    '- "continuation": it is sample text the model might go on to write that introduces the '
    "concept as content without misdescribing the prompt or the user (a line of a poem, an "
    "example sentence, a list item in a drafted reply).\n"
    '- "question": it genuinely asks whether the concept applies ("Is it X?"), rather than '
    "presupposing it.\n"
    '- "other": none of these, or you cannot tell.\n\n'
    "Judge by the segment's wording in the context of the whole readout. "
    "{prior}"
    "For each segment, quote verbatim (copy exactly, 3 to 20 words) the part of the readout that "
    "shows the form. Output only the JSON verdict."
)
PRIOR_LAYER = (
    "A prior to use only when the text is genuinely ambiguous: claims that the probe does not "
    "confirm tend to be assertions (hallucinations) at early layers and options or sample "
    "continuations at late layers. The wording always overrides this prior. "
)
PRIOR_BLIND = ""

USER = (
    "PROMPT (the read site is the end of the prompt{resp_note}):\n<prompt>\n{prompt}\n</prompt>\n\n"
    "WHAT THE MODEL WENT ON TO WRITE after the read site:\n<response>\n{response}\n</response>\n\n"
    "{layer_line}"
    "CONCEPT UNDER REVIEW: {concept}\n"
    "PROBE: {probe}\n\n"
    "READOUT ({n_readouts} sample(s); marked segments name the concept):\n<readout>\n{readout}\n"
    "</readout>\n\n"
    "Segments to label: {seg_ids}."
)


def schema() -> dict[str, Any]:
    return schema_block(
        "sibverdict",
        {
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "form": {"type": "string", "enum": list(FORMS)},
                        "quote": {"type": "string"},
                    },
                    "required": ["id", "form", "quote"],
                    "additionalProperties": False,
                },
            },
            "why": {"type": "string"},
        },
        ["segments", "why"],
    )


def mark_readouts(samples: Sequence[str], segments: Sequence[Mapping[str, Any]]) -> list[str]:
    """Insert ``⟦Sk: ...⟧`` around each unit segment (k = 1.. in segment order)."""
    out = [str(s) for s in samples]
    by_r: dict[int, list[tuple[int, int, int]]] = {}
    for k, s in enumerate(segments, 1):
        by_r.setdefault(int(s["readout"]), []).append((int(s["start"]), int(s["end"]), k))
    for r, spans in by_r.items():
        if r >= len(out):
            continue
        t = out[r]
        for a, b, k in sorted(spans, reverse=True):
            t = t[:a] + f"{SEG_MARK_OPEN}S{k}: " + t[a:b] + SEG_MARK_CLOSE + t[b:]
        out[r] = t
    return out


def probe_line(probe: Mapping[str, Any]) -> str:
    if probe.get("status") != "ok":
        return "no validated probe for this concept at this layer (not available)"
    z = probe.get("z")
    zs = f"{z:+.2f}" if isinstance(z, int | float) else "n/a"
    det = "detected" if probe.get("detected") else "not detected"
    return (
        f"a linear probe for the concept at this activation: {det} (z = {zs} vs. the gold context)"
    )


def render(
    *,
    prompt: str,
    response: str,
    concept: str,
    samples: Sequence[str],
    segments: Sequence[Mapping[str, Any]],
    layer: int | None,
    probe: Mapping[str, Any],
    resp_note: str = "",
    tokens_line: str = "",
) -> tuple[str, str]:
    """(system, user). ``layer=None`` = the layer-blind run (no layer, no depth label, no
    layer prior)."""
    marked = mark_readouts(samples, segments)
    readout = "\n\n".join(f"[sample {i + 1}]\n{t}" for i, t in enumerate(marked))
    if tokens_line:
        readout = f"{tokens_line}\n\n{readout}"
    system = SYSTEM.replace("{prior}", PRIOR_LAYER if layer is not None else PRIOR_BLIND)
    layer_line = (
        f"LENS LAYER: {layer} of 64 ({depth(layer)}: early <= 36, mid 40-42, late >= 44)\n"
        if layer is not None
        else ""
    )
    vals = {
        "resp_note": resp_note,
        "prompt": prompt,
        "response": response or "(none)",
        "layer_line": layer_line,
        "concept": concept,
        "probe": probe_line(probe) if layer is not None else _probe_blind(probe),
        "n_readouts": str(len(samples)),
        "readout": readout,
        "seg_ids": ", ".join(f"S{k}" for k in range(1, len(segments) + 1)),
    }
    user = USER
    for k in vals:
        user = user.replace("{" + k + "}", "\0" + k + "\0")
    for k, v in vals.items():
        user = user.replace("\0" + k + "\0", v)
    return system, user


def _probe_blind(probe: Mapping[str, Any]) -> str:
    # the probe result stays in the blind run; only the layer and depth are removed
    return probe_line(probe)


def validate(raw: Mapping[str, Any] | None, n_segments: int) -> bool:
    if not isinstance(raw, Mapping) or not isinstance(raw.get("segments"), list):
        return False
    ids = {str(s.get("id", "")).strip().upper() for s in raw["segments"] if isinstance(s, Mapping)}
    return all(f"S{k}" in ids for k in range(1, n_segments + 1))


_SEG_ID = re.compile(r"^S(\d+)$")


def _strip_marks(s: str) -> str:
    s = re.sub(re.escape(SEG_MARK_OPEN) + r"S\d+:\s*", "", s)
    return s.replace(SEG_MARK_CLOSE, "")


def parse(raw: Mapping[str, Any] | None, n_segments: int, samples: Sequence[str]) -> dict[str, Any]:
    """-> {seg_forms: [form per segment], quotes, quote_ok, form_label, why, n_unquoted}.
    Fail closed: a missing segment, unknown form or a quote not found verbatim (whitespace-
    normalised, case-sensitive, marks removed) in the readout -> ``other``."""
    hay = normalize_ws("\n".join(str(s) for s in samples))
    forms = ["other"] * n_segments
    quotes: list[str | None] = [None] * n_segments
    ok = [False] * n_segments
    n_unq = 0
    if isinstance(raw, Mapping):
        for s in raw.get("segments") or ():
            if not isinstance(s, Mapping):
                continue
            m = _SEG_ID.match(str(s.get("id", "")).strip().upper())
            if not m:
                continue
            k = int(m.group(1)) - 1
            if not 0 <= k < n_segments:
                continue
            q = _strip_marks(str(s.get("quote") or "")).strip()
            quotes[k] = q
            f = s.get("form")
            if f not in FORMS:
                continue
            if q and normalize_ws(q) in hay:
                forms[k] = f
                ok[k] = True
            else:
                n_unq += 1
    return {
        "seg_forms": forms,
        "quotes": quotes,
        "quote_ok": ok,
        "form_label": unit_form(forms),
        "why": (raw or {}).get("why") if isinstance(raw, Mapping) else None,
        "n_unquoted": n_unq,
        "answered": isinstance(raw, Mapping),
    }
