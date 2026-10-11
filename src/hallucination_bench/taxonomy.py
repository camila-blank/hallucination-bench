"""THE hallucination taxonomy (v1): labels, definitions, examples and rules.

This module is the single source of truth. The classifier prompt (``judges/taxonomy_v1.py``)
is rendered from it by :func:`render`, scoring reads its enums, and the calibration README
points here. Tests assert that the rendered prompt contains every label and definition.

Axes:

* ``grounding`` — set in code from the stage that found the span, never by the classifier.
* ``type`` (concept vs detail) — the primary axis.
* ``target`` — what kind of thing is wrong.
* ``severity`` — 1 minor, 2 moderate, 3 major.
* ``unsupported_kind`` — why an ``unsupported`` span is ungrounded (``None`` otherwise).
"""

from dataclasses import dataclass

TAXONOMY_VERSION = "v1"
UNKNOWN = "unknown"

# ---------------------------------------------------------------- grounding (set in code)

GROUNDINGS: dict[str, str] = {
    "contradicted": (
        "a span the span audit verified as wrong. The response, or the facts, rule it out."
    ),
    "disputed": (
        "the claim audit thinks the response rules the claim out, but the span audit did not "
        "flag it."
    ),
    "unsupported": (
        "the claim is specific, but the response neither states it nor clearly entails it, and "
        "nothing realises it after the marker. Only content that is stated or realised counts as "
        "grounded, so plausible inferences, unrealised predictions and fabricated specifics are "
        "all unsupported. A claim that is true in the world but not stated in the conversation "
        "is also unsupported."
    ),
}

# ---------------------------------------------------------------- unsupported_kind

UNSUPPORTED_KINDS: dict[str, str] = {
    "prediction": ("about what the response will say next, and the response never says it."),
    "inference": (
        "a plausible inference from the text that the text does not state or entail. This "
        "includes claims about the model's intent or mental state."
    ),
    "fabricated": (
        "a specific with no basis in the text, such as an invented name, number, date, "
        "quotation, event or source."
    ),
    "world_knowledge": (
        "a fact about the world, possibly true, that the conversation does not state."
    ),
}
LENIENT_DROPPED_KINDS: tuple[str, ...] = ("inference", "world_knowledge")

# ---------------------------------------------------------------- type

MAIN_TEST = (
    "If a reader believed the span, would they misunderstand what the text is about or what it "
    "claims?"
)


@dataclass(frozen=True)
class TypeDef:
    label: str
    definition: str
    examples: tuple[str, ...]


TYPES: dict[str, TypeDef] = {
    "concept": TypeDef(
        "concept",
        "yes, the meaning is wrong. The span substitutes a different, meaningfully distinct "
        "entity, idea, event or situation as the subject of a claim, or it reverses a claim, "
        "stance or outcome.",
        (
            '"Picasso" when the text is about da Vinci',
            "Johnson where the text says Kennedy",
            '"argues remote work hurts productivity" where the text argues the opposite',
            '"a recipe for bread" where the response is a poem about bread',
            '"the user is angry" where the user is grateful',
            '"a 200 mg dose is safe" where the text says it is dangerous',
        ),
    ),
    "detail": TypeDef(
        "detail",
        "no, the gist is right. A specific value inside an otherwise correct understanding is "
        "wrong or invented.",
        (
            '"Steve" when the text is about an arbitrary character named Joe',
            '"October 24" where the text says October 29',
            '"over 700 died" where the text says over 140',
            '"3 tips" where the text lists 5',
            '"Quick Facts" quoted as a heading where it is "Basic Facts"',
            "an incidental real-world name swapped in a passing mention (the text cites "
            '"a 2019 Stanford study" in an aside and the readout says Harvard)',
        ),
    ),
}

NAME_RULE_INTRO = (
    "Names: apply this decision order. What decides the type is whether the wrong name carries "
    'its own identity. Whether the referent is "the same" does not decide it.'
)
NAME_RULE_STEPS: tuple[str, ...] = (
    "The wrong name denotes a distinct identity and is the subject of the claim -> concept. A "
    "distinct identity is a real person, work, place or organisation, or another character or "
    "entity present in the text. Examples: Picasso for da Vinci; Johnson for Kennedy; confusing "
    'the protagonist with her rival; an arbitrary protagonist Joe called "Barack Obama".',
    "The wrong name denotes a distinct identity but appears only in a passing mention -> "
    'detail. Example: the text cites "a 2019 Stanford study" in an aside and the readout says '
    "Harvard.",
    "Otherwise, the name is an arbitrary or invented label for the right referent -> detail, "
    "even when the referent is the protagonist. Example: Steve for the story's Joe.",
)
VALUE_RULE = (
    "Wrong values: a number, date, label or wording error is detail. The exception is a value "
    "that flips the claim (safe vs unsafe, won vs lost, increase vs decrease), which is concept. "
    "Magnitude alone never makes a span concept; it raises severity instead."
)
CONTRADICTED_RULE = (
    "contradicted and disputed spans use the main test above, since a correct value exists."
)
UNSUPPORTED_RULE_INTRO = (
    'For unsupported spans there is no "correct value", so the test becomes: does believing the '
    "span change the reader's understanding of what the text is about or what it claims?"
)
UNSUPPORTED_RULE_CASES: tuple[str, ...] = (
    "An invented event, claim, topic, situation, intent or stance is concept. Examples: "
    '"the response goes on to recommend therapy", when it never does and nothing suggests it; '
    '"the user is a teacher".',
    "An invented name, number, date, quotation or other specific value attached to something "
    'the text does discuss is detail. Examples: "Joe\'s sister Maria", when the text never '
    'names her; "about 30 minutes".',
)

# ---------------------------------------------------------------- target

TARGETS: dict[str, str] = {
    "entity": "a named person, place, work, organisation or product.",
    "quantity": "a number, date, time, count or measurement.",
    "event": "what happened, or who did what.",
    "attribute": (
        "a property or relation of something, such as a cause, part-of, or the attribution of a "
        "quote."
    ),
    "stance": "an opinion, polarity, negation, sentiment or intent.",
    "structure": (
        "headings, list items, quoted wording, format, or what the response has already written."
    ),
    "topic": "the overall subject or situation of the conversation.",
}

# ---------------------------------------------------------------- severity

SEVERITY_INTRO = "how much the span would mislead a reader about this conversation."
SEVERITIES: dict[int, tuple[str, str]] = {
    1: ("minor", "peripheral; it does not touch anything the response relies on."),
    2: ("moderate", "it misstates, or invents, a specific point the response makes or relies on."),
    3: ("major", "it misstates, or invents, the central subject, claim, answer or situation."),
}


def definitions() -> list[str]:
    """Every label and definition string that the rendered prompt must contain (tests)."""
    out: list[str] = [MAIN_TEST, NAME_RULE_INTRO, VALUE_RULE, CONTRADICTED_RULE]
    out += [*NAME_RULE_STEPS, UNSUPPORTED_RULE_INTRO, *UNSUPPORTED_RULE_CASES]
    for k, v in GROUNDINGS.items():
        out += [k, v]
    for k, v in UNSUPPORTED_KINDS.items():
        out += [k, v]
    for t in TYPES.values():
        out += [t.label, t.definition, *t.examples]
    for k, v in TARGETS.items():
        out += [k, v]
    for n, (name, d) in SEVERITIES.items():
        out += [f"{n} {name}", d]
    out.append(SEVERITY_INTRO)
    return out


def render() -> str:
    """The taxonomy as prompt text (markdown)."""
    lines: list[str] = [f"# Hallucination taxonomy ({TAXONOMY_VERSION})", ""]
    lines += [
        "## grounding (given with each span; you never change it)",
        "",
        *(f"- `{k}`: {v}" for k, v in GROUNDINGS.items()),
        "",
        "## type (the primary axis)",
        "",
        f"The main test decides the type: {MAIN_TEST}",
        "",
    ]
    for t in TYPES.values():
        lines.append(f"- `{t.label}`: {t.definition} Examples:")
        lines += [f"  - {e};" for e in t.examples]
    lines += ["", "Rules:", "", f"- {NAME_RULE_INTRO}"]
    lines += [f"  {i}. {s}" for i, s in enumerate(NAME_RULE_STEPS, 1)]
    lines += [f"- {VALUE_RULE}", f"- {CONTRADICTED_RULE}", f"- {UNSUPPORTED_RULE_INTRO}"]
    lines += [f"  - {c}" for c in UNSUPPORTED_RULE_CASES]
    lines += [
        "",
        "## target (what kind of thing is wrong; exactly one)",
        "",
        *(f"- `{k}`: {v}" for k, v in TARGETS.items()),
        "",
        f"## severity ({SEVERITY_INTRO})",
        "",
        *(f"- `{n} {name}`: {d}" for n, (name, d) in SEVERITIES.items()),
        "",
        "## unsupported_kind (unsupported spans only; null for contradicted and disputed spans)",
        "",
        *(f"- `{k}`: {v}" for k, v in UNSUPPORTED_KINDS.items()),
    ]
    return "\n".join(lines)
