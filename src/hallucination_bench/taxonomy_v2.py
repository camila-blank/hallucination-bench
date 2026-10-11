"""THE claim-map taxonomy (v2): leaves, groups, definitions, precedence, tags and decisions.

Plan 0002 (``plans/0002-claim-map-taxonomy-v2.md``). This module is the single source of truth:
the claim-map judge prompt (``judges/claimmap_v2.py``) is rendered from it by :func:`render`,
scoring and calibration read its enums, and tests assert the rendered prompt contains every
leaf definition (:func:`definitions`).

Every segment of a readout lands in exactly one of the 19 judge leaves, which fall into 7
groups. Two code-assigned pseudo-leaves (``unknown_incorrect``, ``failed``) are kept as separate
rows outside the groups.

**Open decisions** (``OPEN_DECISIONS``) are pending Camila's answer. Each has a ``default`` and
the text the prompt renders for every option; change ``CHOSEN`` (one line per decision) to switch
an option. The rendered prompt, and so every ``:M`` cache fingerprint, follows ``CHOSEN``; bump
``TAXONOMY_VERSION`` when you change it so the cached labels are re-queued knowingly.
"""

from dataclasses import dataclass

TAXONOMY_VERSION = "v2"

# ---------------------------------------------------------------- groups

HALLUCINATION = "hallucination"
CO_ACTIVATION = "co_activation"
CORRECT = "correct"
THINKY = "thinky"
UNSETTLED = "unsettled"
JUNK = "junk"
NON_CLAIM = "non_claim"
GROUPS: tuple[str, ...] = (
    HALLUCINATION,
    CO_ACTIVATION,
    CORRECT,
    THINKY,
    UNSETTLED,
    JUNK,
    NON_CLAIM,
)
GROUP_TITLES: dict[str, str] = {
    HALLUCINATION: "hallucinations: wrong details and false claims",
    CO_ACTIVATION: "plausible co-activation: substitutions of a related object",
    CORRECT: "correct content",
    THINKY: (
        "thinky unverifiable content: unrealised predictions, the model's own state and beliefs, "
        "possible model beliefs"
    ),
    UNSETTLED: "unsettled: ambiguous topic",
    JUNK: "junk: off-topic claims",
    NON_CLAIM: "non-claims",
}

# code-assigned pseudo-leaves (never in the judge's enum)
UNKNOWN_INCORRECT = "unknown_incorrect"
FAILED = "failed"
PSEUDO_LEAVES: tuple[str, ...] = (UNKNOWN_INCORRECT, FAILED)
ROWS: tuple[str, ...] = (*GROUPS, *PSEUDO_LEAVES)  # the identity holds over these 9 rows

# ---------------------------------------------------------------- leaves


@dataclass(frozen=True)
class Leaf:
    name: str
    branch: str  # non_claim | correct | incorrect | thinky | unsettled | junk
    group: str
    definition: str


_LEAVES: tuple[Leaf, ...] = (
    # non-claim
    Leaf(
        "generic",
        "non_claim",
        NON_CLAIM,
        'generic description, nothing concrete ("a friendly reply").',
    ),
    Leaf(
        "filler",
        "non_claim",
        NON_CLAIM,
        "hedges, connectives, repetition, verbatim restatement of the prompt, list scaffolding, "
        'and the framing words of token-bag summaries ("the tokens include…").',
    ),
    Leaf(
        "formatting",
        "non_claim",
        NON_CLAIM,
        "markup, bullets, empty headings, emoji (fragments that assert nothing are filler).",
    ),
    Leaf(
        "junk_text",
        "non_claim",
        NON_CLAIM,
        "garbled or broken tokens, stray code, byte artifacts.",
    ),
    Leaf(
        "lens_meta",
        "non_claim",
        NON_CLAIM,
        "statements about the lens output itself rather than the conversation "
        "(\"Russian 'холод' (cold) and Chinese '冷的' (cold)\", \"emoji byte artifacts\").",
    ),
    # correct
    Leaf(
        "restates_prefix",
        "correct",
        CORRECT,
        "stated in the prompt or in the response before the marker.",
    ),
    Leaf(
        "stated_after",
        "correct",
        CORRECT,
        "stated in the response after the marker (a realised prediction OR an asserted fact "
        "that the continuation states).",
    ),
    Leaf(
        "entailed",
        "correct",
        CORRECT,
        "not stated anywhere, but strictly entailed by the conversation.",
    ),
    Leaf(
        "world_true",
        "correct",
        CORRECT,
        "on topic, true in the world, not stated or entailed by the conversation.",
    ),
    # incorrect
    Leaf(
        "detail",
        "incorrect",
        HALLUCINATION,
        "[hallucination] a wrong specific with no concept of its own (see the Detail rule).",
    ),
    Leaf(
        "false_claim",
        "incorrect",
        HALLUCINATION,
        "[hallucination] a false assertion about the right object: reversed outcome or stance, "
        "wrong agent (who did what to whom), false relation or cause, a wrong real-world "
        "quantity, or an invented event or fact asserted about the conversation.",
    ),
    Leaf(
        "substitution_sibling",
        "incorrect",
        CO_ACTIVATION,
        "[co-activation] a different object of the same class in place of the right one.",
    ),
    Leaf(
        "substitution_category",
        "incorrect",
        CO_ACTIVATION,
        "[co-activation] a different object of a different category.",
    ),
    # unverifiable, on topic (thinky)
    Leaf(
        "unrealised_prediction",
        "thinky",
        THINKY,
        "framed as anticipation of the continuation, plausible from the text before the marker, "
        "and not said by the response.",
    ),
    Leaf(
        "model_state",
        "thinky",
        THINKY,
        "about the model's own beliefs, perceptions, state, intent, plans or awareness, "
        "including claims about the nature or purpose of the conversation (a test, an "
        "evaluation, a roleplay), whatever the framing.",
    ),
    Leaf(
        "possible_belief",
        "thinky",
        THINKY,
        "a world claim the judge believes is false, which the conversation neither states nor "
        "contradicts; the model may really hold it.",
    ),
    Leaf(
        "other_on_topic",
        "thinky",
        THINKY,
        "any other on-topic claim that the conversation and world knowledge cannot settle, "
        "including contested or doubtful world claims.",
    ),
    # unsettled / junk
    Leaf(
        "ambiguous_topic",
        "unsettled",
        UNSETTLED,
        "a claim where it cannot be told whether it refers to this conversation or any subject "
        "it raises, whatever its truth (rule 5).",
    ),
    Leaf(
        "off_topic",
        "junk",
        JUNK,
        "a claim about neither this conversation nor any subject it raises, whatever its truth "
        "(a Mars-probe fact in an answer about idioms; rule 5).",
    ),
)

LEAVES: dict[str, Leaf] = {lf.name: lf for lf in _LEAVES}
LEAF_NAMES: tuple[str, ...] = tuple(LEAVES)
GROUP_OF: dict[str, str] = {
    **{lf.name: lf.group for lf in _LEAVES},
    UNKNOWN_INCORRECT: UNKNOWN_INCORRECT,
    FAILED: FAILED,
}
BRANCH_OF: dict[str, str] = {lf.name: lf.branch for lf in _LEAVES}
INCORRECT_LEAVES: tuple[str, ...] = tuple(n for n, b in BRANCH_OF.items() if b == "incorrect")
SUBSTITUTION_LEAVES: tuple[str, ...] = ("substitution_sibling", "substitution_category")
CORRECT_LEAVES: tuple[str, ...] = tuple(n for n, g in GROUP_OF.items() if g == CORRECT)
THINKY_LEAVES: tuple[str, ...] = tuple(n for n, g in GROUP_OF.items() if g == THINKY)
NON_CLAIM_LEAVES: tuple[str, ...] = tuple(n for n, g in GROUP_OF.items() if g == NON_CLAIM)
ALL_LEAVES: tuple[str, ...] = (*LEAF_NAMES, *PSEUDO_LEAVES)

assert len(LEAF_NAMES) == 19 and len(GROUPS) == 7


def group_of(leaf: str) -> str:
    return GROUP_OF.get(leaf, FAILED)


def is_claim(leaf: str) -> bool:
    """Claim vs non-claim (the claim/non-claim F1): every leaf outside ``non_claim`` and
    ``failed`` asserts something (``unknown_incorrect`` is an established wrong span)."""
    return group_of(leaf) not in (NON_CLAIM, FAILED)


# ---------------------------------------------------------------- open decisions


@dataclass(frozen=True)
class OpenDecision:
    question: str
    default: str
    options: dict[str, str]  # option -> the prompt text it renders


OPEN_DECISIONS: dict[str, OpenDecision] = {
    # 1. Picasso for da Vinci
    "subject_name": OpenDecision(
        "A wrong named entity with its own identity that is the SUBJECT of the claim (Picasso "
        "for da Vinci): substitution (co-activation) or a false claim (hallucination)?",
        "substitution",
        {
            "substitution": (
                "This holds even when the wrong name is the subject of the claim: Picasso for "
                "da Vinci is `substitution_sibling`."
            ),
            "false_claim": (
                "Exception: when the wrong name is the subject of the claim (Picasso for da "
                "Vinci), it is `false_claim`, not a substitution."
            ),
        },
    ),
    # 2. numbers and invented specifics
    "invented_specific": OpenDecision(
        "An invented arbitrary name or number with nothing in the conversation for it to "
        'replace ("the user, Sarah", "her 3 kids"): detail, or an invented fact (false_claim)?',
        "detail",
        {
            "detail": (
                "an **invented** arbitrary specific with nothing in the conversation for it to "
                'replace ("the user, Sarah, …" when nobody is named; "her 3 kids"). It is still '
                "a name or number without a concept of its own."
            ),
            "false_claim": (
                "(An invented arbitrary specific with nothing in the conversation for it to "
                'replace, such as "the user, Sarah, …" when nobody is named, is NOT detail: it '
                "is an invented fact, `false_claim`.)"
            ),
        },
    ),
    # 3. cruxy
    "cruxy": OpenDecision(
        "How to mark important / cruxy errors: a `cruxy` tag on incorrect segments?",
        "tag",
        {
            "tag": (
                "(bool, incorrect segments only): the segment misstates the central "
                "subject, claim or answer of the conversation. Set it to false on every other "
                "segment."
            ),
        },
    ),
    # 4. the wrong-bridge direction
    "wrong_bridge": OpenDecision(
        "World-false content that the response itself states: the conversation wins (correct, "
        "tag world_false) or the world wins (incorrect)?",
        "conversation_wins",
        {
            "conversation_wins": (
                "World-false content that the response itself states is `restates_prefix` or "
                "`stated_after`, with the tag `world_false = true` (the wrong-bridge case: the "
                "response uses the wrong bridge, and the readout finds it)."
            ),
            "world_wins": (
                "World-false content that the response itself states is still incorrect (leaf "
                "by rule 8), with the tag `world_false = true`."
            ),
        },
    ),
    # 5. hedged inferences about the user or setting
    "hedged_inference": OpenDecision(
        "A hedged, unsupported inference about the user or setting: other_on_topic (the lens is "
        "guessing) or incorrect?",
        "other_on_topic",
        {
            "other_on_topic": (
                'When it is explicitly hedged as an inference ("likely", "probably", "may be", '
                '"seems"), it is `other_on_topic`: the lens is guessing, not asserting.'
            ),
            "incorrect": (
                'Even when it is explicitly hedged as an inference ("likely", "probably", "may '
                'be", "seems"), an unsupported fact about the user or setting is incorrect '
                "(leaf by rule 8)."
            ),
        },
    ),
}

# Change one line here to switch an open decision (then bump TAXONOMY_VERSION).
CHOSEN: dict[str, str] = {k: d.default for k, d in OPEN_DECISIONS.items()}


def decision_text(key: str) -> str:
    d = OPEN_DECISIONS[key]
    return d.options[CHOSEN.get(key, d.default)]


# Camila's decisions (2026-10-11, verbatim) and where they are encoded (documentation only).
DECISIONS: tuple[tuple[str, str], ...] = (
    (
        "same class sibling and different category substitution are different from factually "
        "false claims so there should be two categories not two full axes",
        "two substitution leaves, separate from false_claim",
    ),
    (
        "unrealized predictions are unverifiable on topic, they are thinky stuff",
        "rule 10 and unrealised_prediction",
    ),
    (
        "for substitution i think these could plausibly be things the model is really thinking "
        "about…",
        "the substitutions form the co-activation group (not a hallucination); pair logged",
    ),
    ("the invented event is a false claim. it's verifiable.", "rule 10"),
    (
        "world checkable claims that are on topic and correct should be verifiable correct, "
        "claims that are on topic and incorrect are either unverifiable because we are not sure "
        "whether the model is actually thinking about this or not or maybe they are just "
        "verifiably incorrect because they contradict with the following text",
        "world_true; contradicted -> incorrect; silent -> possible_belief; stated -> rule 6",
    ),
    ("hedged lists are fine", "the hedged tag"),
    ("eval awareness (as thinky content)", "rule 4 and model_state"),
)

# ---------------------------------------------------------------- units / segments

UNITS_TEXT: tuple[str, ...] = (
    "Code has pre-split each readout into units with ids `R{readout}.U{n}`; together they cover "
    "the readout. You label every unit.",
    "When a unit carries more than one claim, or mixes claim and non-claim text, **split** it: "
    "quote each sub-span verbatim (character for character from the unit, in left-to-right "
    "order, never overlapping) and give each its own leaf. Give `rest_leaf` for the unquoted "
    "remainder of a split unit; it is required whenever you split. A unit that is one claim, or "
    "one non-claim, is not split.",
    "A segment is a unit or a quoted sub-span. Atomic-claim rule: one segment carries one claim. "
    "A sentence with a correct part and a wrong part must be split. Hedge words stay inside "
    'their claim segment. An invented specific ("her 3 kids") is one `detail` segment, with no '
    "separate existence claim.",
)

# ---------------------------------------------------------------- precedence


@dataclass(frozen=True)
class Rule:
    number: int
    title: str
    body: tuple[str, ...]  # first entry: the rule; the rest: bullets


LENS_FRAMING_RULE = (
    "A summary's lens framing, such as \"The model is representing / anticipating / focused on "
    '/ gearing up for X", is **framing** (`filler`). The X it attributes is judged as an '
    "ordinary claim about the text or as a prediction. It is **never `model_state`**, which is "
    "reserved for readouts that make claims about the model's beliefs or intent as such."
)


def _rules() -> dict[int, Rule]:
    return {
        1: Rule(
            1,
            "Non-claim vs claim.",
            (
                "A segment is a claim if it commits to anything concrete about the "
                "conversation, the model, or the world.",
                'Token-bag summaries: the framing ("the tokens include X") is `filler`, and a '
                "gloss of the tokens is `lens_meta`. The interpretation, including its "
                'connective ("suggesting the passage is about Y"), is one claim segment, judged '
                'as usual. The connective ("suggesting", "indicating", "points to") is **not a '
                "hedge**: the interpretation counts as asserted for rules 10 and 12. "
                + LENS_FRAMING_RULE,
                "Echoes: a literal copy of the prompt's or the response's words that asserts "
                'nothing (a sample that just repeats "I\'d be delighted to") is `filler`, even '
                "in the first person (this is checked before rule 3). Any assertion about the "
                'content ("the user asks who your partner is") is a claim.',
            ),
        ),
        2: Rule(
            2,
            "Established spans must be incorrect.",
            (
                "A unit marked ESTABLISHED is a span that an earlier audit verified as wrong. It "
                "must get an incorrect leaf (`detail`, `false_claim`, `substitution_sibling` or "
                "`substitution_category`; leaf by rule 8). You may split it, but at least one "
                "sub-span must then be incorrect. If you are confident it is NOT wrong, set "
                "`dispute = true`, put your best incorrect reading in `leaf` and your preferred "
                "leaf in `alt_leaf`. Never call an established unit a non-claim instead of "
                "disputing it.",
            ),
        ),
        3: Rule(
            3,
            "Continuation-style text.",
            (
                "Some readouts are written as the response's own continuation, in the first "
                'person ("As an AI, I don\'t have personal relationships"). Such text is an '
                "implicit prediction of the response and is judged by rules 7 and 9:",
                "`restates_prefix` or `stated_after` if the response says it (tag `world_false` "
                "if it is false in the world, per rule 6);",
                "`unrealised_prediction` if it is plausible but the response does not say it;",
                "incorrect if the text before the marker rules it out;",
                "if it is implausible but not ruled out, rule 10's last bullet applies.",
                "It is **not** `model_state`, even though it is in the first person.",
            ),
        ),
        4: Rule(
            4,
            "The model's own beliefs or intent; test / evaluation / roleplay framing.",
            (
                "Claims about the model's own beliefs or intent, or that frame the conversation "
                "as a test, evaluation, roleplay or deception, go to the thinky branch (leaf by "
                "rule 9; usually `model_state`). Evaluation awareness lands here.",
                'Carve-out: a statement of intent that predicts response text ("the model is '
                'about to apologise") is a prediction and goes to rules 3 and 10: '
                "`stated_after` if realised, `unrealised_prediction` if not.",
                'Descriptions of format, genre or setting ("Q&A on an educational platform") '
                "are ordinary claims about the conversation: an invented setting **asserted "
                "flatly** is `false_claim` (rule 10); a hedged one follows rule 12.",
            ),
        ),
        5: Rule(
            5,
            "Topic gate (before truth).",
            (
                "A claim about this conversation or a subject it raises is on topic and is "
                "never `off_topic`. Its branch is set by rules 6, 10, 12 and 13. A claim clearly "
                "about neither is `off_topic`; one where it cannot be told is "
                "`ambiguous_topic`. Both apply **whatever the claim's world truth**: a true "
                "Mars-probe fact in an answer about idioms is `off_topic`.",
            ),
        ),
        6: Rule(
            6,
            "Conversation vs world.",
            (
                "The conversation wins for claims about the conversation and its content. "
                + decision_text("wrong_bridge")
                + " (Predictions are excepted: a claim framed as anticipation follows rule 10, "
                "so it is incorrect only if the text **before** the marker rules it out.) **A "
                "non-prediction world claim is incorrect only if the conversation contradicts "
                "it**, or if it overlaps an established span. An on-topic world-false claim on "
                "which the conversation is silent goes to the thinky branch, leaf by rule 9 "
                "(usually `possible_belief`).",
            ),
        ),
        7: Rule(
            7,
            "Inside correct.",
            ("`restates_prefix` > `stated_after` > `entailed` > `world_true`.",),
        ),
        8: Rule(
            8,
            "Inside incorrect. Apply in this order:",
            (
                "",
                "Step 1. A wrong named object that has its own identity, in place of the right "
                "one -> substitution (`substitution_sibling` if it is of the same class, "
                "`substitution_category` if of a different category). "
                + decision_text("subject_name"),
                "Step 2. Anything matching the Detail rule list, including invented arbitrary "
                "specifics -> `detail`. The Detail rule's carve-outs (real-event quantities, "
                "counts that flip the claim) go on to step 3.",
                "Step 3. Any other false assertion about the right object -> `false_claim`.",
            ),
        ),
        9: Rule(
            9,
            "Inside the thinky branch (unverifiable, on topic).",
            ("`unrealised_prediction` > `model_state` > `possible_belief` > `other_on_topic`.",),
        ),
        10: Rule(
            10,
            "Framing.",
            (
                "",
                "A concrete event or fact **asserted** about the conversation that the "
                "conversation does not support -> the **incorrect** branch (leaf by rule 8: "
                "`detail` for invented names or numbers, otherwise usually `false_claim`).",
                "A claim **framed as anticipation** that is plausible, but that the response "
                "does not say, is `unrealised_prediction`.",
                "A prediction that the text before the marker rules out is incorrect.",
                "A prediction that is implausible but not ruled out -> the thinky branch (rule "
                "9: `possible_belief` if it is world-false, otherwise `other_on_topic`).",
            ),
        ),
        11: Rule(
            11,
            "Echoes.",
            ("Folded into rule 1 (the number is kept so cross-references stay stable).",),
        ),
        12: Rule(
            12,
            "Hedged inferences about the user or setting.",
            (
                "A flat assertion of an unsupported fact about the user or setting -> the "
                'incorrect branch (rule 10; leaf by rule 8). For example, "the user is a '
                'child" or "(Scratch-like community)" stated as fact. '
                + decision_text("hedged_inference")
                + " A hedged inference that the conversation **contradicts** is still "
                "incorrect (rule 8 picks the leaf).",
            ),
        ),
        13: Rule(
            13,
            "On-topic fallback.",
            (
                "An on-topic claim (rule 5) that rules 6, 10 and 12 have not placed, and that "
                "neither the conversation nor world knowledge settles, goes to the **thinky** "
                "branch (leaf by rule 9, else `other_on_topic`).",
            ),
        ),
    }


RULES: dict[int, Rule] = _rules()
BRANCH_RULE_ORDER: tuple[int, ...] = (2, 1, 3, 4, 5, 6, 10, 11, 12, 13)
LEAF_RULE_ORDER: tuple[int, ...] = (7, 8, 9)
RULE_ORDER: tuple[int, ...] = (*BRANCH_RULE_ORDER, *LEAF_RULE_ORDER)
PRECEDENCE_INTRO = (
    "Rules 1–6 and 10–13 **select the branch** (non-claim, correct, incorrect, thinky, "
    "unsettled, junk). Apply them first, top-down in the order listed; the first one that fits "
    "wins. Rule 2 (established spans) is checked before rule 1, so you dispute an established "
    "span rather than call it a non-claim. Rules 7–9 **order the leaves within a branch** and "
    "apply only after the branch is fixed."
)

# ---------------------------------------------------------------- detail rule

DETAIL_RULE_INTRO = "`detail` covers wrong specifics with no concept representation of their own:"


def _detail_items() -> tuple[str, ...]:
    items = [
        "arbitrary or fictional names;",
        "incidental numbers, dates or counts in fiction or scaffolding;",
        "quoted wording or heading text;",
    ]
    if CHOSEN.get("invented_specific", "detail") == "detail":
        items.append(decision_text("invented_specific"))
    return tuple(items)


DETAIL_ITEMS: tuple[str, ...] = _detail_items()
DETAIL_EXTRA: tuple[str, ...] = (
    ()
    if CHOSEN.get("invented_specific", "detail") == "detail"
    else (decision_text("invented_specific"),)
)
DETAIL_CARVE_OUT = (
    'Quantities and dates of **real events**, and counts that **flip the claim** ("0 errors" '
    "for 3), are `false_claim` **when the conversation contradicts them**; if it is silent they "
    'are `possible_belief` (rule 6). Wording and digits are treated alike, so "no errors" and '
    '"0 errors" both count as `false_claim`.'
)

# ---------------------------------------------------------------- tags

TAGS: dict[str, str] = {
    "hedged": (
        "(bool): a short hedged list or disjunction of at most 4 options; it is one segment. "
        "Hit (the realised or stated option is in the list): it gets the correct leaf it would "
        "get on its own. Miss: it is judged as the prediction or claim it is "
        "(`unrealised_prediction` if plausible, incorrect if ruled out). A longer scattershot "
        "list of more than 4 unrelated options is `generic`."
    ),
    "world_false": "(bool): the segment is false in the world; see rule 6.",
    "cruxy": decision_text("cruxy"),
    "dispute": (
        "(bool, ESTABLISHED units only): you are confident the established span is not wrong; "
        "see rule 2. Give `alt_leaf` with it; otherwise `alt_leaf` is null."
    ),
    "pair": (
        "(substitutions only): `{said, actual}`, the object the readout says and the object it "
        "replaces; null for every other leaf."
    ),
}


# ---------------------------------------------------------------- render


def definitions() -> list[str]:
    """Every label and definition string the rendered prompt must contain (tests)."""
    out: list[str] = [PRECEDENCE_INTRO, DETAIL_RULE_INTRO, DETAIL_CARVE_OUT, *UNITS_TEXT]
    for lf in _LEAVES:
        out += [f"`{lf.name}`", lf.definition]
    for r in RULES.values():
        out += [r.title, *(b for b in r.body if b)]
    out += [*DETAIL_ITEMS, *DETAIL_EXTRA]
    for k, v in TAGS.items():
        out += [f"`{k}`", v]
    return out


def render() -> str:
    """The taxonomy as prompt text (markdown)."""
    lines: list[str] = [f"# Claim map taxonomy ({TAXONOMY_VERSION})", "", "## Units", ""]
    lines += [f"- {t}" for t in UNITS_TEXT]
    lines += [
        "",
        "## The tree: 19 leaves in 7 groups",
        "",
        "Every segment gets exactly one leaf.",
        "",
    ]
    branch_titles = {
        "non_claim": "non_claim (group: non_claim)",
        "correct": "claim → correct (group: correct)",
        "incorrect": "claim → incorrect (groups: hallucination [detail, false_claim]; "
        "co-activation [the substitutions])",
        "thinky": "claim → unverifiable → on topic (group: thinky)",
        "unsettled": "claim → unverifiable → ambiguous topic (group: unsettled)",
        "junk": "claim → unverifiable → off topic (group: junk)",
    }
    for b, title in branch_titles.items():
        lines.append(f"### {title}")
        lines.append("")
        lines += [f"- `{lf.name}`: {lf.definition}" for lf in _LEAVES if lf.branch == b]
        lines.append("")
    lines += ["## Precedence", "", PRECEDENCE_INTRO, ""]
    for n in RULE_ORDER:
        r = RULES[n]
        head, *bullets = r.body
        lines.append(f"{n}. **{r.title}** {head}".rstrip())
        lines += [f"   - {b}" for b in bullets]
    lines += ["", "## Detail rule", "", DETAIL_RULE_INTRO]
    lines += [f"- {d}" for d in DETAIL_ITEMS]
    lines += [*DETAIL_EXTRA, "", DETAIL_CARVE_OUT, ""]
    lines += ["## Tags (orthogonal; they never change the leaf)", ""]
    lines += [f"- `{k}` {v}" for k, v in TAGS.items()]
    return "\n".join(lines)
