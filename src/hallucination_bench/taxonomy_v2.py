"""THE claim-map taxonomy (v2): leaves, groups, definitions, precedence, tags and decisions.

Plan 0002 (``plans/0002-claim-map-taxonomy-v2.md``). This module is the single source of truth:
the claim-map judge prompt (``judges/claimmap_v2.py``) is rendered from it by :func:`render`,
scoring and calibration read its enums, and tests assert the rendered prompt contains every
leaf definition (:func:`definitions`).

Every segment of a readout lands in exactly one of the 22 judge leaves, which fall into 8
groups under three top branches (non-claim / claim about the text / supplementary claim; plan
rev 13, option A). Two code-assigned pseudo-leaves (``unknown_incorrect``, ``failed``) are kept
as separate rows outside the groups. Each leaf maps to exactly one **verdict** (``VERDICTS``),
the headline layer of the report.

The prompt is rendered in two modes: single-call (the default; no established v5c spans, so
rule 2 and the ``dispute`` tag are left out) and ``anchored`` (``--anchor v5c``: plan rev 12
behaviour, with established spans).

**Open decisions** (``OPEN_DECISIONS``) are pending Camila's answer. Each has a ``default`` and
the text the prompt renders for every option; change ``CHOSEN`` (one line per decision) to switch
an option. The rendered prompt, and so every ``:M`` cache fingerprint, follows ``CHOSEN``; bump
``TAXONOMY_VERSION`` when you change it so the cached labels are re-queued knowingly.
"""

from dataclasses import dataclass

TAXONOMY_VERSION = "v2-r15"

# ---------------------------------------------------------------- groups

HALLUCINATION = "hallucination"
CO_ACTIVATION = "co_activation"
CORRECT = "correct"
SUPPLEMENTARY = "supplementary"
THINKY = "thinky"
UNSETTLED = "unsettled"
JUNK = "junk"
NON_CLAIM = "non_claim"
GROUPS: tuple[str, ...] = (
    HALLUCINATION,
    CO_ACTIVATION,
    CORRECT,
    SUPPLEMENTARY,
    THINKY,
    UNSETTLED,
    JUNK,
    NON_CLAIM,
)
GROUP_TITLES: dict[str, str] = {
    HALLUCINATION: "hallucinations: wrong details and false claims",
    CO_ACTIVATION: "plausible co-activation: substitutions of a related object",
    CORRECT: "correct content about the text (fidelity only)",
    SUPPLEMENTARY: "supplementary world claims the conversation does not settle",
    THINKY: (
        "thinky unverifiable content: unrealised predictions, the model's own state and beliefs, "
        "other unsettled claims about the text"
    ),
    UNSETTLED: "unsettled: ambiguous topic",
    JUNK: "junk: off-topic claims",
    NON_CLAIM: "non-claims",
}

# code-assigned pseudo-leaves (never in the judge's enum)
UNKNOWN_INCORRECT = "unknown_incorrect"
FAILED = "failed"
PSEUDO_LEAVES: tuple[str, ...] = (UNKNOWN_INCORRECT, FAILED)
ROWS: tuple[str, ...] = (*GROUPS, *PSEUDO_LEAVES)  # the identity holds over these 10 rows

# ---------------------------------------------------------------- leaves


@dataclass(frozen=True)
class Leaf:
    name: str
    branch: str  # non_claim | correct | incorrect | thinky | supplementary | unsettled | junk
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
    # incorrect
    Leaf(
        "detail",
        "incorrect",
        HALLUCINATION,
        "[hallucination] a wrong specific with no concept of its own (see the Detail rule), "
        "that the conversation obviously contradicts.",
    ),
    Leaf(
        "false_claim",
        "incorrect",
        HALLUCINATION,
        "[hallucination] a false assertion about the right object that the conversation "
        "obviously contradicts: reversed outcome or stance, wrong agent (who did what to whom), "
        "false relation or cause, a wrong real-world quantity, or an event, fact, prediction or "
        "continuation about the conversation that the text rules out.",
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
        "unrealised_option",
        "thinky",
        THINKY,
        "[alternative] the readout fills **the same open-ended slot** that the response fills "
        "next (the next item of a list, an example, an illustration, or a word or phrase choice "
        "for the same move) with a different choice that the text does not contradict; "
        "evidence that the model is choosing among options at a decision point.",
    ),
    Leaf(
        "unrealised_prediction",
        "thinky",
        THINKY,
        "[alternative] a different move or a different answer (framed as anticipation or "
        "written as continuation text) that the response does not make, but that the text does "
        "not contradict.",
    ),
    Leaf(
        "unsupported_assertion",
        "thinky",
        THINKY,
        "an event, fact or specific asserted about the conversation (the user, the setting, "
        "people or things in it) that the text never addresses, so it neither supports nor "
        'contradicts it: "the user is a teacher"; "her sister Maria"; "the user, Sarah, …" '
        "when nobody is named.",
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
        "other_on_topic",
        "thinky",
        THINKY,
        "any other on-topic claim about the text that the conversation cannot settle and that "
        "is not a world claim (for example a hedged inference about the user or setting, rule "
        "12). Contested or uncheckable world claims are `supplementary_unsettled`, not this.",
    ),
    # supplementary (on-topic world claims the conversation is silent on)
    Leaf(
        "supplementary_true",
        "supplementary",
        SUPPLEMENTARY,
        "an on-topic world claim that the conversation does not settle (it neither states, "
        "entails nor contradicts it), and that is true in the world.",
    ),
    Leaf(
        "supplementary_false",
        "supplementary",
        SUPPLEMENTARY,
        "an on-topic world claim that the conversation does not settle, and that you believe is "
        "false in the world; the model may really hold it.",
    ),
    Leaf(
        "supplementary_unsettled",
        "supplementary",
        SUPPLEMENTARY,
        "an on-topic world claim that the conversation does not settle, and that world "
        "knowledge cannot settle either: contested, doubtful, a matter of taste, or "
        "uncheckable.",
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
SUPPLEMENTARY_LEAVES: tuple[str, ...] = tuple(n for n, g in GROUP_OF.items() if g == SUPPLEMENTARY)
NON_CLAIM_LEAVES: tuple[str, ...] = tuple(n for n, g in GROUP_OF.items() if g == NON_CLAIM)
ALL_LEAVES: tuple[str, ...] = (*LEAF_NAMES, *PSEUDO_LEAVES)

assert len(LEAF_NAMES) == 22 and len(GROUPS) == 8

# gold written before plan rev 13 uses these leaf names; loaders map them mechanically
# fallback renames (loaders map them and warn). Rev 14 mapped `unrealised_prediction` here; rev 15
# reinstates it as a leaf, so there are none
FALLBACK_RENAMES: dict[str, str] = {}
LEAF_RENAMES: dict[str, str] = {
    "world_true": "supplementary_true",
    "possible_belief": "supplementary_false",
}

# ---------------------------------------------------------------- verdicts (plan rev 13)

V_HALLUCINATION = "hallucination"
V_PROBE = "probe_candidate"
V_ALTERNATIVE = "alternative"
V_CORRECT = "correct"
V_UNRESOLVED = "unresolved"
V_JUNK = "junk"
V_NON_CLAIM = "non_claim"
VERDICTS: dict[str, tuple[str, ...]] = {
    V_HALLUCINATION: ("detail", "false_claim"),
    V_PROBE: (
        "substitution_sibling",
        "substitution_category",
        "supplementary_false",
        "model_state",
        "ambiguous_topic",
    ),
    V_ALTERNATIVE: ("unrealised_option", "unrealised_prediction"),
    V_CORRECT: ("restates_prefix", "stated_after", "entailed", "supplementary_true"),
    V_UNRESOLVED: (
        "other_on_topic",
        "unsupported_assertion",
        "supplementary_unsettled",
        UNKNOWN_INCORRECT,
    ),
    V_JUNK: ("off_topic",),
    V_NON_CLAIM: NON_CLAIM_LEAVES,
}
VERDICT_OF: dict[str, str] = {
    **{lf: v for v, leaves in VERDICTS.items() for lf in leaves},
    FAILED: FAILED,
}
VERDICT_ROWS: tuple[str, ...] = (*VERDICTS, FAILED)  # shares sum to 1 over these 8 rows
PROBE_CANDIDATE_LEAVES: tuple[str, ...] = VERDICTS[V_PROBE]
assert sorted(lf for leaves in VERDICTS.values() for lf in leaves) == sorted(
    (*LEAF_NAMES, UNKNOWN_INCORRECT)
)


def group_of(leaf: str) -> str:
    return GROUP_OF.get(leaf, FAILED)


def verdict_of(leaf: str) -> str:
    return VERDICT_OF.get(leaf, FAILED)


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
        'replace ("the user, Sarah", "her 3 kids"): detail, an invented fact (false_claim), or '
        "(rev 15) unsupported_assertion since the text does not contradict it?",
        "unsupported_assertion",
        {
            "unsupported_assertion": (
                "An **invented** specific with nothing in the conversation for it to replace "
                '("the user, Sarah, …" when nobody is named; "her 3 kids") is NOT detail: the '
                "text does not contradict it, so it is `unsupported_assertion` (rule 10)."
            ),
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
        "rule 10 and unrealised_option (rev 14: narrow; other unrealised predictions are "
        "incorrect)",
    ),
    (
        "unrealized predictions [are] e.g. possible continuations like if the model is "
        "deciding what to say (this should mostly apply to olens) so this category should be "
        "quite small; things the model says will happen that dont happen … are hallucinations",
        "rev 14: unrealised_option (same open-ended slot only); rules 3, 9, 10",
    ),
    (
        "[a wrong arbitrary / user-supplied name] should be detail hallucination; [the model "
        "naming itself as another real AI is] actually interesting and deserves probing",
        "rev 14: rule 8 name anchors (Miss Wang for Miss Li -> detail; 我是 ChatGPT for Qwen "
        "-> substitution_sibling)",
    ),
    (
        "for substitution i think these could plausibly be things the model is really thinking "
        "about…",
        "the substitutions form the co-activation group (not a hallucination); pair logged",
    ),
    ("the invented event is a false claim. it's verifiable.", "rule 10 (superseded by rev 15)"),
    (
        "the only things that are considered hallucinations should be things that are "
        "verifiable and obviously contradictory to the text",
        "rev 15: HALLUCINATION_GATE; unsupported_assertion; unrealised_prediction reinstated; "
        "the alternative verdict",
    ),
    (
        "world checkable claims that are on topic and correct should be verifiable correct, "
        "claims that are on topic and incorrect are either unverifiable because we are not sure "
        "whether the model is actually thinking about this or not or maybe they are just "
        "verifiably incorrect because they contradict with the following text",
        "supplementary_true; contradicted -> incorrect; silent -> supplementary_false; stated "
        "-> rule 6 (rev 13: the test is settlement, not subject)",
    ),
    (
        "option A (rev 13): three top branches, non-claim / claim about the text / "
        "supplementary claim",
        "the supplementary group, rule 6's settlement test, rule 14, VERDICTS",
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
    'their claim segment. An invented specific ("her 3 kids") is one segment, with no separate '
    "existence claim.",
)

# ---------------------------------------------------------------- precedence


@dataclass(frozen=True)
class Rule:
    number: int
    title: str
    body: tuple[str, ...]  # first entry: the rule; the rest: bullets


HALLUCINATION_GATE = (
    "**The hallucination gate.** A segment is incorrect (`detail`, `false_claim`, "
    "`substitution_sibling`, `substitution_category`) **only if the conversation obviously "
    "contradicts it**: the prompt, the response before the marker, or the response after the "
    "marker states something incompatible with the segment. It is **not** enough that a reader "
    "would have to infer the incompatibility, or that the text simply never mentions the claim."
)

SETTLEMENT_RULE = (
    "**The test is settlement, not subject.** An on-topic world claim is **about the text** if "
    "the conversation settles it (states it, entails it, or contradicts it). It is "
    "**supplementary** if the conversation is silent on it, even when it concerns the text's "
    'subject. For example, "Picasso painted Guernica" when the text only discusses the Blue '
    "Period is supplementary."
)

LENS_FRAMING_RULE = (
    "A summary's lens framing, such as \"The model is representing / anticipating / focused on "
    '/ gearing up for X", is **framing** (`filler`). The X it attributes is judged as an '
    "ordinary claim about the text or as a prediction. It is **never `model_state`**, which is "
    "reserved for readouts that make claims about the model's beliefs or intent as such."
)


def _rules(anchored: bool = False) -> dict[int, Rule]:
    est_clause = ", or if it overlaps an established span" if anchored else ""
    rules = {
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
                "implicit prediction of the response and is judged by rules 7, 8 and 10, in "
                "this order:",
                "`restates_prefix` or `stated_after` if the conversation says it **in any "
                "wording** (a paraphrase, another person or framing; tag `world_false` if it is "
                "false in the world, per rule 6). This beats both unrealised leaves;",
                "incorrect if the text obviously contradicts it, before or after the marker "
                "(leaf by rule 8);",
                "`unrealised_option` if it fills the same open-ended slot the response fills "
                "next (the next list item, an example, a word choice for the same move) with a "
                "different choice;",
                "`unrealised_prediction` if it is a different move or a different answer that "
                "the response does not make.",
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
                "`stated_after` if realised in any wording; if not, rule 10 (incorrect only if "
                "the text contradicts it, else `unrealised_option` / `unrealised_prediction`).",
                'Descriptions of format, genre or setting ("Q&A on an educational platform") '
                "are ordinary claims about the conversation: an invented setting **asserted "
                "flatly** is `unsupported_assertion`, or `false_claim` if the text contradicts it "
                "(rule 10); a hedged one follows rule 12.",
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
                + " (Predictions are excepted: a claim framed as anticipation, or written as "
                "continuation text, follows rules 3 and 10: what the response goes on to say "
                "settles it.) **A "
                "non-prediction world claim is incorrect only if the conversation contradicts "
                f"it**{est_clause}.",
                SETTLEMENT_RULE,
                "An on-topic world claim on which the conversation is silent goes to the "
                "**supplementary** branch, whatever its world truth (leaf by rule 14: "
                "`supplementary_true`, `supplementary_false` or `supplementary_unsettled`). It "
                "is never `correct` and never incorrect, even if it concerns the text's subject.",
            ),
        ),
        7: Rule(
            7,
            "Inside correct.",
            ("`restates_prefix` > `stated_after` > `entailed`.",),
        ),
        8: Rule(
            8,
            "Inside incorrect. Apply in this order:",
            (
                "Only a segment that passes the hallucination gate (the conversation obviously "
                "contradicts it) is incorrect.",
                "Step 1. A wrong named object that has its own identity, in place of the right "
                "one -> substitution (`substitution_sibling` if it is of the same class, "
                '`substitution_category` if of a different category). "Its own identity" '
                "means a real-world entity (person, work, place, organisation, product) or "
                "another distinct entity that appears in the text. A fictional or arbitrary name "
                "for the right referent never has its own identity: it is step 2 (`detail`), "
                "even for a protagonist (Nora for the story's Mia; Steve for Joe), and a wrong "
                "arbitrary or user-supplied name for someone is `detail` (the user addressed "
                '"Miss Li" and the readout says "Miss Wang"). The model naming itself as another '
                'real AI system ("我是 ChatGPT" where it is Qwen) is `substitution_sibling`: a '
                "real product has its own identity. " + decision_text("subject_name"),
                "Step 2. Anything matching the Detail rule list -> `detail`. The Detail rule's "
                "carve-outs (real-event quantities, "
                "counts that flip the claim) go on to step 3.",
                "Step 3. Any other false assertion about the right object -> `false_claim`.",
            ),
        ),
        9: Rule(
            9,
            "Inside the thinky branch (unverifiable, on topic).",
            (
                "`unrealised_option` > `unrealised_prediction` > `model_state` > "
                "`unsupported_assertion` > `other_on_topic`.",
            ),
        ),
        10: Rule(
            10,
            "Framing.",
            (
                HALLUCINATION_GATE,
                "A concrete event, fact or specific **asserted** about the conversation: "
                "incorrect if the text obviously contradicts it (leaf by rule 8); if the text "
                'never addresses it, `unsupported_assertion` ("the user is a teacher"; "her '
                'sister Maria"; "the user, Sarah, …" when nobody is named).',
                "A prediction or continuation (framed as anticipation or written as "
                "continuation text) that the response realises **in any wording** is "
                "`stated_after`: a paraphrase, a first-person version or a different framing of "
                'what the response says counts ("Alibaba Group engineers who created me" when '
                "the response says it was created by Alibaba's engineers). This beats both "
                "unrealised leaves.",
                "A prediction or continuation that the text obviously contradicts, before or "
                "after the marker, is incorrect (leaf by rule 8: `detail` if only a specific "
                "value or wording differs within the same content, otherwise usually "
                "`false_claim`).",
                "Otherwise, an unrealised prediction or continuation is an **alternative**:",
                "`unrealised_option`: the readout fills the same open-ended slot that the "
                "response fills next (the next item of a list, an example, an illustration, or a "
                "word or phrase choice for the same move) with a different choice. Example: at "
                '"What kind of performance was it (music, theater, dance, ⟦READ⟧ public '
                'speaking)?", the readout "Speech or storytelling" is an option.',
                "`unrealised_prediction`: a different move or a different answer that the "
                "response does not make but the text does not contradict (\"Here's a guide "
                'tailored to public speaking" when the response only asks questions). A '
                "different answer that the response's actual answer rules out is incorrect.",
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
                "A flat assertion of an unsupported fact about the user or setting that the "
                'text never addresses is `unsupported_assertion` (rule 10). For example, "the '
                'user is a child" or "(Scratch-like community)" stated as fact. '
                + decision_text("hedged_inference")
                + " A flat or hedged claim that the conversation **obviously contradicts** is "
                "incorrect (rule 8 picks the leaf).",
            ),
        ),
        13: Rule(
            13,
            "On-topic fallback.",
            (
                "An on-topic claim (rule 5) that rules 6, 10 and 12 have not placed, and that "
                "the conversation does not settle: a world claim goes to the **supplementary** "
                "branch (rule 6; leaf by rule 14), including contested or uncheckable world "
                "claims; any other claim goes to the **thinky** branch (leaf by rule 9, else "
                "`other_on_topic`). Never the incorrect branch: that needs an obvious "
                "contradiction.",
            ),
        ),
        14: Rule(
            14,
            "Inside supplementary (on-topic world claims the conversation does not settle).",
            (
                "`supplementary_true` if it is true in the world; `supplementary_false` if you "
                "believe it is false in the world; `supplementary_unsettled` if world knowledge "
                "cannot settle it (contested, doubtful, a matter of taste, or uncheckable).",
            ),
        ),
    }
    if not anchored:
        del rules[2]
    return rules


RULES: dict[int, Rule] = _rules()  # single-call (no rule 2)
RULES_ANCHORED: dict[int, Rule] = _rules(anchored=True)
BRANCH_RULE_ORDER_ANCHORED: tuple[int, ...] = (2, 1, 3, 4, 5, 6, 10, 11, 12, 13)
BRANCH_RULE_ORDER: tuple[int, ...] = tuple(n for n in BRANCH_RULE_ORDER_ANCHORED if n != 2)
LEAF_RULE_ORDER: tuple[int, ...] = (7, 8, 9, 14)
RULE_ORDER: tuple[int, ...] = (*BRANCH_RULE_ORDER, *LEAF_RULE_ORDER)
RULE_ORDER_ANCHORED: tuple[int, ...] = (*BRANCH_RULE_ORDER_ANCHORED, *LEAF_RULE_ORDER)
PRECEDENCE_INTRO = (
    "Rules 1, 3–6 and 10–13 **select the branch** (non-claim, correct, incorrect, thinky, "
    "supplementary, unsettled, junk). Apply them first, top-down in the order listed; the first "
    "one that fits wins. Rules 7–9 and 14 **order the leaves within a branch** and apply only "
    "after the branch is fixed."
)
PRECEDENCE_INTRO_ANCHORED = (
    "Rules 1–6 and 10–13 **select the branch** (non-claim, correct, incorrect, thinky, "
    "supplementary, unsettled, junk). Apply them first, top-down in the order listed; the first "
    "one that fits wins. Rule 2 (established spans) is checked before rule 1, so you dispute an "
    "established span rather than call it a non-claim. Rules 7–9 and 14 **order the leaves "
    "within a branch** and apply only after the branch is fixed."
)

# ---------------------------------------------------------------- detail rule

DETAIL_RULE_INTRO = "`detail` covers wrong specifics with no concept representation of their own:"


def _detail_items() -> tuple[str, ...]:
    items = [
        "arbitrary or fictional names;",
        "incidental numbers, dates or counts in fiction or scaffolding;",
        "quoted wording or heading text;",
    ]
    if CHOSEN.get("invented_specific") == "detail":
        items.append(decision_text("invented_specific"))
    return tuple(items)


DETAIL_ITEMS: tuple[str, ...] = _detail_items()
DETAIL_EXTRA: tuple[str, ...] = (
    () if CHOSEN.get("invented_specific") == "detail" else (decision_text("invented_specific"),)
)
DETAIL_CARVE_OUT = (
    'Quantities and dates of **real events**, and counts that **flip the claim** ("0 errors" '
    "for 3), are `false_claim` **when the conversation contradicts them**; if it is silent they "
    'are `supplementary_false` (rule 6). Wording and digits are treated alike, so "no errors" and '
    '"0 errors" both count as `false_claim`.'
)

# ---------------------------------------------------------------- tags

TAGS: dict[str, str] = {
    "hedged": (
        "(bool): a short hedged list or disjunction of at most 4 options; it is one segment. "
        "Hit (the realised or stated option is in the list): it gets the correct leaf it would "
        "get on its own. Miss: it is judged as the prediction or claim it is (rule 10: "
        "incorrect if the text contradicts it, else `unrealised_option` / "
        "`unrealised_prediction`). A longer scattershot "
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
    "probe_target": (
        "(string, probe-candidate segments only): on every `substitution_sibling`, "
        "`substitution_category`, `supplementary_false`, `model_state` and `ambiguous_topic` "
        'segment, a short phrase naming the concept a probe should test for ("Germany"; '
        '"this is an evaluation"; "ChatGPT"). For a substitution it names the object the '
        "readout says (the `pair` keeps the actual one). Null for every other leaf."
    ),
}
ANCHORED_ONLY_TAGS: tuple[str, ...] = ("dispute",)


def tags(anchored: bool = False) -> dict[str, str]:
    return {k: v for k, v in TAGS.items() if anchored or k not in ANCHORED_ONLY_TAGS}


# ---------------------------------------------------------------- render


TREE_INTRO = (
    "The tree has three top branches: **non-claim**, **claim about the text** and "
    "**supplementary claim**. The topic gate (rule 5) runs first on every claim and routes "
    "off-topic and ambiguous claims to `off_topic` / `ambiguous_topic`. An on-topic claim is "
    "then about the text or supplementary (rule 6). A claim about the text is incorrect only "
    "if the conversation obviously contradicts it (the hallucination gate, rule 10)."
)


def _mode(anchored: bool) -> tuple[dict[int, Rule], tuple[int, ...], str]:
    if anchored:
        return RULES_ANCHORED, RULE_ORDER_ANCHORED, PRECEDENCE_INTRO_ANCHORED
    return RULES, RULE_ORDER, PRECEDENCE_INTRO


def definitions(anchored: bool = False) -> list[str]:
    """Every label and definition string the rendered prompt must contain (tests)."""
    rules, _order, intro = _mode(anchored)
    out: list[str] = [intro, TREE_INTRO, DETAIL_RULE_INTRO, DETAIL_CARVE_OUT, *UNITS_TEXT]
    for lf in _LEAVES:
        out += [f"`{lf.name}`", lf.definition]
    for r in rules.values():
        out += [r.title, *(b for b in r.body if b)]
    out += [*DETAIL_ITEMS, *DETAIL_EXTRA]
    for k, v in tags(anchored).items():
        out += [f"`{k}`", v]
    return out


BRANCH_TITLES: dict[str, str] = {
    "non_claim": "non-claim (group: non_claim)",
    "unsettled": "claim → topic gate, ambiguous topic (group: unsettled)",
    "junk": "claim → topic gate, off topic (group: junk)",
    "correct": "claim about the text → correct (group: correct)",
    "incorrect": "claim about the text → incorrect (groups: hallucination [detail, "
    "false_claim]; co-activation [the substitutions])",
    "thinky": "claim about the text → unverifiable (group: thinky)",
    "supplementary": "supplementary claim: an on-topic world claim the conversation does not "
    "settle (group: supplementary)",
}


def render(anchored: bool = False) -> str:
    """The taxonomy as prompt text (markdown); ``anchored`` adds rule 2 and the dispute tag."""
    rules, order, intro = _mode(anchored)
    lines: list[str] = [f"# Claim map taxonomy ({TAXONOMY_VERSION})", "", "## Units", ""]
    lines += [f"- {t}" for t in UNITS_TEXT]
    lines += [
        "",
        f"## The tree: {len(LEAF_NAMES)} leaves in {len(GROUPS)} groups",
        "",
        "Every segment gets exactly one leaf. " + TREE_INTRO,
        "",
        SETTLEMENT_RULE,
        "",
    ]
    for b, title in BRANCH_TITLES.items():
        lines.append(f"### {title}")
        lines.append("")
        lines += [f"- `{lf.name}`: {lf.definition}" for lf in _LEAVES if lf.branch == b]
        lines.append("")
    lines += ["## Precedence", "", intro, ""]
    for n in order:
        r = rules[n]
        head, *bullets = r.body
        lines.append(f"{n}. **{r.title}** {head}".rstrip())
        lines += [f"   - {b}" for b in bullets]
    lines += ["", "## Detail rule", "", DETAIL_RULE_INTRO]
    lines += [f"- {d}" for d in DETAIL_ITEMS]
    lines += [*DETAIL_EXTRA, "", DETAIL_CARVE_OUT, ""]
    lines += ["## Tags (orthogonal; they never change the leaf)", ""]
    lines += [f"- `{k}` {v}" for k, v in tags(anchored).items()]
    return "\n".join(lines)
