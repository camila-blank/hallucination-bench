"""THE claim-map taxonomy (v2): leaves, groups, definitions, precedence, tags and decisions.

Plan 0002 (``plans/0002-claim-map-taxonomy-v2.md``). This module is the single source of truth:
the claim-map judge prompt (``judges/claimmap_v2.py``) is rendered from it by :func:`render`,
scoring and calibration read its enums, and tests assert the rendered prompt contains every
leaf definition (:func:`definitions`).

Plan rev 16 (the lean tree): every segment of a readout lands in exactly one of **12** judge
leaves, which fall into 8 groups. A leaf stays separate only if it carries a distinct verdict or
Camila asked for it by name; every other distinction is a **tag** (recorded, never changes a
verdict): ``reason`` on ``non_claim``, ``source`` on ``correct``, ``same_slot_alternative`` /
``world`` / ``prediction`` on ``unverifiable``. Two code-assigned pseudo-leaves
(``unknown_incorrect``, ``failed``) are kept as separate rows outside the groups. Each leaf maps
to exactly one **verdict** (``VERDICTS``), the headline layer of the report.

The hallucination gate (rev 15) is unchanged: a segment is incorrect only if the text obviously
contradicts it. Predictions get no category of their own (rev 16 rule 1). Rev 17 makes the judge
quote its evidence: every incorrect segment carries ``contradicted_by`` and every ``correct``
segment (except ``source = entailed``) carries ``supported_by``, a verbatim span of the
conversation; code checks the quote and fails closed to ``unverifiable``
(``judges/claimmap_v2.parse``).

The prompt is rendered in two modes: single-call (the default; no established v5c spans, so
rule 2 and the ``dispute`` tag are left out) and ``anchored`` (``--anchor v5c``: plan rev 12
behaviour, with established spans).

**Rev 18 (junk = non-claim):** the judge still answers in the rev-16/17 enum
(``JUDGE_LEAF_NAMES``, 12 leaves, rendered unchanged so ``claimmap-v2.7`` and every cached ``:M``
answer stay valid), and code merges ``non_claim`` and ``off_topic`` into one scoring leaf,
``junk``, right after parsing (``JUDGE_TO_LEAF``; the ``reason`` tag keeps the kind, with
``off_topic`` as a reason). Scoring, reports and calibration use the 11 leaves in ``LEAF_NAMES``,
7 groups and 5 verdicts. **Rev 19**, also after parsing: a segment the judge tags ``hedged`` (a
list or disjunction of options) is scored ``unverifiable`` with ``same_slot_alternative`` and
``hedged`` set, whatever its leaf (``raw_leaf`` keeps the judge's); ``model_state`` gets its own
verdict, ``thinky``, so there are 6 verdicts. **Rev 20**: ``substitution_category`` is the third
hallucination component (with ``false_claim`` and ``detail``); ``substitution_sibling`` stays a
probe candidate. **Rev 21** (plan 0009, scoring only): a ``substitution_sibling`` segment whose
sibling-gate unit ends ``unstable`` or ``probe_unconfirmed`` is scored as the scoring-only leaf
``substitution_sibling_gated`` (verdict hallucination, group unchanged; :func:`effective_leaf`).
``VERDICTS`` and its assert are unchanged, so a consumer that misses the status join falls back
to rev 20.

Leaves of earlier revisions map mechanically to rev 16 (``LEAF_RENAMES`` + ``MERGE_TAGS``); the
calibration loader applies them to old gold.

**Open decisions** (``OPEN_DECISIONS``) each have a ``default`` and the text the prompt renders
for every option; change ``CHOSEN`` (one line per decision) to switch an option. The rendered
prompt, and so every ``:M`` cache fingerprint, follows ``CHOSEN``; bump ``TAXONOMY_VERSION`` when
you change it so the cached labels are re-queued knowingly.
"""

from dataclasses import dataclass
from typing import Any

TAXONOMY_VERSION = "v2-r17"  # rendered into the prompt; rev 18 changes scoring only
SCORING_VERSION = "v2-r21"  # r18 junk merge; r19 hedged/thinky; r20 category -> hallucination;
# r21 (plan 0009): gated sibling substitutions -> the scoring-only leaf SIBLING_GATED

# ---------------------------------------------------------------- groups

HALLUCINATION = "hallucination"
CO_ACTIVATION = "co_activation"
CORRECT = "correct"
SUPPLEMENTARY = "supplementary"
THINKY = "thinky"
UNSETTLED = "unsettled"
JUNK = "junk"
NON_CLAIM = "non_claim"
# the judge's groups (rendered in the prompt); scoring merges non_claim into junk (rev 18)
JUDGE_GROUPS: tuple[str, ...] = (
    HALLUCINATION,
    CO_ACTIVATION,
    CORRECT,
    SUPPLEMENTARY,
    THINKY,
    UNSETTLED,
    JUNK,
    NON_CLAIM,
)
GROUPS: tuple[str, ...] = tuple(g for g in JUDGE_GROUPS if g != NON_CLAIM)
GROUP_TITLES: dict[str, str] = {
    HALLUCINATION: "hallucinations: wrong details and false claims (obviously contradicted)",
    CO_ACTIVATION: "plausible co-activation: substitutions of a related object",
    CORRECT: "correct content about the text (fidelity only)",
    SUPPLEMENTARY: "supplementary world claims the conversation does not settle",
    THINKY: "the model's own state, and unverifiable on-topic content",
    UNSETTLED: "unsettled: ambiguous topic",
    JUNK: "junk: non-claims and off-topic claims (rev 18)",
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
    branch: str  # non_claim | correct | incorrect | thinky | supplementary | unsettled | junk
    group: str
    definition: str


_LEAVES: tuple[Leaf, ...] = (
    Leaf(
        "non_claim",
        "non_claim",
        NON_CLAIM,
        "asserts nothing concrete: generic description, filler (hedges, connectives, "
        "repetition, list scaffolding, echoes of the prompt or response, the framing words of "
        "token-bag summaries), formatting (markup, bullets, empty headings, emoji), junk text "
        "(garbled tokens, stray code, byte artifacts), or statements about the lens output "
        "itself rather than the conversation. Tag `reason`.",
    ),
    Leaf(
        "correct",
        "correct",
        CORRECT,
        "the conversation states it (in the prompt or the response before the marker), the "
        "response realises it after the marker **in any wording**, or the conversation strictly "
        "entails it. Tag `source`.",
    ),
    Leaf(
        "detail",
        "incorrect",
        HALLUCINATION,
        "[hallucination] obviously contradicted, and the wrong part is a specific with no "
        "concept of its own: a name, number, date or wording, or a wrong context word inside an "
        'otherwise on-topic reading ("RLC in the context of **Android**" for an answer about '
        "the 3GPP spec). See the Detail rule.",
    ),
    Leaf(
        "false_claim",
        "incorrect",
        HALLUCINATION,
        "[hallucination] an obviously contradicted proposition or situation: reversed outcome "
        "or stance, wrong agent (who did what to whom), wrong relation or cause, a wrong "
        'real-world quantity, a continuation that assumes a situation the text rules out ("if '
        "it's from a flood or sewage backup\" in a chat about tobacco smell), or a different "
        'answer to the question the response answers ("my better half is your development '
        'environment" when the response says its better half is the team behind it).',
    ),
    Leaf(
        "substitution_sibling",
        "incorrect",
        CO_ACTIVATION,
        "[co-activation] contradicted: a different object of the same class in place of the "
        'right one, including the model naming itself as another real AI system ("我是 '
        'ChatGPT" where it is Qwen).',
    ),
    Leaf(
        "substitution_category",
        "incorrect",
        CO_ACTIVATION,
        "[co-activation] contradicted: a different object of a different category in place of "
        "the right one.",
    ),
    Leaf(
        "model_state",
        "thinky",
        THINKY,
        "about the model's own beliefs, perceptions, state, intent, plans or awareness, "
        "including claims that frame the conversation as a test, an evaluation or a roleplay, "
        "whatever the framing.",
    ),
    Leaf(
        "unverifiable",
        "thinky",
        THINKY,
        'on topic, not contradicted and not settled: uncontradicted predictions ("the spill '
        'has been cleaned up"), unsupported assertions about the conversation ("welcome to '
        'Oakville"), contested or uncheckable world claims, and everything else on topic that '
        "no other leaf takes. Tags `same_slot_alternative`, `world`, `prediction`.",
    ),
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

# The judge's enum (rendered in the prompt and the schema; unchanged since rev 16).
JUDGE_LEAVES: dict[str, Leaf] = {lf.name: lf for lf in _LEAVES}
JUDGE_LEAF_NAMES: tuple[str, ...] = tuple(JUDGE_LEAVES)

# Rev 18: the scoring leaves. `non_claim` and `off_topic` merge into `junk` after parsing.
JUNK_LEAF = Leaf(
    "junk",
    "junk",
    JUNK,
    "asserts nothing concrete (generic, filler, formatting, junk text, lens meta) or is a claim "
    "about neither the conversation nor a subject it raises (off topic). Tag `reason`.",
)
JUDGE_TO_LEAF: dict[str, str] = {"non_claim": "junk", "off_topic": "junk"}
_SCORING_LEAVES: tuple[Leaf, ...] = tuple(
    JUNK_LEAF if lf.name == "non_claim" else lf for lf in _LEAVES if lf.name != "off_topic"
)
LEAVES: dict[str, Leaf] = {lf.name: lf for lf in _SCORING_LEAVES}
LEAF_NAMES: tuple[str, ...] = tuple(LEAVES)
GROUP_OF: dict[str, str] = {
    **{lf.name: lf.group for lf in _SCORING_LEAVES},
    **dict.fromkeys(JUDGE_TO_LEAF, JUNK),  # unmerged judge leaves, if any reach scoring
    UNKNOWN_INCORRECT: UNKNOWN_INCORRECT,
    FAILED: FAILED,
}
BRANCH_OF: dict[str, str] = {lf.name: lf.branch for lf in _SCORING_LEAVES}
INCORRECT_LEAVES: tuple[str, ...] = tuple(n for n, b in BRANCH_OF.items() if b == "incorrect")
SUBSTITUTION_LEAVES: tuple[str, ...] = ("substitution_sibling", "substitution_category")
CORRECT_LEAVES: tuple[str, ...] = tuple(n for n, g in GROUP_OF.items() if g == CORRECT)
THINKY_LEAVES: tuple[str, ...] = tuple(n for n, g in GROUP_OF.items() if g == THINKY)
SUPPLEMENTARY_LEAVES: tuple[str, ...] = tuple(n for n, g in GROUP_OF.items() if g == SUPPLEMENTARY)
JUNK_LEAVES: tuple[str, ...] = ("junk",)
NON_CLAIM_LEAVES: tuple[str, ...] = JUNK_LEAVES  # rev 18: non-claims are junk
ALL_LEAVES: tuple[str, ...] = (*LEAF_NAMES, *PSEUDO_LEAVES)

assert len(JUDGE_LEAF_NAMES) == 12 and len(JUDGE_GROUPS) == 8
assert len(LEAF_NAMES) == 11 and len(GROUPS) == 7

# ---------------------------------------------------------------- leaf tags (rev 16)

REASONS: tuple[str, ...] = ("generic", "filler", "formatting", "junk_text", "lens_meta")  # judge
JUNK_REASONS: tuple[str, ...] = ("off_topic", *REASONS)  # the merged leaf's reasons (rev 18)
SOURCES: tuple[str, ...] = ("prefix", "after", "entailed")
UNVERIFIABLE_FLAGS: tuple[str, ...] = ("same_slot_alternative", "world", "prediction")

# ---------------------------------------------------------------- earlier leaves -> rev 16

# old leaf -> rev-16 leaf (mechanical; loaders apply it to old gold)
LEAF_RENAMES: dict[str, str] = {
    "world_true": "supplementary_true",
    "possible_belief": "supplementary_false",
    **dict.fromkeys(REASONS, "junk"),
    "non_claim": "junk",
    "off_topic": "junk",
    "restates_prefix": "correct",
    "stated_after": "correct",
    "entailed": "correct",
    "other_on_topic": "unverifiable",
    "supplementary_unsettled": "unverifiable",
    "unsupported_assertion": "unverifiable",
    "unrealised_option": "unverifiable",
    "unrealised_prediction": "unverifiable",
}
# the tags the merge carries over (old leaf -> tags on the rev-16 leaf)
MERGE_TAGS: dict[str, dict[str, Any]] = {
    **{r: {"reason": r} for r in REASONS},
    "off_topic": {"reason": "off_topic"},
    "restates_prefix": {"source": "prefix"},
    "stated_after": {"source": "after"},
    "entailed": {"source": "entailed"},
    "supplementary_unsettled": {"world": True},
    "unrealised_option": {"same_slot_alternative": True, "prediction": True},
    "unrealised_prediction": {"prediction": True},
}
# fallback renames (loaders map them and warn); none since rev 15
FALLBACK_RENAMES: dict[str, str] = {}

# ---------------------------------------------------------------- verdicts

V_HALLUCINATION = "hallucination"
V_PROBE = "probe_candidate"
V_THINKY = "thinky"  # rev 19: model_state ("possible thinky stuff"), not a probe candidate
V_CORRECT = "correct"
V_UNRESOLVED = "unresolved"
V_JUNK = "junk"
V_NON_CLAIM = "non_claim"  # rev 16-17 verdict; merged into junk in rev 18
VERDICTS: dict[str, tuple[str, ...]] = {
    # rev 20: substitution_category is the third hallucination component
    V_HALLUCINATION: ("detail", "false_claim", "substitution_category"),
    V_PROBE: (
        "substitution_sibling",
        "supplementary_false",
        "ambiguous_topic",
    ),
    V_THINKY: ("model_state",),
    V_CORRECT: ("correct", "supplementary_true"),
    V_UNRESOLVED: ("unverifiable", UNKNOWN_INCORRECT),
    V_JUNK: ("junk",),
}
VERDICT_OF: dict[str, str] = {
    **{lf: v for v, leaves in VERDICTS.items() for lf in leaves},
    **dict.fromkeys(JUDGE_TO_LEAF, V_JUNK),
    FAILED: FAILED,
}
VERDICT_ROWS: tuple[str, ...] = (*VERDICTS, FAILED)  # shares sum to 1 over these 7 rows
PROBE_CANDIDATE_LEAVES: tuple[str, ...] = VERDICTS[V_PROBE]
assert sorted(lf for leaves in VERDICTS.values() for lf in leaves) == sorted(
    (*LEAF_NAMES, UNKNOWN_INCORRECT)
)


# ---------------------------------------------------------------- rev 21: the sibling gate

# Plan 0009: scoring-only. VERDICTS, VERDICT_ROWS, PROBE_CANDIDATE_LEAVES and the assert above
# are unchanged (rev 20); the gated leaf enters only the derived scoring maps, so a consumer that
# misses the status join keeps exact rev-20 behaviour.
SIBLING_GATED = "substitution_sibling_gated"
SIBLING_HALLUCINATION = frozenset({"unstable", "probe_unconfirmed"})
VERDICT_OF[SIBLING_GATED] = V_HALLUCINATION
GROUP_OF[SIBLING_GATED] = GROUP_OF["substitution_sibling"]  # group shares are unchanged


def effective_leaf(leaf: str, sibling_status: str | None) -> str:
    """``SIBLING_GATED`` iff ``leaf`` is ``substitution_sibling`` and the unit's final status is a
    hallucination (``SIBLING_HALLUCINATION``); else ``leaf`` (a null / missing status keeps it)."""
    if leaf == "substitution_sibling" and sibling_status in SIBLING_HALLUCINATION:
        return SIBLING_GATED
    return leaf


def group_of(leaf: str) -> str:
    return GROUP_OF.get(leaf, FAILED)


def verdict_of(leaf: str) -> str:
    return VERDICT_OF.get(leaf, FAILED)


def is_claim(leaf: str) -> bool:
    """Claim vs non-claim (the claim/non-claim F1): every leaf outside ``junk`` and ``failed``
    asserts something on topic (``unknown_incorrect`` is an established wrong span). Rev 18:
    off-topic claims are junk, so they count as non-claims here."""
    return group_of(leaf) not in (JUNK, NON_CLAIM, FAILED)


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
        "unverifiable since the text does not contradict it (rev 15-16)?",
        "unverifiable",
        {
            "unverifiable": (
                "An **invented** specific with nothing in the conversation for it to replace "
                '("the user, Sarah, …" when nobody is named; "her 3 kids") is NOT detail: the '
                "text does not contradict it, so it is `unverifiable` (rule 10)."
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
                "World-false content that the response itself states is `correct`, with the "
                "tag `world_false = true` (the wrong-bridge case: the response uses the wrong "
                "bridge, and the readout finds it)."
            ),
            "world_wins": (
                "World-false content that the response itself states is still incorrect (leaf "
                "by rule 8), with the tag `world_false = true`."
            ),
        },
    ),
    # 5. hedged inferences about the user or setting
    "hedged_inference": OpenDecision(
        "A hedged, unsupported inference about the user or setting: unverifiable (the lens is "
        "guessing) or incorrect?",
        "unverifiable",
        {
            "unverifiable": (
                'When it is explicitly hedged as an inference ("likely", "probably", "may be", '
                '"seems"), it is `unverifiable` too: the lens is guessing, not asserting.'
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
        "for substitution i think these could plausibly be things the model is really thinking "
        "about…",
        "the substitutions are probe candidates (co-activation), not hallucinations; pair logged",
    ),
    (
        "[a wrong arbitrary / user-supplied name] should be detail hallucination; [the model "
        "naming itself as another real AI is] actually interesting and deserves probing",
        "rule 8 name anchors (Miss Wang for Miss Li -> detail; 我是 ChatGPT for Qwen -> "
        "substitution_sibling)",
    ),
    (
        "the only things that are considered hallucinations should be things that are "
        "verifiable and obviously contradictory to the text",
        "HALLUCINATION_GATE (rules 8 and 10)",
    ),
    (
        "if we have too many categories that kind of overlap then we can't be sure how "
        "consistent the judge is",
        "rev 16: 12 leaves; other distinctions are tags",
    ),
    (
        "most predictions that the text contradicts should be hallucinations",
        "rev 16 rule 10: predictions are judged as ordinary claims",
    ),
    ("unrealized options should be unverifiable", "unverifiable + same_slot_alternative tag"),
    (
        "we should always report false claim and detail and separate metrics",
        "report_v2: false_claim and detail columns next to hallucination in every headline table",
    ),
    (
        "world checkable claims that are on topic and correct should be verifiable correct, "
        "claims that are on topic and incorrect are either unverifiable … or maybe they are just "
        "verifiably incorrect because they contradict with the following text",
        "supplementary_true / supplementary_false when the text is silent; incorrect when it "
        "contradicts (rule 6)",
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

QUOTE_RULE = (
    "**Quote the contradiction.** Every `detail`, `false_claim`, `substitution_sibling` or "
    "`substitution_category` segment must carry `contradicted_by`: a verbatim span of the "
    "conversation (the prompt, or the response before or after the marker) that rules the "
    "segment out. If you cannot quote the text that rules it out, it is not contradicted: label "
    "it `unverifiable`. Code checks every quote against the conversation; a missing or "
    "unfindable quote turns the segment into `unverifiable`."
)
SUPPORT_RULE = (
    "**Quote the support.** Every `correct` segment, except `source = entailed`, must carry "
    "`supported_by`: a verbatim span of the conversation that states or realises it. A missing "
    "or unfindable quote turns the segment into `unverifiable`."
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
    '/ gearing up for X", is **framing** (`non_claim`, reason `filler`). The X it attributes is '
    "judged as an ordinary claim about the text or as a prediction. It is **never "
    "`model_state`**, which is reserved for readouts that make claims about the model's beliefs "
    "or intent as such."
)


def _rules(anchored: bool = False) -> dict[int, Rule]:
    est_clause = ", or if it overlaps an established span" if anchored else ""
    rules = {
        1: Rule(
            1,
            "Non-claim vs claim.",
            (
                "A segment is a claim if it commits to anything concrete about the "
                "conversation, the model, or the world. Anything else is `non_claim`, with the "
                "tag `reason`.",
                'Token-bag summaries: the framing ("the tokens include X") is `non_claim` '
                "(reason `filler`), and a gloss of the tokens is `non_claim` (reason "
                '`lens_meta`). The interpretation, including its connective ("suggesting the '
                'passage is about Y"), is one claim segment, judged as usual. The connective '
                '("suggesting", "indicating", "points to") is **not a hedge**: the '
                "interpretation counts as asserted for rules 10 and 12. " + LENS_FRAMING_RULE,
                "Echoes: a literal copy of the prompt's or the response's words that asserts "
                'nothing (a sample that just repeats "I\'d be delighted to") is `non_claim` '
                "(reason `filler`), even in the first person (this is checked before rule 3). "
                'Any assertion about the content ("the user asks who your partner is") is a '
                "claim.",
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
                "implicit prediction of the response, judged as an ordinary claim (rule 10), in "
                "this order:",
                "`correct` if the conversation says it **in any wording** (a paraphrase, "
                "another person or framing; tag `world_false` if it is false in the world, per "
                "rule 6);",
                "incorrect if the text obviously contradicts it, before or after the marker "
                "(leaf by rule 8);",
                "otherwise `unverifiable`, with `prediction = true` (and "
                "`same_slot_alternative = true` if it fills the same open-ended slot the "
                "response fills with a different choice).",
                "It is **not** `model_state`, even though it is in the first person.",
            ),
        ),
        4: Rule(
            4,
            "The model's own beliefs or intent; test / evaluation / roleplay framing.",
            (
                "Claims about the model's own beliefs or intent, or that frame the conversation "
                "as a test, evaluation, roleplay or deception, are `model_state`. Evaluation "
                "awareness lands here.",
                'Carve-out: a statement of intent that predicts response text ("the model is '
                'about to apologise") is a prediction and goes to rules 3 and 10.',
                'Descriptions of format, genre or setting ("Q&A on an educational platform") '
                "are ordinary claims about the conversation: an invented setting is "
                "`unverifiable`, or `false_claim` if the text contradicts it (rule 10); a hedged "
                "one follows rule 12.",
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
                + " (Predictions are judged like any other claim: what the conversation says, "
                "before or after the marker, settles them; rule 10.) **A world claim is "
                f"incorrect only if the conversation contradicts it**{est_clause}.",
                SETTLEMENT_RULE,
                "An on-topic world claim on which the conversation is silent is supplementary: "
                "`supplementary_true` if it is true in the world, `supplementary_false` if you "
                "believe it is false, and `unverifiable` with `world = true` if world knowledge "
                "cannot settle it (contested, doubtful, a matter of taste, or uncheckable). It "
                "is never `correct` and never incorrect, even if it concerns the text's subject.",
            ),
        ),
        7: Rule(
            7,
            "Correct beats everything uncontradicted.",
            (
                "Content the conversation states, realises after the marker **in any wording** "
                '(a paraphrase, a first-person version, a different framing: "Alibaba Group '
                "engineers who created me\" when the response says Alibaba's engineers built "
                "it), or strictly entails, is `correct`. Tag `source`: `prefix` (prompt or "
                "response before the marker) > `after` (response after the marker) > "
                "`entailed`.",
                SUPPORT_RULE,
            ),
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
                "Step 2. `detail` when the reading is on topic and only a specific is wrong: a "
                'value, name, date, wording, or a context word ("RLC in the context of Android" '
                "for an answer about the 3GPP spec). See the Detail rule.",
                "Step 3. `false_claim` when the proposition or situation itself is wrong: a "
                "reversed outcome, a wrong agent or relation, a continuation that assumes a "
                'situation the text rules out ("if it\'s from a flood or sewage backup" in a '
                "chat about tobacco smell), or a different answer to the question the response "
                'answers ("my better half is your development environment").',
                QUOTE_RULE,
            ),
        ),
        9: Rule(
            9,
            "Inside the thinky branch.",
            (
                "`model_state` (rule 4) > `unverifiable`. `unverifiable` carries three tags: "
                "`prediction` (it is framed as anticipation or written as continuation text), "
                "`same_slot_alternative` (a different choice for the same open-ended slot the "
                'response fills: "3. Baking soda" where item 3 is coffee grounds) and `world` (a '
                "contested or uncheckable world claim).",
            ),
        ),
        10: Rule(
            10,
            "Framing.",
            (
                HALLUCINATION_GATE,
                "Predictions get **no special category**: a prediction or continuation (framed "
                "as anticipation or written as continuation text) is judged as an ordinary "
                "claim. Realised in any wording -> `correct` (rule 7). Obviously contradicted, "
                "before or after the marker -> incorrect (leaf by rule 8). Otherwise "
                '`unverifiable` with `prediction = true` ("the spill has been cleaned up" when '
                "the response never says whether it has).",
                "A concrete event, fact or specific **asserted** about the conversation: "
                "incorrect if the text obviously contradicts it (leaf by rule 8); if the text "
                'never addresses it, `unverifiable` ("welcome to Oakville"; "the user is a '
                'teacher"; "her sister Maria").',
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
                'text never addresses is `unverifiable` (rule 10). For example, "the user is a '
                'child" or "(Scratch-like community)" stated as fact. '
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
                "the conversation does not settle: a true or false world claim is "
                "supplementary (rule 6); anything else is `unverifiable`. Never the incorrect "
                "branch: that needs an obvious contradiction.",
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
LEAF_RULE_ORDER: tuple[int, ...] = (7, 8, 9)
RULE_ORDER: tuple[int, ...] = (*BRANCH_RULE_ORDER, *LEAF_RULE_ORDER)
RULE_ORDER_ANCHORED: tuple[int, ...] = (*BRANCH_RULE_ORDER_ANCHORED, *LEAF_RULE_ORDER)
PRECEDENCE_INTRO = (
    "Rules 1, 3–6 and 10–13 **select the branch** (non-claim, correct, incorrect, thinky, "
    "supplementary, unsettled, junk). Apply them first, top-down in the order listed; the first "
    "one that fits wins. Rules 7–9 **pick the leaf and its tags within a branch** and apply "
    "only after the branch is fixed."
)
PRECEDENCE_INTRO_ANCHORED = (
    "Rules 1–6 and 10–13 **select the branch** (non-claim, correct, incorrect, thinky, "
    "supplementary, unsettled, junk). Apply them first, top-down in the order listed; the first "
    "one that fits wins. Rule 2 (established spans) is checked before rule 1, so you dispute an "
    "established span rather than call it a non-claim. Rules 7–9 **pick the leaf and its tags "
    "within a branch** and apply only after the branch is fixed."
)

# ---------------------------------------------------------------- detail rule

DETAIL_RULE_INTRO = (
    "`detail` covers wrong specifics with no concept representation of their own, when the "
    "text obviously contradicts them and the reading is otherwise on topic:"
)


def _detail_items() -> tuple[str, ...]:
    items = [
        "arbitrary or fictional names, including a wrong user-supplied name;",
        "incidental numbers, dates or counts in fiction or scaffolding;",
        "quoted wording or heading text;",
        "a wrong context word inside an otherwise on-topic reading.",
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
    "reason": (
        "(`non_claim` only, else null): `generic`, `filler` (hedges, connectives, repetition, "
        "scaffolding, echoes, summary framing), `formatting`, `junk_text` or `lens_meta`."
    ),
    "source": (
        "(`correct` only, else null): `prefix` (stated in the prompt or the response before "
        "the marker), `after` (realised after the marker, in any wording) or `entailed`."
    ),
    "same_slot_alternative": (
        "(bool, `unverifiable` only): a different choice for the same open-ended slot the "
        "response fills (the next list item, an example, a word choice for the same move)."
    ),
    "world": "(bool, `unverifiable` only): a contested or uncheckable world claim.",
    "prediction": (
        "(bool, `unverifiable` only): framed as anticipation of the continuation or written as "
        "continuation text."
    ),
    "hedged": (
        "(bool): a short hedged list or disjunction of at most 4 options; it is one segment. "
        "Hit (the realised or stated option is in the list): it gets the leaf it would get on "
        "its own (usually `correct`). Miss: it is judged as the prediction or claim it is (rule "
        "10). A longer scattershot list of more than 4 unrelated options is `non_claim` (reason "
        "`generic`)."
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
    "contradicted_by": (
        "(string, incorrect segments only, else null): the verbatim span of the conversation "
        "(prompt or response, before or after the marker) that rules the segment out; see rule "
        "8. Quote it character for character, without the marker."
    ),
    "supported_by": (
        "(string, `correct` segments only, else null): the verbatim span of the conversation "
        "that states or realises the segment; null only for `source = entailed`; see rule 7."
    ),
    "probe_target": (
        "(string, probe-candidate segments only): on every `substitution_sibling`, "
        "`substitution_category`, `model_state`, `supplementary_false` and `ambiguous_topic` "
        'segment, a short phrase naming the concept a probe should test for ("Germany"; "this '
        'is an evaluation"; "ChatGPT"). For a substitution it names the object the readout says '
        "(the `pair` keeps the actual one). Null for every other leaf."
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
    "if the conversation obviously contradicts it (the hallucination gate, rule 10). Finer "
    "distinctions are tags, which never change the leaf."
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
    "incorrect": "claim about the text → incorrect, only if obviously contradicted (groups: "
    "hallucination [detail, false_claim]; co-activation [the substitutions])",
    "thinky": "claim about the text → the model's state, or unverifiable (group: thinky)",
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
        f"## The tree: {len(JUDGE_LEAF_NAMES)} leaves in {len(JUDGE_GROUPS)} groups",
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
