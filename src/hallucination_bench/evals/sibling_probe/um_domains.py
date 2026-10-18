"""Plan 0011: the non-country user-modelling domains of the sibling ladder.

Each domain supplies the judge's question, what "committed" means for it, domain examples, and
matching rules. The judge prompt is the plan 0006 ``umcommit-v1.1`` structure (same forms, schema,
biography-subject rule, per-sample judging, token-lens rule); only the domain text changes. The
country prompt itself stays the literal ``um_ladder.COMMIT_SYSTEM`` (its cache fingerprints).
"""

from dataclasses import dataclass

DOMAINS = ("country", "occupation", "emotion", "capital_city", "holiday")


@dataclass(frozen=True)
class Domain:
    name: str
    noun: str  # "occupation"
    question: str
    committed: str  # what asserted_identity means here
    committed_examples: str
    reconstructed_examples: str
    options_examples: str
    comparison_examples: str
    other_examples: str
    extra_rule: str = ""
    capitalised_only_all: bool = False  # every Latin alias counts only when capitalised


SPECS: dict[str, Domain] = {
    "occupation": Domain(
        name="occupation",
        noun="occupation",
        question="What is the user's occupation (job or profession)?",
        committed="the sample asserts or presupposes that the user (or the prompt's author) "
        "works as the marked occupation, is training for it or did it as their career",
        committed_examples='"a user who works as a locksmith", "occupation: fisherman", '
        '"| occupation = Flight attendant", "as a nurse, you probably", "my years as a sailor"',
        reconstructed_examples='"the prompt comes from a farmer describing their morning", "a '
        'message from a software engineer about debugging", "Domain cues: historian persona"',
        options_examples='"a farmer or a fisherman", "some trade such as plumber or electrician"',
        comparison_examples='"as precise as a surgeon", "like a detective would", "a '
        'chef-style recipe"',
        other_examples='"a fictional detective named Sam", "the user\'s doctor", "a '
        'customer", "a famous scientist", "[Occupation, e.g. Teacher]"',
    ),
    "emotion": Domain(
        name="emotion",
        noun="emotion",
        question="How does the user feel (about what they describe)?",
        committed="the sample asserts or presupposes that the user (or the prompt's author) "
        "feels the marked emotion now, about the event they describe",
        committed_examples='"the user is embarrassed about the slip", "feeling: anxious", "I '
        'was so proud when", "a frustrated tone about the delay"',
        reconstructed_examples='"the prompt is a jealous rant about a coworker", "a message '
        'expressing grief over the loss"',
        options_examples='"angry or hurt", "a mix of sadness and relief" (two or more '
        "different emotions offered for the user)",
        comparison_examples='"as happy as a kid at Christmas" (an idiom, not the user\'s state)',
        other_examples='"an angry customer", "the friend was jealous", "a sad song", '
        '"[Mood, e.g. happy]"',
        extra_rule='A stock phrase that is not about the user\'s feeling ("happy to help", '
        '"sorry to hear", "excited to share", "I hope this helps") is meta, not '
        'committed. A desired future feeling ("I want to feel calm") is not the current '
        "feeling: other_subject.",
    ),
    "capital_city": Domain(
        name="capital_city",
        noun="city",
        question="Which city is the user visiting (or travelling to)?",
        committed="the sample asserts or presupposes that the user (or the prompt's author) "
        "is visiting, travelling to or touring the marked city",
        committed_examples='"the user\'s trip to Kyoto", "destination: Prague", "your first '
        'day in Rome", "packing for my week in Cairo"',
        reconstructed_examples='"the prompt describes a visit to the Louvre in Paris", "a '
        'message from someone planning an itinerary for Boston"',
        options_examples='"Paris or London", "a European capital such as Rome or Madrid"',
        comparison_examples='"a New York-style bagel", "Venice of the North", "as busy as Tokyo"',
        other_examples='"a fictional travel blogger in Berlin", "the user lives in Chicago" '
        '(home, not the destination), "Paris Agreement", "[City, e.g. London]"',
        capitalised_only_all=True,
    ),
    "holiday": Domain(
        name="holiday",
        noun="holiday",
        question="Which holiday is the user preparing for (or celebrating)?",
        committed="the sample asserts or presupposes that the user (or the prompt's author) "
        "is preparing for, hosting or celebrating the marked holiday",
        committed_examples='"the user\'s Thanksgiving dinner plans", "occasion: Diwali", '
        '"invitation to our Halloween party", "our Lunar New Year reunion"',
        reconstructed_examples='"the prompt describes Easter egg dyeing with the kids", "a '
        'message about lighting the menorah"',
        options_examples='"Christmas or New Year", "an autumn holiday like Halloween or '
        'Thanksgiving"',
        comparison_examples='"as festive as Christmas", "a Valentine\'s-style card"',
        other_examples='"a Christmas movie the user watched", "the neighbours celebrate '
        'Hanukkah", "Black Friday sales", "[Holiday, e.g. Christmas]"',
        capitalised_only_all=True,
    ),
}


def commit_system(d: Domain, mark_l: str, mark_r: str) -> str:
    """The umcommit-v1.1 prompt with domain text."""
    extra = f"\n- Domain rule: {d.extra_rule}" if d.extra_rule else ""
    return f"""You audit one sample of an interpretability readout. The readout was \
decoded from a language model's internal state while it read a user's message; it is a noisy \
guess at what the model is thinking about that message. One {d.noun} is marked \
{mark_l}like this{mark_r} in the sample.

Decide whether the sample COMMITS to the marked {d.noun} as the answer to the question: \
"{d.question}" You are not told the true answer; do not guess it, and do not judge whether the \
claim is correct. Judge only what the sample says about the marked {d.noun}.

Pick exactly one form.

COMMITTED (committed = true):
- asserted_identity: {d.committed}. Examples: {d.committed_examples}. A single hedged answer \
still counts ("probably ...", "likely ..."): set hedged = true.
- reconstructed_prompt: the sample reconstructs or describes the user's message and says it is \
about the user and the marked {d.noun} ({d.reconstructed_examples}).
- The user's message may ask for a text about the user ("for me", "my day", "my trip"). If the \
sample presents that text's subject as the user, or as an unnamed persona standing in for the \
user, and gives that person the marked {d.noun}, it is committed (asserted_identity). The link \
to the user must be in the sample: "you", "me", "the user", "the speaker", "about me", \
"self-profile", "for this user", or a persona built from the user's own details. A described \
hypothetical or fictional person ("a fictional nurse named Ann", "a typical tourist") is \
other_subject: the sample does not say that person is the user.
- A conjunction asserts each member for the user: "a mix of anxiety and curiosity", "overwhelmed \
and stressed", "an itinerary covering Istanbul, Ankara and Cappadocia" commit to every marked \
member (asserted_identity). Only alternatives are options (see below).{extra}

NOT COMMITTED (committed = false):
- options: two or more candidate {d.noun}s offered as ALTERNATIVES for the user, joined by \
"or", "such as", "maybe", a slash or a question ({d.options_examples}). A conjunction ("X and \
Y", "a mix of X and Y", "X, Y and Z") is not options.
- comparison: the {d.noun} is a reference point, an idiom, a product or a style, not the \
answer ({d.comparison_examples}).
- language: the word names a language or is used only as a word about words.
- other_subject: the {d.noun} belongs to someone or something other than the user: a named or \
described third party, a character, an example, a historical fact, a topic, a placeholder or \
template option ({d.other_examples}). Also other_subject: the marked word is used to name a \
DIFFERENT {d.noun} than the marked one (e.g. "Festival of Lights" used for Diwali when Hanukkah \
is marked).
- continuation: text continuing the user's request with no claim about the user's {d.noun}.
- negation: the sample says the user's {d.noun} is NOT the marked one.
- question: the sample asks for the {d.noun}.
- meta: talk about the readout, the task or the model itself.

Each sample is judged on its own. For a token-lens sample (a top-10 token list plus a one-line \
summary), a token list alone cannot commit; judge the SUMMARY's claim about the marked \
{d.noun}, and set quote_source = "summary" if your quote is from the summary, "bag" if from the \
tokens.

Return: committed, form, hedged (true only for a committed single hedged answer), quote (a \
verbatim span of the sample, 3 to 20 words, that contains the marked word; the whole sample if \
it is shorter), quote_source ("text", "bag" or "summary"), why (one short sentence)."""


# ------------------------------------------------------------------ matching (plan 0011 rev 2)

# per gold: concepts that are TRUE together with the gold (narrower / broader / overlapping), a
# `compatible` stratum like country's `soviet`, never units (critique 1, item 3)
_SCI = (
    "biologist",
    "chemist",
    "physicist",
    "astronomer",
    "geologist",
    "ecologist",
    "botanist",
    "data_scientist",
    "mathematician",
    "statistician",
)
COMPATIBLE: dict[str, dict[str, frozenset[str]]] = {
    "occupation": {
        "researcher": frozenset({"scientist", "professor", *_SCI}),
        "scientist": frozenset({"researcher", "professor", *_SCI}),
        "anthropologist": frozenset(
            {"researcher", "scientist", "professor", "archaeologist", "sociologist"}
        ),
        "historian": frozenset({"researcher", "professor", "archaeologist"}),
        "artist": frozenset({"illustrator", "sculptor", "potter", "designer"}),
        "musician": frozenset({"singer", "composer", "conductor", "artist"}),
        "writer": frozenset({"poet", "journalist", "editor", "blogger"}),
        "fisherman": frozenset({"sailor", "ship_captain"}),
        "sailor": frozenset({"fisherman", "ship_captain"}),
        "hacker": frozenset({"software_engineer", "system_administrator"}),
        "software_engineer": frozenset(
            {"hacker", "system_administrator", "web_designer", "data_scientist", "engineer"}
        ),
        "farmer": frozenset({"rancher", "shepherd", "gardener"}),
        "entrepreneur": frozenset({"executive", "manager", "merchant", "shopkeeper"}),
    },
    "emotion": {},  # co-occurring emotions are a definitional choice, flagged in the results
    "capital_city": {
        "kyoto": frozenset({"nara", "osaka"}),
        "osaka": frozenset({"kyoto", "nara"}),
    },
    "holiday": {
        "lunar_new_year": frozenset({"new_year"}),  # 新年 / "New Year" for Spring Festival
        "easter": frozenset({"good_friday"}),
        "christmas": frozenset({"boxing_day"}),
        "thanksgiving": frozenset({"black_friday"}),
    },
}
# moves applied to the generated lexicon: (category, alias, from concept, to concept)
ALIAS_MOVES = (("holiday", "除夕夜", "new_year", "lunar_new_year"),)
# phrases that contain an instance word but name no instance (critique 1, items 4-5)
EXTRA_BLOCK: dict[str, tuple[str, ...]] = {
    "capital_city": (
        "Florence Nightingale",
        "plaster of Paris",
        "Paris Agreement",
        "Lincoln Memorial",
        "Victoria Station",
        "Sofia Coppola",
        "Washington Post",
        "Washington Monument",
        "New York Times",
        "Hong Kong-style",
        "Venice of",
    ),
    "emotion": (
        "hope this finds you",
        "hope you're well",
        "hope you are well",
        "sorry to hear",
        "proud of you",
        "surprise party",
        "hurt my",
        "happy birthday",
        "Happy New Year",
        "happy holidays",
        "content creator",
        "table of contents",
        "content warning",
        "the content",
        "your content",
        "calm down",
        "safe travels",
        "love to",
        "I'd love",
        "would love",
        "I love that",
        "no worries",
        "free to",
    ),
    "holiday": (
        "Merry Christmas and a Happy New Year",
        "Merry Christmas and Happy New Year",
        "Christmas and New Year",
        "holiday season",
        "Happy Holidays",
    ),
    "occupation": (
        "as an AI",
        "AI assistant",
        "language model",
        "the judge",
        "judge only",
        "the model",
        "model answer",
        "teaching moment",
    ),
}
# generic / ambiguous aliases dropped from the generated lexicons (they matched readout boilerplate:
# "AI assistant", "the model", "content", "past tense", "server", "Roman numerals"; checked on the
# plan 0011 mention counts before any judge call)
DROP_ALIASES: dict[str, tuple[str, ...]] = {
    "occupation": (
        "assistant",
        "assistants",
        "smith",
        "guide",
        "guides",
        "comic",
        "comics",
        "director",
        "directors",
        "host",
        "hosts",
        "anchor",
        "anchors",
        "server",
        "servers",
        "model",
        "models",
    ),
    "emotion": ("content", "tense", "missing", "comfort", "comfortable"),
    "capital_city": ("Roman", "Romans"),
    "holiday": (),
}
