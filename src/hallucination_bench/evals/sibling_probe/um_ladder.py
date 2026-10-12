"""Plan 0006 (rev 4): the user-modelling sibling ladder, country first.

Pure functions (no I/O except :func:`load_lexicon`, no API); the driver is
``scripts/sibling_probe/um_ladder.py``.

1. **Mentions** (:func:`build_mentions`): a lexicon hit of a country that is not the item's gold
   (nor a gold equivalent) and not named in the user turn. Plan 0005's matcher on plan 0005's
   country lexicon with the Set A patch (:data:`SET_A_LEXICON_PATCH`). One unit per
   (method, item, layer, pos, concept); hits whose alias also occurs in Qwen's response, the
   Soviet aliases on former-Soviet golds and J-lens summary-only hits are strata, never units.
2. **Commitment judge** ``umcommit-v1`` (:func:`render_commit`, :func:`parse_commit`): one call
   per (unit, sample); fail closed to committed; a unit is committed if any sample is
   (:func:`rollup`).
3. **Probe fire rule** (:func:`fires_at`, :func:`fire_label`): the sibling's comparable z beats
   every distractor of the item at that layer in both probe types; n/a rules; neighbour layers.
4. **Label** (:func:`ladder_label`) and the nulls / metrics (:mod:`um_metrics` lives here too).
"""

import copy
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from hallucination_bench.evals.sibling_probe.ladder_lex import (
    CASE_SENSITIVE,
    LEXICON_PATH,
    Hit,
    Matcher,
    has_cjk,
    norm_alias,
)
from hallucination_bench.llm import schema_block
from hallucination_bench.readouts import decode_token, normalize_ws

CATEGORY = "country"
NLA42 = "nla-rl-L42"
METHODS = ("s3d", "mlnla-rl600", "jlens", NLA42)
METHOD_LABEL = {
    "s3d": "s3d O-lens",
    "mlnla-rl600": "ML-NLA RL600",
    "jlens": "J-lens",
    NLA42: "NLA-RL L42",
}
GRID_LAYERS = (20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60)
PROBE_LAYERS = tuple(sorted((*GRID_LAYERS, 42)))

# Set A gold id -> lexicon concept id (all other golds are identical)
GOLD_MAP = {"usa": "united_states", "uk": "united_kingdom"}
SOVIET = "soviet"  # pseudo-concept carrying the Soviet aliases (a stratum on former-Soviet golds)
SOVIET_GOLDS = frozenset({"ukraine", "belarus", "russia"})
# gold -> concepts that are the gold for that item (never mentions)
GOLD_EQUIV: dict[str, frozenset[str]] = {
    "united_kingdom": frozenset({"scotland", "wales", "northern_ireland"}),
}

SET_A_LEXICON_PATCH: dict[str, Any] = {
    # concept -> aliases to add
    "add_en": {
        "south_korea": ["Korea", "Korean", "Koreans"],
        "poland": ["Polish", "Pole"],
    },
    "add_cjk": {"south_korea": ["한국", "朝鲜"]},
    # concept -> aliases to remove
    "remove_en": {
        "russia": ["USSR", "U.S.S.R.", "Soviet Union", "Soviet", "Soviets"],
        "ghana": ["Gold Coast"],
    },
    "remove_cjk": {"north_korea": ["朝鲜"], "russia": ["苏联"]},
    # the soviet pseudo-concept (Russia for non-Soviet golds, a stratum for Soviet ones)
    "soviet": {
        "label": "Soviet Union",
        "en": ["USSR", "U.S.S.R.", "Soviet Union", "Soviet", "Soviets", "SSR"],
        "cjk": ["苏联", "苏维埃"],
        "tokens": [],
    },
    # phrases that contain a country alias but name no country
    "block": [
        "New South Wales",
        "Gold Coast",
        "New England",
        "New Mexico",
        "New Jersey",
        "Indian Ocean",
        "Latin America",
        "Latin American",
        "Latin Americans",
        "South America",
        "South American",
        "South Americans",
        "Central America",
        "Central American",
        "North America",
        "North American",
        "North Americans",
        "Pan-American",
        "Native American",
        "Native Americans",
        "American Indian",
        "Indian Ocean",
    ],
}
# capitals that stay blocked: ambiguous with a common word, a person or another place
CAPITALS_KEEP_BLOCKED = frozenset(
    {"Washington", "Male", "Victoria", "Sofia", "Kingston", "Georgetown", "Nassau"}
)
# Latin aliases (normalised) that only count when capitalised, in text and as J-lens tokens
CAPITALISED_ONLY = frozenset(CASE_SENSITIVE) | frozenset({"pole", "poles", "polish"})
REGIONAL_WORDS = re.compile(
    r"(?<![A-Za-z])(East Asian|Southeast Asian|South Asian|Asian|European|Eastern European|"
    r"Slavic|Latin American|Latino|Latina|Hispanic|African|Middle Eastern|Scandinavian|"
    r"Nordic|Balkan|Caribbean|Pacific Islander)(?![A-Za-z])"
)


def gold_concept(gold: str) -> str:
    return GOLD_MAP.get(gold, gold)


def gold_equivalents(gold: str) -> frozenset[str]:
    g = gold_concept(gold)
    return frozenset({g}) | GOLD_EQUIV.get(g, frozenset())


def patch_lexicon(lexicon: Mapping[str, Any]) -> dict[str, Any]:
    """The plan 0005 lexicon with the Set A patch applied (the input is not modified)."""
    lex = copy.deepcopy(dict(lexicon))
    cat = lex["categories"][CATEGORY]
    con = cat["concepts"]
    for cid, al in SET_A_LEXICON_PATCH["add_en"].items():
        con[cid]["en"] = [*con[cid]["en"], *[a for a in al if a not in con[cid]["en"]]]
    for cid, al in SET_A_LEXICON_PATCH["add_cjk"].items():
        con[cid]["cjk"] = [*con[cid]["cjk"], *[a for a in al if a not in con[cid]["cjk"]]]
    for cid, al in SET_A_LEXICON_PATCH["remove_en"].items():
        con[cid]["en"] = [a for a in con[cid]["en"] if a not in al]
    for cid, al in SET_A_LEXICON_PATCH["remove_cjk"].items():
        con[cid]["cjk"] = [a for a in con[cid]["cjk"] if a not in al]
    con[SOVIET] = copy.deepcopy(SET_A_LEXICON_PATCH["soviet"])
    # capitals name their country in Set A ("grew up in Beijing"): un-block and alias them
    cities: dict[str, str] = {}
    for cid, city in sorted(lex.get("capitals", {}).items()):
        if city in CAPITALS_KEEP_BLOCKED or cid not in con:
            continue
        cities[city] = cid
        if city not in con[cid]["en"]:
            con[cid]["en"] = [*con[cid]["en"], city]
    cat["block"] = [
        b for b in [*cat.get("block", []), *SET_A_LEXICON_PATCH["block"]] if b not in cities
    ]
    lex["set_a_city_aliases"] = cities
    return lex


def load_lexicon(path: Path = LEXICON_PATH) -> dict[str, Any]:
    return patch_lexicon(json.loads(Path(path).read_text()))


class FastMatcher:
    """Plan 0005's :class:`Matcher`, run only on the candidate patterns of a text.

    ``Matcher.find`` runs every alias regex (~1.6k) on every text. Here one combined regex finds
    the first words present in the text and only the patterns starting with them (plus the CJK
    aliases that occur as substrings) are run; candidate resolution (longest first,
    non-overlapping) is the same code path, so hits are identical (tested). On top of it the Set A
    case rule: aliases in :data:`CAPITALISED_ONLY` count only when capitalised (``pole vault``,
    ``polish the floor``, a `` turkey`` / `` china`` token)."""

    def __init__(self, lexicon: Mapping[str, Any]) -> None:
        self.m = Matcher(lexicon, CATEGORY)
        self.city_aliases: dict[str, str] = dict(lexicon.get("set_a_city_aliases", {}))
        self._city_norm = {norm_alias(c) for c in self.city_aliases}
        self._by_first: dict[str, list[Any]] = defaultdict(list)
        self._cjk: list[Any] = []
        for p in self.m._pats:
            if p.rx is None:
                self._cjk.append(p)
            else:
                first = re.split(r"[\s\-]+", p.alias.strip())[0].casefold()
                self._by_first[first].append(p)
        firsts = sorted((re.escape(f) for f in self._by_first), key=len, reverse=True)
        self._pre = re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(firsts) + ")", re.IGNORECASE)

    @property
    def concepts(self) -> Mapping[str, Mapping[str, Any]]:
        return self.m.concepts

    def _candidates(self, text: str) -> list[tuple[int, int, Any, str]]:
        pats = {
            id(p): p
            for f in {m.group(0).casefold() for m in self._pre.finditer(text)}
            for p in self._by_first.get(f, ())
        }
        cands: list[tuple[int, int, Any, str]] = []
        for p in pats.values():
            for m in p.rx.finditer(text):
                cands.append((m.start(), m.end(), p, m.group(0)))
        if has_cjk(text):
            for p in self._cjk:
                i = text.find(p.alias)
                while i >= 0:
                    cands.append((i, i + len(p.alias), p, p.alias))
                    i = text.find(p.alias, i + 1)
        return cands

    def find(self, text: str) -> list[Hit]:
        """Same contract as ``Matcher.find`` (no per-call block, no ambiguous aliases: the
        country category has none), plus the capitalised-only rule."""
        if not text:
            return []
        cands = self._candidates(text)
        cands.sort(key=lambda c: (-(c[1] - c[0]), c[0]))
        taken: list[tuple[int, int]] = []
        hits: list[Hit] = []
        for st, en, p, txt in cands:
            if any(st < te and ts < en for ts, te in taken):
                continue
            taken.append((st, en))
            hits.append(Hit(p.concept, txt, st, en))
        hits.sort(key=lambda h: h.start)
        return [h for h in hits if not self._lowercase_only(h.alias)]

    def _lowercase_only(self, alias: str) -> bool:
        """Capitalised-only aliases (and every capital-city alias) written in lower case."""
        a = alias.strip()
        if a and not has_cjk(a) and not a[0].isupper():
            if norm_alias(a) in self._city_norm:
                return True
        return _lowercase_only_hit(a)

    def token_concept(self, token: str) -> str | None:
        if self._lowercase_only(token.strip()):
            return None
        r = self.m.token_concept(token)
        return r[0] if r else None

    def concepts_in(self, text: str) -> set[str]:
        return {h.concept for h in self.find(text) if h.concept}

    def via_city(self, alias: str) -> bool:
        return alias in self.city_aliases


def _lowercase_only_hit(alias: str) -> bool:
    """A capitalised-only alias written in lower case (``pole``, ``polish``, ``turkey``)."""
    a = alias.strip()
    if not a or has_cjk(a):
        return False
    base = norm_alias(a)
    forms = {
        base,
        base[:-1] if base.endswith("s") else base,
        base[:-2] if base.endswith("es") else base,
    }
    return bool(forms & CAPITALISED_ONLY) and not a[0].isupper()


# ------------------------------------------------------------------ readout samples


def split_samples(raw: Any) -> list[str]:
    """Plan 0005 rule: an O-lens record packs its samples into one ``"- a\\n- b"`` string (each
    ``"\\n- "`` starts a sample); a record not starting with ``"- "`` is one sample."""
    items = raw if isinstance(raw, list) else [raw]
    out: list[str] = []
    for item in items:
        text = str(item)
        if text.startswith("- "):
            out.extend(p.strip() for p in ("\n" + text).split("\n- ") if p.strip())
        elif text.strip():
            out.append(text.strip())
    return out


def bag_text(tokens: Sequence[str]) -> str:
    return " | ".join(t.strip() for t in tokens)


def cell_key(item: str, layer: int, pos: int) -> str:
    return f"{item}__L{layer:03d}__p{pos}"


def uid_of(method: str, item: str, layer: int, pos: int, concept: str) -> str:
    return f"{method}|{item}|{layer}|{pos}|{concept}"


def eff_layer(method: str, layer: int) -> int:
    return 42 if method == NLA42 else layer


# ------------------------------------------------------------------ step 1: mentions


def _resolve(concept: str, gold: str) -> str:
    """The soviet pseudo-concept is Russia unless the gold is former-Soviet."""
    if concept == SOVIET and gold_concept(gold) not in SOVIET_GOLDS:
        return "russia"
    return concept


def item_context(fm: FastMatcher, item: Mapping[str, Any]) -> dict[str, Any]:
    """Per-item matching context: gold equivalents, user-turn concepts, response aliases."""
    user = next(m["content"] for m in item["messages"] if m["role"] == "user")
    resp = item.get("response") or ""
    gold = item["gold"]
    return {
        "gold": gold_concept(gold),
        "gold_raw": gold,
        "equiv": gold_equivalents(gold),
        "user_concepts": {_resolve(c, gold) for c in fm.concepts_in(user)},
        "response_aliases": {norm_alias(h.alias) for h in fm.find(resp) if h.concept},
        "response_concepts": {_resolve(c, gold) for c in fm.concepts_in(resp)},
    }


def classify_hit(concept: str | None, alias: str, ctx: Mapping[str, Any]) -> str | None:
    """``unit`` / a stratum name / ``None`` (not a mention: gold, equivalent, user turn, block)."""
    if concept is None:
        return None
    if concept == SOVIET and ctx["gold"] in SOVIET_GOLDS:
        return "soviet"
    c = _resolve(concept, ctx["gold_raw"])
    if c in ctx["equiv"] or c in ctx["user_concepts"]:
        return None
    if norm_alias(alias) in ctx["response_aliases"]:
        return "response_named"
    return "unit"


def cell_samples(method: str, row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Readout record -> samples ``{si, text}`` (prose) or ``{si, tokens}`` (J-lens, decoded)."""
    if method == "jlens":
        return [{"si": 0, "tokens": [decode_token(t) for t in row.get("tokens") or ()]}]
    raw = row.get("samples") or []
    texts = (
        split_samples(raw) if method == "s3d" else [str(s).strip() for s in raw if str(s).strip()]
    )
    return [{"si": i, "text": t} for i, t in enumerate(texts)]


def jlens_token_hits(fm: FastMatcher, tokens: Sequence[str]) -> list[dict[str, Any]]:
    out = []
    for rank, t in enumerate(tokens):
        c = fm.token_concept(t)
        if c:
            out.append({"concept": c, "alias": t.strip(), "rank": rank, "source": "token"})
    return out


def jlens_needs_summary(fm: FastMatcher, row: Mapping[str, Any], ctx: Mapping[str, Any]) -> bool:
    """A J-lens cell needs a summary iff it has a non-gold decoded token hit that is a unit."""
    toks = [decode_token(t) for t in row.get("tokens") or ()]
    return any(
        classify_hit(h["concept"], h["alias"], ctx) == "unit" for h in jlens_token_hits(fm, toks)
    )


def build_mentions(
    fm: FastMatcher,
    method: str,
    row: Mapping[str, Any],
    ctx: Mapping[str, Any],
    summary: str | None = None,
    summary_status: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One readout cell -> (units, strata records). J-lens units are token hits only; summary
    hits mark the judge's text, and a summary-only concept is a ``summary_only`` stratum."""
    item, layer, pos = row["id"], int(row["layer"]), int(row["pos"])
    base = {"method": method, "item": item, "layer": layer, "pos": pos, "gold": ctx["gold"]}
    per: dict[str, list[dict[str, Any]]] = defaultdict(list)  # concept -> samples with hits
    strata: list[dict[str, Any]] = []
    samples = cell_samples(method, row)
    for smp in samples:
        hits_by: dict[str, list[dict[str, Any]]] = defaultdict(list)
        kinds: dict[str, set[str]] = defaultdict(set)
        if method == "jlens":
            for h in jlens_token_hits(fm, smp["tokens"]):
                k = classify_hit(h["concept"], h["alias"], ctx)
                if k is None:
                    continue
                c = _resolve(h["concept"], ctx["gold_raw"])
                kinds[c].add(k)
                if k == "unit":
                    hits_by[c].append(h)
            sm_hits: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for h in fm.find(summary or ""):
                k = classify_hit(h.concept, h.alias, ctx)
                if k is None or h.concept is None:
                    continue
                c = _resolve(h.concept, ctx["gold_raw"])
                sm_hits[c].append(
                    {
                        "concept": c,
                        "alias": h.alias,
                        "start": h.start,
                        "end": h.end,
                        "source": "summary",
                    }
                )
                if k != "unit":
                    kinds[c].add(k)
            for c, hs in sm_hits.items():
                if c in hits_by:
                    hits_by[c].extend(hs)
                elif "unit" not in kinds[c]:
                    kinds[c].add("summary_only")
            smp = {**smp, "summary": summary, "summary_status": summary_status}
        else:
            for h in fm.find(smp["text"]):
                k = classify_hit(h.concept, h.alias, ctx)
                if k is None or h.concept is None:
                    continue
                c = _resolve(h.concept, ctx["gold_raw"])
                kinds[c].add(k)
                if k == "unit":
                    hits_by[c].append(
                        {
                            "concept": c,
                            "alias": h.alias,
                            "start": h.start,
                            "end": h.end,
                            "source": "text",
                        }
                    )
        for c, hs in hits_by.items():
            per[c].append({**smp, "hits": hs})
        for c, ks in kinds.items():
            for k in ks - {"unit"}:
                if c not in hits_by:
                    strata.append({**base, "concept": c, "stratum": k, "si": smp["si"]})
    units = []
    for c, smps in sorted(per.items()):
        units.append(
            {
                **base,
                "uid": uid_of(method, item, layer, pos, c),
                "concept": c,
                "eff_layer": eff_layer(method, layer),
                "n_samples_cell": len(samples),
                "samples": smps,
            }
        )
    for smp in samples:
        if method != "jlens":
            for m in REGIONAL_WORDS.finditer(smp["text"]):
                strata.append(
                    {
                        **base,
                        "concept": None,
                        "stratum": "regional",
                        "alias": m.group(0),
                        "si": smp["si"],
                    }
                )
    return units, strata


# ------------------------------------------------------------------ step 3: commitment judge

COMMIT_PROMPT_VERSION = "umcommit-v1.1"
MARK_L, MARK_R = "⟦", "⟧"
COMMITTED_FORMS = ("asserted_identity", "reconstructed_prompt")
NOT_COMMITTED_FORMS = (
    "options",
    "comparison",
    "language",
    "other_subject",
    "continuation",
    "negation",
    "question",
    "meta",
)
FORMS = (*COMMITTED_FORMS, *NOT_COMMITTED_FORMS)
SUPPORT_FORMS = ("asserted_identity", "reconstructed_prompt", "options")
ATTRIBUTE_QUESTION = "Which country is the user from (or identifies with)?"

COMMIT_SYSTEM = f"""You audit one sample of an interpretability readout. The readout was \
decoded from a language model's internal state while it read a user's message; it is a noisy \
guess at what the model is thinking about that message. One country (or its adjective / \
demonym) is marked {MARK_L}like this{MARK_R} in the sample.

Decide whether the sample COMMITS to the marked country as the answer to the question: \
"{ATTRIBUTE_QUESTION}" — i.e. the user's nationality, heritage, home country, where they live \
or grew up, or their own culture. You are not told the true answer; do not guess it, and do not \
judge whether the claim is correct. Judge only what the sample says about the marked country.

Pick exactly one form.

COMMITTED (committed = true):
- asserted_identity: the sample asserts or presupposes that the user (or the prompt's author) \
is from / of / lives in the marked country, or that it is the user's own culture. Examples: \
"a user of Chinese heritage", "nationality: Chinese", "| birth_place = Taipei, Taiwan", "as her \
Japanese grandmother taught her", "my Korean upbringing". A single hedged answer still counts \
("probably of Chinese heritage", "likely American"): set hedged = true.
- reconstructed_prompt: the sample reconstructs or describes the user's message and says it is \
about the user's connection to the marked country ("the prompt describes a Persian cultural \
memory", "a message from a Polish user about pierogi", "sharing mate in Uruguay" for the user's \
own ritual, "Domain cues: Australian persona"). The sample claims the input's cultural identity \
is that country.
- The user's message usually asks for a biography or profile OF THE USER ("for me"). If the \
sample presents the requested biography as the user's own, or as an unnamed persona standing in \
for the user, and gives that person the marked nationality or birthplace, it is committed \
(asserted_identity): "an infobox for a person born in Northern Ireland", "a self-deprecating \
entry about a coffee-shop enthusiast in the Philippines", "I have followed the 'Canadian' \
constraint" (for the persona requested). The link to the user must be in the sample: \
"you", "me", "the user", "the speaker", "about me", "self-profile", "for this user", or a \
persona built from the user's own details. A described hypothetical or fictional person with a \
role ("a hypothetical British children's author", "a fictional Indigenous Canadian author", \
"a hypothetical U.S. philanthropist named Jane Doe", "a Chinese business executive", "an \
elderly Chinese veteran", "a Singaporean military officer profile") is other_subject, even \
though the user asked for a biography: the sample does not say that person is the user.

NOT COMMITTED (committed = false):
- options: two or more candidate countries offered as alternatives for the user ("Korean or \
Chinese", "somewhere in East Asia such as China or Japan", "Irish or Scottish?").
- comparison: the country is a reference point, a dish, a product or a travel destination, not \
the user's identity ("similar to Chinese dumplings", "an American-style infobox", "cooking \
Italian pasta with grandma", "a family trip to Tokyo").
- language: the word names a language or a language edition ("written in English", "translate \
to Chinese", "a German-language Wikipedia page").
- other_subject: the country belongs to someone or something other than the user: a specific \
named or described third-party figure or character ("a fictional East German singer named Karl \
Weber", "a Korean Empire general", "a fictional Chinese composer", "the Chinese woman behind the \
bomb"), a company, a band, a news topic or market ("China's food safety", "the Russian market"), \
an example, a historical fact, a placeholder or template option ("e.g. American / British", \
"[Nationality, e.g. Canadian]").
- continuation: text continuing the user's request with no claim about the user's country.
- negation: the sample says the user is NOT from that country.
- question: the sample asks which country.
- meta: talk about the readout, the task or the model itself.

Each sample is judged on its own. For a token-lens sample (a top-10 token list plus a one-line \
summary), a token list alone cannot commit; judge the SUMMARY's claim about the marked country, \
and set quote_source = "summary" if your quote is from the summary, "bag" if from the tokens.

Return: committed, form, hedged (true only for a committed single hedged answer), quote (a \
verbatim span of the sample, 3 to 20 words, that contains the marked word; the whole sample if \
it is shorter), quote_source ("text", "bag" or "summary"), why (one short sentence)."""

COMMIT_SCHEMA = schema_block(
    "umcommit",
    {
        "committed": {"type": "boolean"},
        "form": {"type": "string", "enum": list(FORMS)},
        "hedged": {"type": "boolean"},
        "quote": {"type": "string"},
        "quote_source": {"type": "string", "enum": ["text", "bag", "summary"]},
        "why": {"type": "string"},
    },
    ["committed", "form", "hedged", "quote", "quote_source", "why"],
)


def _mark(text: str, spans: Iterable[tuple[int, int]]) -> str:
    out, last = [], 0
    for s, e in sorted(set(spans)):
        if s < last:
            continue
        out.append(text[last:s] + MARK_L + text[s:e] + MARK_R)
        last = e
    out.append(text[last:])
    return "".join(out)


def user_turn(item: Mapping[str, Any]) -> str:
    return next(m["content"] for m in item["messages"] if m["role"] == "user")


def render_commit(
    unit: Mapping[str, Any], sample: Mapping[str, Any], item: Mapping[str, Any], label: str
) -> tuple[str, str]:
    aliases = sorted({h["alias"] for h in sample["hits"]})
    head = (
        f"USER MESSAGE: {user_turn(item)}\nQUESTION: {ATTRIBUTE_QUESTION}\n"
        f"MARKED COUNTRY: {label} (matched: {', '.join(repr(a) for a in aliases)})\n\n"
    )
    if "tokens" in sample:
        ranks = {h["rank"] for h in sample["hits"] if h["source"] == "token"}
        toks = [
            (MARK_L + t.strip() + MARK_R) if i in ranks else t.strip()
            for i, t in enumerate(sample["tokens"])
        ]
        spans = [(h["start"], h["end"]) for h in sample["hits"] if h["source"] == "summary"]
        body = (
            "READOUT SAMPLE (token lens)\nTOP-10 TOKENS (best first): "
            + " | ".join(toks)
            + "\nSUMMARY: "
            + _mark(sample.get("summary") or "(none)", spans)
        )
    else:
        body = "READOUT SAMPLE:\n" + _mark(
            sample["text"], [(h["start"], h["end"]) for h in sample["hits"]]
        )
    return COMMIT_SYSTEM, head + body


_LATIN_WORD = re.compile(r"[A-Za-z0-9À-ɏ']+")
_CJK_CHAR = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")


def n_words(s: str) -> int:
    """Word count for the quote-length rule: Latin words, plus CJK characters at two per word
    (a 20-word limit would otherwise reject any 21-character Chinese clause)."""
    return len(_LATIN_WORD.findall(s)) + (len(_CJK_CHAR.findall(s)) + 1) // 2


def sample_haystacks(sample: Mapping[str, Any]) -> dict[str, str]:
    if "tokens" in sample:
        return {"bag": bag_text(sample["tokens"]), "summary": sample.get("summary") or ""}
    return {"text": sample["text"]}


def quote_ok(quote: str, source: str, sample: Mapping[str, Any]) -> bool:
    q = normalize_ws(quote.replace(MARK_L, "").replace(MARK_R, ""))
    hay = sample_haystacks(sample)
    if not q or source not in hay:
        return False
    h = normalize_ws(hay[source])
    if q not in h:
        return False
    nw, total = n_words(q), n_words(h)
    if nw > 20 or (nw < 3 and total >= 3):
        return False
    if source == "bag":
        return any(
            norm_alias(t) and norm_alias(t) in q.casefold()
            for t in (x["alias"] for x in sample["hits"] if x["source"] == "token")
        )
    aliases = {x["alias"].strip().casefold() for x in sample["hits"]}
    return any(a and a in q.casefold() for a in aliases)


def valid_commit(r: Any, sample: Mapping[str, Any]) -> bool:
    if not isinstance(r, Mapping):
        return False
    if r.get("form") not in FORMS or not isinstance(r.get("committed"), bool):
        return False
    if r["committed"] != (r["form"] in COMMITTED_FORMS):
        return False
    if not isinstance(r.get("hedged"), bool):
        return False
    return quote_ok(str(r.get("quote", "")), str(r.get("quote_source", "")), sample)


def parse_commit(r: Any, sample: Mapping[str, Any]) -> dict[str, Any]:
    """Judge answer -> sample verdict. Invalid / missing -> fail closed to committed."""
    if r is None or not valid_commit(r, sample):
        return {
            "committed": True,
            "form": "fail_closed",
            "hedged": False,
            "quote": None,
            "quote_source": None,
            "why": "invalid or missing judge answer (fail closed)",
            "fail_closed": True,
        }
    return {
        "committed": bool(r["committed"]),
        "form": r["form"],
        "hedged": bool(r["hedged"]) and bool(r["committed"]),
        "quote": normalize_ws(str(r["quote"]).replace(MARK_L, "").replace(MARK_R, "")),
        "quote_source": r["quote_source"],
        "why": r.get("why", ""),
        "fail_closed": False,
    }


def rollup(verdicts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Unit verdict: committed if any sample is; counts, forms, the first committed quote, and
    the hedged-reading variant (committed only via non-hedged samples)."""
    com = [v for v in verdicts if v["committed"]]
    firm = [v for v in com if not v.get("hedged")]
    pick = com[0] if com else (verdicts[0] if verdicts else None)
    return {
        "committed": bool(com),
        "committed_unhedged": bool(firm),
        "n_committed": len(com),
        "n_samples": len(verdicts),
        "n_fail_closed": sum(bool(v.get("fail_closed")) for v in verdicts),
        "fail_closed_only": bool(com) and all(v.get("fail_closed") for v in com),
        "forms": sorted({v["form"] for v in verdicts}),
        "quote": pick.get("quote") if pick else None,
        "quote_source": pick.get("quote_source") if pick else None,
        "via_summary": bool(com) and all(v.get("quote_source") == "summary" for v in com),
        "supports": any(v["form"] in SUPPORT_FORMS for v in verdicts),
    }


# ------------------------------------------------------------------ step 4c / 4d: probes

N_DISTRACTORS = 8
N_NEAREST = 4
MIN_DISTRACTORS = 6
READABLE_SHARE = 0.5
VALIDITY_MIN = 0.90
TYPES = ("a", "b")
DISTRACTOR_LAYER = 44


def neighbours(layer: int) -> tuple[int, ...]:
    """Probe layers in [L - 4, L + 4] other than L (L42 is a neighbour of 40 and 44)."""
    return tuple(x for x in PROBE_LAYERS if x != layer and abs(x - layer) <= 4)


def select_distractors(
    item: str,
    gold: str,
    concepts: Sequence[str],
    centroids: Mapping[str, Sequence[float]],
    valid44: Mapping[str, bool],
    committed_for_item: Iterable[str],
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """D_i: 4 eligible concepts nearest the gold (cosine of grand-mean-centred implied
    centroids at L44) + 4 seeded random eligible ones. Eligible = in C, valid at L44 in both
    types, not the gold / an equivalent, not committed for the item by any method."""
    import random

    import numpy as np

    excl = set(gold_equivalents(gold)) | set(committed_for_item)
    elig = sorted(c for c in concepts if c not in excl and valid44.get(c, False))
    g = np.asarray(centroids[gold_concept(gold)], dtype=np.float64)

    def cos(c: str) -> float:
        v = np.asarray(centroids[c], dtype=np.float64)
        return float(v @ g / (np.linalg.norm(v) * np.linalg.norm(g) + 1e-12))

    ranked = sorted(elig, key=lambda c: (-cos(c), c))
    nearest = ranked[:N_NEAREST]
    rest = [c for c in elig if c not in nearest]
    rng = random.Random(f"{seed}:{item}")
    rnd = sorted(rng.sample(rest, min(N_DISTRACTORS - len(nearest), len(rest))))
    return {
        "item": item,
        "gold": gold_concept(gold),
        "nearest": nearest,
        "random": rnd,
        "distractors": [*nearest, *rnd],
        "n_eligible": len(elig),
        "cos_to_gold": {c: round(cos(c), 4) for c in [*nearest, *rnd]},
    }


class ZTable:
    """Fold-averaged null z of every concept of C at every (item, layer, pos), both types."""

    def __init__(self, concepts: Sequence[str]) -> None:
        self.concepts = list(concepts)
        self.ci = {c: i for i, c in enumerate(self.concepts)}
        self._z: dict[tuple[str, int, int], dict[str, Any]] = {}

    def add_layer(
        self, layer: int, item: Sequence[str], pos: Sequence[int], za: Any, zb: Any
    ) -> None:
        for r, (i, p) in enumerate(zip(item, pos, strict=True)):
            self._z[(str(i), int(layer), int(p))] = {"a": za[r], "b": zb[r]}

    def has(self, item: str, layer: int, pos: int) -> bool:
        return (item, layer, pos) in self._z

    def get(self, item: str, layer: int, pos: int, t: str, c: str) -> float:
        return float(self._z[(item, layer, pos)][t][self.ci[c]])

    def positions(self, item: str, layer: int) -> list[int]:
        return sorted(p for (i, L, p) in self._z if i == item and L == layer)


def beats_all(zt: ZTable, item: str, layer: int, pos: int, c: str, others: Sequence[str]) -> bool:
    """c's z above every concept of ``others`` in BOTH types (NaN never beats)."""
    import math

    if not others:
        return False
    for t in TYPES:
        zc = zt.get(item, layer, pos, t, c)
        if math.isnan(zc):
            return False
        for d in others:
            zd = zt.get(item, layer, pos, t, d)
            if math.isnan(zd) or zc <= zd:
                return False
    return True


class ProbeContext:
    """Everything the fire rule needs: z table, validity per (concept, layer, type), D_i,
    readability per (item, layer)."""

    def __init__(
        self,
        zt: ZTable,
        validity: Mapping[tuple[str, int, str], float],
        distractors: Mapping[str, Sequence[str]],
        gold_of: Mapping[str, str],
    ) -> None:
        self.zt, self.validity, self.dist, self.gold_of = zt, validity, distractors, gold_of
        self._read: dict[tuple[str, int], float] = {}

    def valid(self, c: str, layer: int) -> bool:
        return all(self.validity.get((c, layer, t), 0.0) >= VALIDITY_MIN for t in TYPES)

    def d_at(self, item: str, layer: int) -> list[str]:
        return [d for d in self.dist.get(item, ()) if self.valid(d, layer)]

    def readable_share(self, item: str, layer: int) -> float:
        k = (item, layer)
        if k not in self._read:
            g, D = self.gold_of[item], self.d_at(item, layer)
            ps = self.zt.positions(item, layer)
            hits = [beats_all(self.zt, item, layer, p, g, D) for p in ps] if D else []
            self._read[k] = sum(hits) / len(hits) if hits else 0.0
        return self._read[k]

    def na_reason(self, item: str, layer: int, c: str) -> str | None:
        if c not in self.zt.ci:
            return "no probe"
        if not self.zt.positions(item, layer):
            return "no scores"
        if not self.valid(c, layer):
            return "invalid probe at L"
        if len(self.d_at(item, layer)) < MIN_DISTRACTORS:
            return "few distractors at L"
        if self.readable_share(item, layer) < READABLE_SHARE:
            return "item unreadable at L"
        return None

    def fires(self, item: str, layer: int, pos: int, c: str) -> bool:
        return self.na_reason(item, layer, c) is None and beats_all(
            self.zt, item, layer, pos, c, [d for d in self.d_at(item, layer) if d != c]
        )

    def probe_result(self, item: str, layer: int, pos: int, c: str) -> dict[str, Any]:
        """At the unit's (effective) layer and its neighbours."""
        reason = self.na_reason(item, layer, c)
        fire_l = reason is None and self.fires(item, layer, pos, c)
        nb = [L for L in neighbours(layer) if self.na_reason(item, L, c) is None]
        fire_nb = [L for L in nb if self.fires(item, L, pos, c)]
        return {
            "na_reason": reason,
            "adjudicable": reason is None or bool(nb),
            "fire_L": fire_l,
            "fire_nearby_layers": fire_nb,
            "fire_nearby": (not fire_l) and bool(fire_nb),
            "readable_share": round(self.readable_share(item, layer), 3)
            if self.zt.positions(item, layer)
            else None,
            "n_distractors": len(self.d_at(item, layer)),
        }


# ------------------------------------------------------------------ step 5: label

L_NOT = "not committed"
L_SAME = "probe-faithful (same layer)"
L_NEAR = "probe-faithful (nearby layer)"
L_CROSS = "cross-method allowed (adjudicable)"
L_CROSS_NA = "cross-method allowed (probe n/a)"
L_HALL = "hallucination"
L_HALL_NA = "hallucination (probe n/a)"
LABELS = (L_NOT, L_SAME, L_NEAR, L_CROSS, L_CROSS_NA, L_HALL, L_HALL_NA)
ADJUDICATED = (L_SAME, L_NEAR, L_CROSS, L_HALL)


def ladder_label(committed: bool, probe: Mapping[str, Any], cross: bool) -> str:
    if not committed:
        return L_NOT
    if probe["fire_L"]:
        return L_SAME
    if probe["fire_nearby"]:
        return L_NEAR
    if cross:
        return L_CROSS if probe["adjudicable"] else L_CROSS_NA
    return L_HALL if probe["adjudicable"] else L_HALL_NA


def support_index(
    commit_rows: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str], list[tuple[str, int]]]:
    """(item, concept) -> [(method, effective layer)] of units whose judged samples include a
    supporting form (asserted_identity / reconstructed_prompt / options; never fail-closed)."""
    idx: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    for r in commit_rows:
        if any(s.get("form") in SUPPORT_FORMS and not s.get("fail_closed") for s in r["samples"]):
            idx[(r["item"], r["concept"])].append((r["method"], eff_layer(r["method"], r["layer"])))
    return idx


def cross_support(
    unit: Mapping[str, Any],
    idx: Mapping[tuple[str, str], list[tuple[str, int]]],
    *,
    any_layer: bool = False,
) -> list[str]:
    L = eff_layer(unit["method"], unit["layer"])
    return sorted(
        {
            m
            for m, l2 in idx.get((unit["item"], unit["concept"]), ())
            if m != unit["method"] and (any_layer or abs(l2 - L) <= 4)
        }
    )


def lens_prior(units: Iterable[Mapping[str, Any]], n_items: int) -> dict[tuple[str, str], set[str]]:
    """(method, concept) -> items for which the method mentions the concept (any unit)."""
    out: dict[tuple[str, str], set[str]] = defaultdict(set)
    for u in units:
        out[(u["method"], u["concept"])].add(u["item"])
    return out


def bootstrap_rate(
    rows: Sequence[Mapping[str, Any]],
    num: Any,
    den: Any,
    cluster: str,
    n: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Ratio sum(num) / sum(den) with a cluster bootstrap CI (clusters resampled)."""
    import random

    by: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        by[str(r[cluster])].append(r)
    keys = sorted(by)
    tot_n = sum(num(r) for r in rows)
    tot_d = sum(den(r) for r in rows)
    est = tot_n / tot_d if tot_d else float("nan")
    rng = random.Random(seed)
    sims = []
    for _ in range(n if keys else 0):
        sample = [k for k in (rng.choice(keys) for _ in keys)]
        nn = sum(num(r) for k in sample for r in by[k])
        dd = sum(den(r) for k in sample for r in by[k])
        if dd:
            sims.append(nn / dd)
    sims.sort()
    lo = sims[int(0.025 * len(sims))] if sims else float("nan")
    hi = sims[min(len(sims) - 1, int(0.975 * len(sims)))] if sims else float("nan")
    return {"rate": est, "lo": lo, "hi": hi, "num": tot_n, "den": tot_d, "n_clusters": len(keys)}
