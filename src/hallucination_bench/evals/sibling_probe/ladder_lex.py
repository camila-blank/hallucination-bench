"""Plan 0005 step 1: category lexicons and the mention matcher (basic_readout entity items).

``data/sibling_probe/br_lexicons.json``::

    {"version", "provenance", "categories": {cat: {"concepts": {id: {"label", "en": [...],
      "cjk": [...], "tokens": [...]}}, "block": [...]}}, "ambiguous": {cat: {alias: [ids]}},
      "checks": {...}}

Matching rules (plan rev 2 item 6):

* every alias of every concept of the item's category is located in the text; Latin aliases
  need word boundaries (an optional plural ``s`` / ``es``), CJK aliases (>= 2 characters) are
  plain substrings; ``block`` phrases (capital cities, "Mexico City", the item's own entity
  name, ...) take part in the matching but name no concept;
* **longest match wins**: overlapping matches are resolved longest first (field hockey is not
  hockey, table tennis is not tennis, Papua New Guinea is not Guinea);
* an ``ambiguous`` alias ("football") resolves to the item's gold when the gold is one of its
  options, else to the item's default (``FOOTBALL_DEFAULT``) and the hit is flagged;
* concepts named in the prompt and the gold (with its equivalents, which are aliases of the
  gold concept) are not mentions;
* J-lens token bags: a decoded token (stripped, casefolded) equal to an alias / token alias
  (or its plural), or a CJK token containing a CJK alias.
"""

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from hallucination_bench import DATA

LEXICON_PATH = DATA / "sibling_probe" / "br_lexicons.json"
LEXICON_VERSION = "brlex-v1"
CATEGORIES = ("sport", "country", "musical_instrument")

# country aliases that are also common words / names: matched case-sensitively
CASE_SENSITIVE = frozenset(
    {
        "chad",
        "jordan",
        "georgia",
        "turkey",
        "china",
        "guinea",
        "togo",
        "mali",
        "niger",
        "cuba",
        "japan",
        "france",
        "panama",
        "malta",
        "monaco",
        "oman",
        "peru",
        "fiji",
        "nauru",
    }
)
# "football" for an item whose gold is neither code: the athlete's own country's usage
FOOTBALL_DEFAULT = {
    "br-ent-ath-ali": "american_football",
    "br-ent-ath-james": "american_football",
    "br-ent-ath-jordan": "american_football",
    "br-ent-ath-ruth": "american_football",
    "br-ent-ath-williams": "american_football",
    "br-ent-ath-woods": "american_football",
    "br-ent-ath-gretzky": "american_football",
    "br-ent-ath-federer": "soccer",
    "br-ent-ath-nadal": "soccer",
}
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")


def has_cjk(s: str) -> bool:
    return bool(_CJK.search(s))


def norm_alias(s: str) -> str:
    return unicodedata.normalize("NFKC", s).strip().casefold()


@dataclass(frozen=True)
class Hit:
    concept: str | None  # None = a block phrase
    alias: str  # the matched text, verbatim
    start: int
    end: int
    ambiguous: bool = False


@dataclass
class _Pat:
    concept: str | None
    alias: str
    rx: re.Pattern[str] | None  # None = CJK substring
    ambiguous: tuple[str, ...] = ()


def _latin_rx(alias: str, case_sensitive: bool) -> re.Pattern[str]:
    parts = [re.escape(p) for p in re.split(r"[\s\-]+", alias.strip()) if p]
    body = r"[\s\-]+".join(parts)
    return re.compile(
        r"(?<![A-Za-z0-9])" + body + r"(?:s|es)?(?![A-Za-z0-9])",
        0 if case_sensitive else re.IGNORECASE,
    )


class Matcher:
    """Mentions of one category's concepts in text and token bags."""

    def __init__(self, lexicon: Mapping[str, Any], category: str) -> None:
        cat = lexicon["categories"][category]
        self.category = category
        self.concepts: dict[str, Mapping[str, Any]] = dict(cat["concepts"])
        self.ambiguous: dict[str, list[str]] = {
            norm_alias(k): list(v)
            for k, v in lexicon.get("ambiguous", {}).get(category, {}).items()
        }
        self._pats: list[_Pat] = []
        self._tok: dict[str, str] = {}  # normalised token form -> concept
        self._tok_cjk: list[tuple[str, str]] = []
        for cid, c in self.concepts.items():
            for a in c.get("en", []):
                self._add(cid, a)
                self._tok.setdefault(norm_alias(a), cid)
            for a in c.get("cjk", []):
                if len(a) >= 2:
                    self._pats.append(_Pat(cid, a, None))
                    self._tok_cjk.append((a, cid))
            for a in c.get("tokens", []):
                self._tok.setdefault(norm_alias(a), cid)
        for a in cat.get("block", []):
            self._add(None, a)
        for a, opts in lexicon.get("ambiguous", {}).get(category, {}).items():
            if has_cjk(a):
                self._pats.append(_Pat(None, a, None, tuple(opts)))
            else:
                self._pats.append(_Pat(None, a, _latin_rx(a, False), tuple(opts)))
            self._tok.setdefault(norm_alias(a), "__amb__:" + a)

    def _add(self, cid: str | None, alias: str) -> None:
        if not alias.strip():
            return
        if has_cjk(alias):
            if len(alias) >= 2:
                self._pats.append(_Pat(cid, alias, None))
            return
        cs = (alias.isupper() and len(alias) <= 5) or (
            self.category == "country" and norm_alias(alias) in CASE_SENSITIVE
        )
        self._pats.append(_Pat(cid, alias, _latin_rx(alias, cs)))

    def _resolve_amb(self, opts: Sequence[str], gold: str, item: str) -> str | None:
        if gold in opts:
            return gold
        d = FOOTBALL_DEFAULT.get(item)
        if d in opts:
            return d
        return opts[0] if opts else None

    def find(
        self, text: str, *, gold: str = "", item: str = "", block: Iterable[str] = ()
    ) -> list[Hit]:
        """Longest-first, non-overlapping hits in ``text`` (block phrases included, concept
        None). ``block`` adds per-call phrases (the item's entity name)."""
        cands: list[tuple[int, int, _Pat, str]] = []
        extra = [_Pat(None, b, _latin_rx(b, False) if not has_cjk(b) else None) for b in block if b]
        for p in [*self._pats, *extra]:
            if p.rx is None:
                i = text.find(p.alias)
                while i >= 0:
                    cands.append((i, i + len(p.alias), p, p.alias))
                    i = text.find(p.alias, i + 1)
            else:
                for m in p.rx.finditer(text):
                    cands.append((m.start(), m.end(), p, m.group(0)))
        cands.sort(key=lambda c: (-(c[1] - c[0]), c[0]))
        taken: list[tuple[int, int]] = []
        hits: list[Hit] = []
        for s, e, p, txt in cands:
            if any(s < te and ts < e for ts, te in taken):
                continue
            taken.append((s, e))
            if p.ambiguous:
                hits.append(Hit(self._resolve_amb(p.ambiguous, gold, item), txt, s, e, True))
            else:
                hits.append(Hit(p.concept, txt, s, e))
        hits.sort(key=lambda h: h.start)
        return hits

    def token_concept(
        self, token: str, *, gold: str = "", item: str = ""
    ) -> tuple[str, bool] | None:
        """Concept named by one decoded J-lens token: (concept, ambiguous) or None."""
        t = norm_alias(token)
        if not t:
            return None
        forms = [t]
        if t.endswith("es"):
            forms.append(t[:-2])
        if t.endswith("s"):
            forms.append(t[:-1])
        for f in forms:
            c = self._tok.get(f)
            if c is not None:
                if c.startswith("__amb__:"):
                    opts = self.ambiguous.get(norm_alias(c.split(":", 1)[1]), [])
                    r = self._resolve_amb(opts, gold, item)
                    return (r, True) if r else None
                return c, False
        if has_cjk(token):
            best = None
            for a, cid in self._tok_cjk:
                if a in token and (best is None or len(a) > len(best[0])):
                    best = (a, cid)
            if best:
                return best[1], False
        return None

    def prompt_concepts(self, prompt: str, item: str = "") -> set[str]:
        """Concepts named in the prompt (never mentions for that item)."""
        return {h.concept for h in self.find(prompt, item=item) if h.concept}


def overlap_report(lexicon: Mapping[str, Any]) -> dict[str, Any]:
    """Code checks on a lexicon: duplicate aliases across concepts (conflicts), CJK aliases
    shorter than 2 characters, aliases that are substrings of another concept's alias (resolved
    by longest match; listed for the record)."""
    out: dict[str, Any] = {}
    for cat, body in lexicon["categories"].items():
        owner: dict[str, set[str]] = {}
        short_cjk: list[str] = []
        for cid, c in body["concepts"].items():
            for a in [*c.get("en", []), *c.get("cjk", [])]:
                owner.setdefault(norm_alias(a), set()).add(cid)
                if has_cjk(a) and len(a) < 2:
                    short_cjk.append(f"{cid}:{a}")
        dup = {a: sorted(v) for a, v in owner.items() if len(v) > 1}
        subs = []
        keys = sorted(owner)
        for a in keys:
            for b in keys:
                if a != b and len(a) >= 3 and owner[a] != owner[b]:
                    if has_cjk(a):
                        if a in b:
                            subs.append(f"{a}<{b}")
                    elif re.search(r"(?<![a-z0-9])" + re.escape(a) + r"(?![a-z0-9])", b):
                        subs.append(f"{a}<{b}")
        out[cat] = {
            "n_concepts": len(body["concepts"]),
            "n_aliases": sum(
                len(c.get("en", [])) + len(c.get("cjk", [])) for c in body["concepts"].values()
            ),
            "duplicates": dup,
            "short_cjk": short_cjk,
            "substring_pairs": subs[:400],
            "n_substring_pairs": len(subs),
        }
    return out
