"""Concept inventory (``data/sibling_probe/concepts.json``) and the alias matcher.

File schema::

    {"version": str, "provenance": str,
     "categories": {<category>: {
         "question": str,              # the pre-registered implication question
         "requests": [str, ...],       # the category's eval request sentences (1 or 2)
         "merges": [{"concept": id, "members": [..], "note": str}],
         "concepts": [{"id", "label", "gold": bool, "aliases": [..en.., ..zh..],
                       "exclude": {alias: [phrase, ..]},   # every CJK alias has a key
                       "token_aliases": [..], "token_eligible": bool}]}}}

Matching rules (plan 0003 §Concept inventory):

* English aliases: lower-cased, curly apostrophes folded to ``'``, on ASCII-alphanumeric
  boundaries (so ``Korea的`` still matches ``Korea``, but ``holi`` never matches inside
  ``holiday``). Plurals and demonyms are listed explicitly; nothing is stemmed.
* CJK aliases (at least 2 characters): plain substring.
* An alias hit is discarded if it lies inside an occurrence of one of that alias's exclusion
  phrases (``happy`` inside ``happy to help``), or strictly inside a longer hit of another
  concept (``new year`` inside ``lunar new year``).
* Token aliases (J-lens top-k bags): a decoded vocabulary token matches a concept iff the token,
  whitespace-stripped and lower-cased, equals one of the concept's token aliases normalised the
  same way. Every listed token alias is a single Qwen3.6-27B token (checked by a test); there is
  no first-subtoken matching.
"""

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from hallucination_bench import DATA
from hallucination_bench.readouts import decode_token

CONCEPTS_PATH = DATA / "sibling_probe" / "concepts.json"
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿가-힯]")
_EDGE = "a-z0-9À-ɏ"  # ASCII + Latin-1/Extended letters count as word characters


def is_cjk(s: str) -> bool:
    return bool(_CJK.search(s))


def normalize_text(text: str) -> str:
    """Length-preserving normalisation the English matcher runs on."""
    return text.lower().replace("’", "'").replace("‘", "'")


def normalize_token(tok: str) -> str:
    return tok.strip().lower().replace("’", "'")


@dataclass(frozen=True)
class Concept:
    id: str
    label: str
    category: str
    gold: bool
    aliases: tuple[str, ...]
    exclude: dict[str, tuple[str, ...]] = field(default_factory=dict)
    token_aliases: tuple[str, ...] = ()
    token_eligible: bool = False


@dataclass(frozen=True)
class Hit:
    concept: str
    alias: str
    start: int
    end: int


@dataclass(frozen=True)
class Category:
    name: str
    question: str
    requests: tuple[str, ...]
    concepts: tuple[Concept, ...]
    merges: tuple[dict[str, Any], ...] = ()

    @property
    def ids(self) -> list[str]:
        return [c.id for c in self.concepts]

    def concept(self, cid: str) -> Concept:
        for c in self.concepts:
            if c.id == cid:
                return c
        raise KeyError(f"{self.name}: no concept {cid!r}")


def _parse_category(name: str, d: dict[str, Any]) -> Category:
    concepts = []
    for c in d["concepts"]:
        exclude = {a: tuple(v) for a, v in (c.get("exclude") or {}).items()}
        concepts.append(
            Concept(
                id=c["id"],
                label=c["label"],
                category=name,
                gold=bool(c.get("gold", False)),
                aliases=tuple(c["aliases"]),
                exclude=exclude,
                token_aliases=tuple(c.get("token_aliases") or ()),
                token_eligible=bool(c.get("token_eligible", False)),
            )
        )
    return Category(
        name=name,
        question=d["question"],
        requests=tuple(d["requests"]),
        concepts=tuple(concepts),
        merges=tuple(d.get("merges") or ()),
    )


def parse_inventory(doc: dict[str, Any]) -> dict[str, Category]:
    return {name: _parse_category(name, d) for name, d in doc["categories"].items()}


@cache
def load_inventory(path: Path = CONCEPTS_PATH) -> dict[str, Category]:
    return parse_inventory(json.loads(Path(path).read_text(encoding="utf-8")))


def validate_inventory(cats: dict[str, Category]) -> list[str]:
    """Structural problems (empty list = valid). Overlap is checked by ``overlap_problems``."""
    errs: list[str] = []
    for cat in cats.values():
        ids = cat.ids
        if len(set(ids)) != len(ids):
            errs.append(f"{cat.name}: duplicate concept ids")
        if not 1 <= len(cat.requests) <= 2:
            errs.append(f"{cat.name}: {len(cat.requests)} request sentences")
        for c in cat.concepts:
            if not c.aliases:
                errs.append(f"{cat.name}/{c.id}: no aliases")
            for a in c.aliases:
                if a != a.strip() or not a:
                    errs.append(f"{cat.name}/{c.id}: alias {a!r} has edge whitespace")
                if is_cjk(a):
                    if len(a) < 2:
                        errs.append(f"{cat.name}/{c.id}: CJK alias {a!r} shorter than 2 chars")
                    if a not in c.exclude:
                        errs.append(f"{cat.name}/{c.id}: CJK alias {a!r} has no exclusion list")
            for a in c.exclude:
                if a not in c.aliases:
                    errs.append(f"{cat.name}/{c.id}: exclusion key {a!r} is not an alias")
            if c.token_eligible != bool(c.token_aliases):
                errs.append(f"{cat.name}/{c.id}: token_eligible disagrees with token_aliases")
    return errs


class Matcher:
    """Alias matcher for one category."""

    def __init__(self, category: Category):
        self.category = category
        self._en: list[tuple[str, str, re.Pattern[str], tuple[str, ...]]] = []
        self._cjk: list[tuple[str, str, tuple[str, ...]]] = []
        self._tok: dict[str, str] = {}
        for c in category.concepts:
            for a in c.aliases:
                excl = tuple(c.exclude.get(a, ()))
                if is_cjk(a):
                    self._cjk.append((c.id, a, excl))
                else:
                    pat = re.compile(rf"(?<![{_EDGE}]){re.escape(normalize_text(a))}(?![{_EDGE}])")
                    self._en.append((c.id, a, pat, tuple(normalize_text(x) for x in excl)))
            for t in c.token_aliases:
                self._tok[normalize_token(t)] = c.id

    @staticmethod
    def _spans(hay: str, needle: str) -> list[tuple[int, int]]:
        out, i = [], hay.find(needle)
        while i >= 0:
            out.append((i, i + len(needle)))
            i = hay.find(needle, i + 1)
        return out

    def hits(self, text: str) -> list[Hit]:
        """Every alias hit in ``text`` (character offsets into ``text``), after exclusions and
        the cross-concept containment rule, sorted by position."""
        norm = normalize_text(text)
        raw: list[Hit] = []
        for cid, alias, pat, excl in self._en:
            ex = [s for x in excl for s in self._spans(norm, x)]
            for m in pat.finditer(norm):
                if not any(a <= m.start() and m.end() <= b for a, b in ex):
                    raw.append(Hit(cid, alias, m.start(), m.end()))
        for cid, alias, excl in self._cjk:
            ex = [s for x in excl for s in self._spans(text, x)]
            for s, e in self._spans(text, alias):
                if not any(a <= s and e <= b for a, b in ex):
                    raw.append(Hit(cid, alias, s, e))
        kept = [
            h
            for h in raw
            if not any(
                o.concept != h.concept
                and o.start <= h.start
                and h.end <= o.end
                and (o.end - o.start) > (h.end - h.start)
                for o in raw
            )
        ]
        return sorted(kept, key=lambda h: (h.start, h.end, h.concept))

    def mentions(self, text: str) -> set[str]:
        return {h.concept for h in self.hits(text)}

    def token_mentions(self, tokens: Iterable[str], *, raw: bool = True) -> set[str]:
        """Concepts hit by a token bag. ``raw``: tokens are byte-level BPE vocabulary strings
        (the J-lens readouts' ``tokens``), decoded first; else already-decoded text."""
        out = set()
        for t in tokens:
            cid = self._tok.get(normalize_token(decode_token(t) if raw else t))
            if cid is not None:
                out.add(cid)
        return out

    def token_concept(self, tok: str, *, raw: bool = True) -> str | None:
        return self._tok.get(normalize_token(decode_token(tok) if raw else tok))


@cache
def matcher(category: str, path: Path = CONCEPTS_PATH) -> Matcher:
    return Matcher(load_inventory(path)[category])


def overlap_problems(cat: Category) -> list[str]:
    """Each alias (and the label) must match its own concept and no other; token aliases must
    not be shared between concepts."""
    m = Matcher(cat)
    errs: list[str] = []
    for c in cat.concepts:
        for a in (*c.aliases, c.label):
            got = m.mentions(a)
            if got != {c.id}:
                errs.append(f"{cat.name}/{c.id}: {a!r} matches {sorted(got)}")
    seen: dict[str, str] = {}
    for c in cat.concepts:
        for t in c.token_aliases:
            k = normalize_token(t)
            if k in seen and seen[k] != c.id:
                errs.append(f"{cat.name}: token alias {t!r} in {seen[k]} and {c.id}")
            seen[k] = c.id
    return errs
