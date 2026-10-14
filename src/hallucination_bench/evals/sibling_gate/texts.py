"""Plan 0009 §3b-c: probe texts (``sibgen-chat-v1``), their code checks, the read positions,
and the punctuation token-class key.

Texts are short wsbench_chat-style exchanges (system, user turn, assistant reply). Per concept of
C: 24 ``explicit`` (the reply names the concept and no other concept of C) and 24 ``implied``
(the reply makes the concept unambiguous without naming it); per class 24 ``neutral`` (the reply
names nothing in C; not a probe class). Code checks (``um_ladder.text_check`` adapted to C):
naming as specified, no word 8-gram shared with any wsbench_chat item text, no duplicates.

Read positions: every punctuation token of the reply after the concept's first explicit or
implied mention, up to 24 per text. **New recipe** (stated as such): centring and the null are
keyed by the read position's punctuation token class, through ``um_probe._key`` /
``eval_key`` with ``ending = 0`` and ``offset = class index``.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from hallucination_bench.cache import Cache
from hallucination_bench.llm import LLM, Usage, schema_block

from . import units as U
from .fill_judge import EFFORT, MODEL, cached_json_call, gather_limited

GEN_VERSION = "sibgen-chat-v1"
PER_KIND = 24
NEUTRAL_N = 24
MAX_READ = 24
NGRAM = 8
CACHE_PATH = U.OUT / "texts" / "cache.jsonl"

# ---------------------------------------------------------------- punctuation token classes

PUNCT_CLASSES: tuple[str, ...] = (
    ".",
    ",",
    "!",
    "?",
    ":",
    ";",
    "\n",
    ")",
    "。",
    "，",
    "！",
    "？",
    "：",
    "other",
)
OTHER = PUNCT_CLASSES.index("other")
MIN_CLASS_POSITIONS = 50  # a class with fewer training positions merges into `other`


def is_punct_token(tok: str) -> bool:
    """A token made only of punctuation and whitespace, with at least one punctuation mark or a
    newline (a pure-space token is not a read position)."""
    if not tok or not tok.strip(" \t"):
        return False
    return all(ch.isspace() or unicodedata.category(ch).startswith("P") for ch in tok)


def punct_class(tok: str) -> int:
    """Class index of a punctuation token: spaces and tabs are ignored; a mark followed by
    newlines takes the mark's class; a newline-only token is ``\\n``; anything else ``other``."""
    s = tok.strip(" \t")
    core = s.rstrip("\n")
    if core in PUNCT_CLASSES[:-1]:
        return PUNCT_CLASSES.index(core)
    if s and not core:
        return PUNCT_CLASSES.index("\n")
    return OTHER


def merge_small(counts: Mapping[int, int], min_n: int = MIN_CLASS_POSITIONS) -> dict[int, int]:
    """class index -> effective index: classes with fewer than ``min_n`` training positions
    merge into ``other``."""
    return {
        c: (c if counts.get(c, 0) >= min_n or c == OTHER else OTHER)
        for c in range(len(PUNCT_CLASSES))
    }


def class_key(
    tokens: Sequence[str], merge: Mapping[int, int] | None = None
) -> tuple[list[int], list[int]]:
    """(ending, offset) arrays for ``um_probe._key`` / ``eval_key``: ending 0, offset = class."""
    off = [punct_class(t) for t in tokens]
    if merge is not None:
        off = [merge.get(o, OTHER) for o in off]
    return [0] * len(off), off


def read_positions(
    tokens: Sequence[str],
    starts: Sequence[int],
    reply_start: int,
    mention_char: int,
    max_n: int = MAX_READ,
) -> list[int]:
    """Token indices of punctuation tokens of the reply starting at/after ``mention_char`` (a
    char offset into the rendered text; ``starts`` = each token's char start)."""
    lo = max(reply_start, mention_char)
    out = [
        i
        for i, (t, s) in enumerate(zip(tokens, starts, strict=True))
        if s >= lo and is_punct_token(t)
    ]
    return out[:max_n]


# ---------------------------------------------------------------- naming checks


def _pattern(name: str) -> re.Pattern[str]:
    n = re.escape(name.strip())
    if re.search(r"[A-Za-z0-9]", name):
        return re.compile(rf"(?<![A-Za-z0-9]){n}(?![A-Za-z0-9])", re.IGNORECASE)
    return re.compile(n)


class Namer:
    """Which concepts of C a text names (canonical names + aliases, word-bounded for Latin)."""

    def __init__(self, surface: Mapping[str, Sequence[str]]) -> None:
        self.pats = {
            c: [_pattern(a) for a in {c, *forms} if a.strip()] for c, forms in surface.items()
        }

    def found(self, text: str) -> set[str]:
        return {c for c, ps in self.pats.items() if any(p.search(text) for p in ps)}

    def first(self, text: str, c: str) -> int | None:
        hits = [m.start() for p in self.pats.get(c, []) if (m := p.search(text))]
        return min(hits) if hits else None


_WORD = re.compile(r"\w+", re.UNICODE)


def ngrams(text: str, n: int = NGRAM) -> set[tuple[str, ...]]:
    w = [x.casefold() for x in _WORD.findall(text)]
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}


def item_ngram_index(items: Iterable[Mapping[str, Any]], n: int = NGRAM) -> set[tuple[str, ...]]:
    idx: set[tuple[str, ...]] = set()
    for it in items:
        idx |= ngrams(it.get("prompt") or "", n) | ngrams(it.get("response") or "", n)
    return idx


def normalize_text(t: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", t).casefold().split())


def text_check(
    namer: Namer,
    concept: str | None,
    kind: str,
    reply: str,
    index: set[tuple[str, ...]],
    seen: set[str],
    full_text: str | None = None,
) -> str | None:
    """None if a probe text passes, else the failure cause. ``reply`` is the assistant turn
    (naming is checked there); ``full_text`` (system + user + reply) is checked for 8-grams and
    duplicates."""
    found = namer.found(reply)
    if kind == "explicit":
        if concept not in found:
            return "own_missing"
        if found - {concept}:
            return "names_other:" + ",".join(sorted(found - {concept}))
    elif found:  # implied and neutral name nothing in C (implied: not even its own concept)
        return "names_concept:" + ",".join(sorted(found))
    whole = full_text if full_text is not None else reply
    if ngrams(whole) & index:
        return "ngram8"
    if normalize_text(whole) in seen:
        return "duplicate"
    return None


# ---------------------------------------------------------------- generator (sibgen-chat-v1)

GEN_SYSTEM = """You write short probe texts for an interpretability study. Each text is a short \
chat exchange in the style of a casual user chatting with an AI assistant: a one-line system \
prompt, one user turn, and the assistant's reply (2-4 sentences, with ordinary punctuation). \
Every text is different in topic, wording and situation. For each text also give "mention": \
the verbatim span of the assistant reply where the concept is first named (explicit) or first \
made unambiguous (implied); an empty string for neutral texts. Return JSON only."""

KIND_RULES = {
    "explicit": "The assistant reply names {concept} explicitly (by its usual name) and names no "
    "other member of the class {cls}.",
    "implied": "The assistant reply makes {concept} unambiguous WITHOUT naming it (no name, "
    "alias or abbreviation of it), e.g. by a distinctive description, and names no other member "
    "of the class {cls}.",
    "neutral": "The assistant reply is about something unrelated and names no member of the "
    "class {cls}.",
}

GEN_SCHEMA = schema_block(
    "sibgen_chat",
    {
        "texts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["system", "user", "assistant", "mention"],
                "properties": {
                    "system": {"type": "string"},
                    "user": {"type": "string"},
                    "assistant": {"type": "string"},
                    "mention": {"type": "string"},
                },
            },
        }
    },
    ["texts"],
)


def gen_user(cls: str, concept: str | None, kind: str, n: int, avoid: Sequence[str]) -> str:
    rule = KIND_RULES[kind].format(concept=concept or "", cls=cls)
    return "\n".join(
        [
            f"CLASS: {cls}",
            *([f"CONCEPT: {concept}"] if concept else []),
            f"KIND: {kind}. {rule}",
            f"NUMBER OF TEXTS: {n}",
            "Members of the class you must not name (unless the rule says to name the concept): "
            + json.dumps(list(avoid), ensure_ascii=False),
        ]
    )


def mention_char(t: Mapping[str, str], kind: str) -> int | None:
    """Char offset (in the reply) of the concept's first mention; 0 for neutral texts; ``None``
    when the generator's ``mention`` span is not in the reply."""
    if kind == "neutral":
        return 0
    m = (t.get("mention") or "").strip()
    i = t["assistant"].find(m) if m else -1
    return i if i >= 0 else None


def full_text(t: Mapping[str, str]) -> str:
    return f"{t['system']}\n{t['user']}\n{t['assistant']}"


async def run_texts(
    concepts: Mapping[str, Mapping[str, Any]],
    surface: Mapping[str, Mapping[str, Sequence[str]]],
    index: set[tuple[str, ...]],
    llm: LLM | None,
    cache: Cache,
    *,
    rounds: int = 4,
) -> tuple[list[dict[str, Any]], Counter, Usage]:
    """Per class: 24 explicit + 24 implied per concept, 24 neutral; checked; up to ``rounds``
    top-up rounds. ``surface``: class -> concept -> aliases."""
    spend = Usage()
    kept: list[dict[str, Any]] = []
    causes: Counter = Counter()
    for cls, row in sorted(concepts.items()):
        C = list(row["concepts"])
        namer = Namer({c: list(surface.get(cls, {}).get(c, [])) for c in C})
        need = {(c, k): PER_KIND for c in C for k in ("explicit", "implied")}
        need[(None, "neutral")] = NEUTRAL_N
        have: dict[tuple[str | None, str], list[dict[str, Any]]] = {k: [] for k in need}
        seen: set[str] = set()
        for rnd in range(rounds):
            todo = [k for k, n in need.items() if len(have[k]) < n]
            if not todo:
                break

            async def one(
                k: tuple[str | None, str],
                rnd: int = rnd,
                cls: str = cls,
                n_want: int = 0,
                C: list[str] = C,
            ):
                c, kind = k
                st, res = await cached_json_call(
                    cache,
                    llm,
                    key=f"gen:{cls}:{c or 'neutral'}:{kind}:r{rnd}",
                    version=GEN_VERSION,
                    model=MODEL,
                    effort=EFFORT,
                    system=GEN_SYSTEM,
                    user=gen_user(cls, c, kind, n_want, C),
                    schema=GEN_SCHEMA,
                    validate=lambda r: isinstance(r.get("texts"), list),
                    spend=spend,
                )
                return k, (res or {}).get("texts") or [] if st == "ok" else []

            jobs = [one(k, n_want=need[k] - len(have[k]) + 6) for k in todo]
            for k, texts in await gather_limited(jobs):
                c, kind = k
                for t in texts:
                    if len(have[k]) >= need[k]:
                        break
                    why = text_check(namer, c, kind, t["assistant"], index, seen, full_text(t))
                    # explicit: the code-found first name; implied: the generator's span
                    mc = namer.first(t["assistant"], c) if kind == "explicit" and c else None
                    mc = mc if mc is not None else mention_char(t, kind)
                    if why is None and mc is None:
                        why = "mention_missing"
                    if why is not None:
                        causes[why.split(":")[0]] += 1
                        continue
                    seen.add(normalize_text(full_text(t)))
                    i = len(have[k])
                    have[k].append(
                        {
                            "id": f"{cls}|{c or 'neutral'}|{kind}|{i:02d}",
                            "class": cls,
                            "concept": c,
                            "kind": kind,
                            "request_idx": 0,
                            **{f: t[f] for f in ("system", "user", "assistant")},
                            "mention_char": mc,
                        }
                    )
        for k in need:
            kept.extend(have[k])
            if len(have[k]) < need[k]:
                causes[f"short:{cls}"] += need[k] - len(have[k])
    return kept, causes, spend
