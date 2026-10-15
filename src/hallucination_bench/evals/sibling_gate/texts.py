"""Plan 0009 §3b-c: probe texts (``sibgen-chat-v2``), their code checks, the read positions,
and the token-class key (rev 8 / 8.1).

Texts are short wsbench_chat-style exchanges (system, user turn, assistant reply). Per concept of
C: 24 ``explicit`` (the reply names the concept and no other concept of C) and 24 ``implied``
(the reply makes the concept unambiguous without naming it); per class 24 ``neutral`` (the reply
names nothing in C; not a probe class). Rev 8: the replies follow the wsbench_chat house style
(a short paragraph, then a bold-labelled bullet list or a ``###`` heading; ``!``, ``?`` and
parentheses where natural; the concept mentioned early), so the probe bank holds the tokens the
eval sites sit on. Code checks (``um_ladder.text_check`` adapted to C): naming as specified, no
word 8-gram shared with any wsbench_chat item text, no duplicates.

**New recipe** (stated as such): centring and the null are keyed by the read position's token
class (:func:`token_class`, the same function for read positions and eval sites), through
``um_probe._key`` / ``eval_key`` with ``ending = 0`` and ``offset = key``. Merging into
``other`` is per concept class and fixed across layers (:func:`class_merge`, < 400 implied-text
null positions). Read positions: reply tokens of a key class after the concept's first mention,
up to 32 per text and at most 8 per (merged) key.
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

GEN_VERSION = "sibgen-chat-v2"  # rev 8: wsbench_chat house style (v1 = plain prose)
PER_KIND = 24
NEUTRAL_N = 24
MAX_READ = 32
PER_KEY_READ = 8
NGRAM = 8
CACHE_PATH = U.OUT / "texts" / "cache.jsonl"

# ---------------------------------------------------------------- token classes (rev 8 / 8.1)

TOKEN_CLASSES: tuple[str, ...] = (
    "comma",  # , ，
    "period",  # . 。
    "newline",  # any token made only of newlines and spaces (Ċ, ĊĊ, ' ĊĊ', '  Ċ')
    "bold",  # any token containing ** (' **', ':**', '**:', '**.', '?**', ' **"')
    "heading",  # # ### ####
    "exclaim",  # ! ！
    "question",  # ? ？
    "colon",  # : ：  (alone; ':**' is bold)
    "dash",  # bullet / dash: - — –
    "open_bracket",  # ( （
    "close_bracket",  # ) ）  ('),', ').', '):')
    "quote",  # " “ ”
    "star",  # italic / star: a lone '*' / ' *' (in wsbench_chat mostly the '*   ' bullet)
    "other",
)
OTHER = TOKEN_CLASSES.index("other")
N_CLASSES = len(TOKEN_CLASSES)
MIN_NULL_POSITIONS = 400  # a key class with fewer implied-text null positions merges into other

# first non-space mark -> class (a token containing ** is bold before this table is consulted)
_MARK_CLASS = {
    **dict.fromkeys(",，", "comma"),
    **dict.fromkeys(".。", "period"),
    **dict.fromkeys("#", "heading"),
    **dict.fromkeys("!！", "exclaim"),
    **dict.fromkeys("?？", "question"),
    **dict.fromkeys(":：", "colon"),
    **dict.fromkeys("-—–", "dash"),
    **dict.fromkeys("(（", "open_bracket"),
    **dict.fromkeys(")）", "close_bracket"),
    **dict.fromkeys('"“”', "quote"),
    **dict.fromkeys("*", "star"),
}


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte-level BPE alphabet (byte -> printable char)."""
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs, strict=True)}


_BYTE_OF = {c: b for b, c in _bytes_to_unicode().items()}


def decode_bpe(tok: str) -> str:
    """Byte-level BPE string -> text ('âĢĶ' -> '—', 'Ċ' -> '\\n', 'Ġ' -> ' '). The readouts mix
    BPE and decoded characters (' ĊĊ'), so a char outside the BPE alphabet is kept as its UTF-8
    bytes. A string that is not valid BPE (already-decoded text such as 'é') is returned as is."""
    if not any(c in _BYTE_OF and not c.isascii() for c in tok):
        return tok  # pure ASCII / no BPE-only chars: nothing to decode
    raw = b"".join(bytes([_BYTE_OF[c]]) if c in _BYTE_OF else c.encode() for c in tok)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return tok


def token_class(tok: str) -> int:
    """Class index of a token (decoded text or byte-level BPE), the one key for both probe read
    positions and eval sites. Normalise: decode BPE, strip whitespace; a token made only of
    newlines and spaces is ``newline``; a token containing ``**`` is ``bold``; else the class of
    its first mark (``.,`` -> period, ``).`` -> close_bracket, ``?)`` -> question, ``*)`` ->
    star); ``other`` when that mark has no class (``;``, ``/``, ``&``, ``'``, ``__``, words)."""
    s = decode_bpe(tok)
    core = s.strip()
    if not core:
        return TOKEN_CLASSES.index("newline") if ("\n" in s or "\r" in s) else OTHER
    if "**" in core:
        return TOKEN_CLASSES.index("bold")
    name = _MARK_CLASS.get(core[0])
    return TOKEN_CLASSES.index(name) if name else OTHER


def is_read_token(tok: str) -> bool:
    """A probe read-position candidate: after decoding, only whitespace, punctuation and
    non-emoji symbols, with at least one mark or a newline (a pure-space token is not one)."""
    s = decode_bpe(tok)
    if not s or not s.strip(" \t"):
        return False
    return all(
        ch.isspace()
        or (cat := unicodedata.category(ch)).startswith("P")
        or cat in ("Sm", "Sc", "Sk")
        for ch in s
    )


def select_capped(
    keys: Sequence[int], max_n: int = MAX_READ, per_key: int = PER_KEY_READ
) -> list[int]:
    """Indices into ``keys`` (in order): at most ``per_key`` per key, ``max_n`` in all."""
    out: list[int] = []
    n: Counter = Counter()
    for i, k in enumerate(keys):
        if len(out) >= max_n:
            break
        if n[k] < per_key:
            n[k] += 1
            out.append(i)
    return out


def class_merge(
    implied_keys: Iterable[Sequence[int]],
    min_n: int = MIN_NULL_POSITIONS,
    max_n: int = MAX_READ,
    per_key: int = PER_KEY_READ,
) -> dict[int, int]:
    """Raw class -> key for ONE concept class, fixed across layers: a key class with fewer than
    ``min_n`` implied-text null positions in the class's bank merges into ``other``. The
    per-text caps apply after merging, so this iterates to a fixed point (keys only ever move
    into ``other``). ``implied_keys``: per implied text, the raw classes of its candidates."""
    texts = [list(k) for k in implied_keys]
    merge = {c: c for c in range(N_CLASSES)}
    while True:
        counts: Counter = Counter()
        for ks in texts:
            mk = [merge[k] for k in ks]
            counts.update(mk[i] for i in select_capped(mk, max_n, per_key))
        bad = {c for c in range(N_CLASSES) if c != OTHER and merge[c] == c and counts[c] < min_n}
        if not bad:
            return merge
        merge = {c: (OTHER if m in bad else m) for c, m in merge.items()}


def class_key(
    tokens: Sequence[str], merge: Mapping[int, int] | None = None
) -> tuple[list[int], list[int]]:
    """(ending, offset) arrays for ``um_probe._key`` / ``eval_key``: ending 0, offset = key."""
    off = [token_class(t) for t in tokens]
    if merge is not None:
        off = [merge.get(o, OTHER) for o in off]
    return [0] * len(off), off


def candidates(
    tokens: Sequence[str], starts: Sequence[int], reply_start: int, mention_char: int
) -> list[int]:
    """Token indices of read-token candidates of the reply at/after ``mention_char`` (a char
    offset into the rendered text; ``starts`` = each token's char start)."""
    lo = max(reply_start, mention_char)
    return [
        i
        for i, (t, s) in enumerate(zip(tokens, starts, strict=True))
        if s >= lo and is_read_token(t)
    ]


def read_positions(
    tokens: Sequence[str],
    starts: Sequence[int],
    reply_start: int,
    mention_char: int,
    merge: Mapping[int, int] | None = None,
    max_n: int = MAX_READ,
    per_key: int = PER_KEY_READ,
) -> list[int]:
    """§3b read positions: reply tokens of a key class after the first mention, at most
    ``per_key`` per (merged) key and ``max_n`` per text."""
    cand = candidates(tokens, starts, reply_start, mention_char)
    _e, keys = class_key([tokens[i] for i in cand], merge)
    return [cand[j] for j in select_capped(keys, max_n, per_key)]


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


# ---------------------------------------------------------------- generator (sibgen-chat-v2)

GEN_SYSTEM = """You write short probe texts for an interpretability study. Each text is a short \
chat exchange in the style of a casual user chatting with an AI assistant: a one-line system \
prompt, one user turn, and the assistant's reply. Write the reply in a typical chat-assistant \
house style, in Markdown: a short opening paragraph (1-3 sentences) that mentions the concept \
early, then EITHER a bullet list of 2-4 items with bold labels ("- **Label:** text" or \
"* **Label**: text") OR a "### " heading followed by a sentence or two. Use "!", "?" and \
parentheses where they come naturally, plus ordinary commas and full stops; separate the \
paragraph from the list or heading with a blank line. Keep each reply under 120 words. Every \
text is different in topic, wording, situation and layout. For each text also give "mention": \
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

    async def one_class(cls: str, row: Mapping[str, Any]):
        kept: list[dict[str, Any]] = []
        causes: Counter = Counter()
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
        return kept, causes

    # classes are independent (own ``seen`` set, own cache keys): run them concurrently (the
    # LLM semaphore bounds the calls in flight); the output order is the sorted class order
    kept_all: list[dict[str, Any]] = []
    causes_all: Counter = Counter()
    for k, c in await gather_limited([one_class(cl, rw) for cl, rw in sorted(concepts.items())]):
        kept_all.extend(k)
        causes_all.update(c)
    return kept_all, causes_all, spend
