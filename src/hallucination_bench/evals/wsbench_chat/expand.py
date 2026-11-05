"""Plan 0028: the 3x main bank (``wsbench_chat_x3``) — seed screen, read sites, thinning.

Literal ports of the rules that built the 149-item WorkspaceBench chat bank, so new items are
drawn and sited exactly as the old ones were:

- seed screen / content hash / seed key: global-workspace
  ``scripts/oracle_lens_evals/jlens_pr/items.py`` (``screen_chat_seed``, ``content_hash_hex``,
  ``seed_key``) and the pool rule of ``jlens_pr/build_seeds.py``;
- read sites: ``global_workspace.judges.hallucination_bench`` (``is_summary_token``,
  ``site_kind``, ``response_sites``) and the thinning of
  ``scripts/oracle_lens_evals/hallucination/chat_bank.py`` (``thin``).

``thin(sites, 16)`` keeps every site ``thin(sites, 8)`` keeps: index ``int(i * n / 8)`` equals
``int(2i * n / 16)`` exactly (halving a float is exact), so the WSB sites stay a subset.
"""

from __future__ import annotations

import hashlib
import random
import re
import unicodedata
from collections.abc import Sequence
from typing import Any, Literal

MAX_SITES_WSB = 8  # the WorkspaceBench bank (HAL_MAX_SITES)
MAX_SITES_X3 = 16  # plan 0028: 2x the read positions per prompt

# ------------------------------------------------------------------ seeds (jlens_pr/items.py)

_CODE_OR_URL = re.compile(r"```|https?://")
_NAME_PLACEHOLDER = re.compile(r"NAME_\d+")


def content_hash_hex(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def seed_key(cell: str, source: str, text: str) -> str:
    return f"{cell}:{source}:{content_hash_hex(text)}"


def _mostly_ascii(text: str, thresh: float = 0.98) -> bool:
    if not text:
        return False
    return sum(ch.isascii() for ch in text) / len(text) >= thresh


def screen_chat_seed(text: str, *, max_chars: int = 2000, min_chars: int = 20) -> bool:
    t = text.strip()
    if not (min_chars <= len(t) <= max_chars):
        return False
    if _CODE_OR_URL.search(t) or _NAME_PLACEHOLDER.search(t):
        return False
    return _mostly_ascii(t)


# ------------------------------------------------------------------ sites (hallucination_bench)

SiteKind = Literal["sentence", "clause", "quote", "newline", "markup", "eos"]
_SENTENCE = set(".!?…")
_CLAUSE = set(",;:—–-")  # typographic dashes are exactly what we detect
_QUOTE = set("\"'“”‘’«»")  # ditto for curly quotes


def is_summary_token(text: str) -> bool:
    """Non-empty after stripping spaces and made only of Unicode punctuation (``P*``) and/or
    whitespace."""
    s = text.strip(" ")
    return bool(s) and all(unicodedata.category(c).startswith("P") or c.isspace() for c in s)


def site_kind(text: str) -> SiteKind | None:
    if not is_summary_token(text):
        return None
    s = text.strip(" ")
    chars = set(s)
    if chars & _SENTENCE:
        return "sentence"
    if chars & _CLAUSE:
        return "clause"
    if chars & _QUOTE:
        return "quote"
    if s.isspace():
        return "newline"
    return "markup"


def response_sites(
    tokens: Sequence[str], response_start: int, *, eos_pos: int | None = None
) -> list[dict[str, Any]]:
    """Every summary-site token at a position ``>= response_start`` (+ ``eos`` at ``eos_pos``)."""
    out: list[dict[str, Any]] = []
    end = eos_pos if eos_pos is not None else len(tokens)
    char = 0
    for pos in range(max(0, response_start), len(tokens)):
        if pos < end:
            char += len(tokens[pos])
        if eos_pos is not None and pos == eos_pos:
            continue
        kind = site_kind(tokens[pos])
        if kind is not None:
            out.append({"pos": pos, "token": tokens[pos], "kind": kind, "char": char})
    if eos_pos is not None and 0 <= eos_pos < len(tokens):
        out.append({"pos": eos_pos, "token": tokens[eos_pos], "kind": "eos", "char": char})
    out.sort(key=lambda s: int(s["pos"]))
    return out


def thin(sites: list[dict[str, Any]], max_sites: int) -> list[dict[str, Any]]:
    """``chat_bank.thin``: ``step = len / max``, keep ``sites[int(i * step)]``."""
    if max_sites and len(sites) > max_sites:
        step = len(sites) / max_sites
        return [sites[int(i * step)] for i in range(max_sites)]
    return sites


def item_sites(decode, ids: Sequence[int], prompt_len: int, max_sites: int) -> list[dict[str, Any]]:
    """The bank's sites for one rollout (``chat_bank.main``): per-token decodes pick the sites
    (no eos site: rollouts carry no ``<|im_end|>``), whole-span decodes give ``char``.
    ``decode(ids) -> str`` must keep special tokens."""
    tokens = [decode([t]) for t in ids]
    sites = thin(response_sites(tokens, prompt_len, eos_pos=None), max_sites)
    response = decode(list(ids[prompt_len:]))
    out = []
    for s in sites:
        prefix = decode(list(ids[prompt_len : s["pos"] + 1]))
        if not response.startswith(prefix):
            raise ValueError(f"site {s} prefix is not a prefix of the response")
        out.append({**s, "char": len(prefix)})
    return out


# ------------------------------------------------------------------ the plan-0028 expansion draw

# Camila 2026-11-04: NEW lmsys seeds must also carry lmsys-chat-1m's own language == "English"
# (on top of the ASCII screen); the WSB picks (some non-English) and dailydialog are unchanged.
ENGLISH_ONLY_SOURCES = frozenset({"lmsys"})


def expansion_draw(
    pool: list[dict[str, Any]], picked: set[str], source: str, n_new: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """The new seeds of one source: the pool minus the WSB picks (by key), NFC-only (the Qwen
    tokenizer NFC-normalises, so a non-NFC prompt would not round-trip), English-only for
    ``ENGLISH_ONLY_SOURCES`` (``language == "English"``), then shuffled with
    ``random.Random("hb0028:<source>")`` and cut to ``n_new``, in that seeded order."""
    rest = [s for s in pool if s["key"] not in picked]
    n_non_nfc = sum(unicodedata.normalize("NFC", s["text"]) != s["text"] for s in rest)
    rest = [s for s in rest if unicodedata.normalize("NFC", s["text"]) == s["text"]]
    counts = {"non_nfc_excluded": n_non_nfc}
    if source in ENGLISH_ONLY_SOURCES:
        n_before = len(rest)
        rest = [s for s in rest if s.get("language") == "English"]
        counts["non_english_excluded"] = n_before - len(rest)
    random.Random(f"hb0028:{source}").shuffle(rest)
    return rest[:n_new], counts
