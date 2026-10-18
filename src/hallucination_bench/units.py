"""The deterministic readout splitter and the offset-returning ``locate`` (plan 0002, §Units).

Code pre-splits each readout into **units** (``R{readout}.U{n}``, ``n`` from 1) with raw
character offsets; together they cover every non-whitespace character of the readout. The
claim-map judge labels each unit and may split one into quoted sub-spans, which code finds with
:func:`locate`.

Splitter rules (applied to the whole readout, then trimmed; whitespace-only pieces dropped):

* split on newlines; a leading bullet / numbered marker (``- ``, ``* ``, ``2. ``, ``2: ``,
  ``1)``, possibly nested like ``- 2. ``) stays with its line;
* split after ``. ! ? 。 ！ ？ …`` only when followed by whitespace or the end of the text, never
  after a digit (``2.``, decimals), after a known abbreviation (``ABBREVIATIONS``) or inside an
  open quotation (``"…"``, ``“…”``, ``「…」``); quote state resets at every newline and a straight
  ``"`` toggles it, so a stray quote cannot swallow more than its own line;
* split after ``;`` ``；`` and after ``:`` ``：`` only when followed by whitespace (URLs and times
  stay whole); the quotation rule applies to these too;
* markup tags (``<explanation>``, ``</response>``) are their own units;
* established v5c spans are located first (:func:`locate_established`) and each becomes its own
  unit: no other cut falls inside one, and their edges are cuts.
"""

import itertools
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from hallucination_bench.readouts import _QUOTES, normalize_ws

TERMINATORS = frozenset(".!?。！？…")
SEMICOLONS = frozenset(";；")
COLONS = frozenset(":：")
ABBREVIATIONS: tuple[str, ...] = (
    "e.g.",
    "i.e.",
    "etc.",
    "vs.",
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "st.",
    "u.s.",
    "prof.",
    "cf.",
    "approx.",
)
_MARKER = re.compile(r"[ \t]*(?:(?:[-*•+]|\d+[.:)])[ \t]+)+")
_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9_:-]*\s*/?>")


@dataclass(frozen=True)
class Unit:
    id: str
    readout: int
    n: int
    start: int
    end: int
    text: str
    established: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def unit_id(readout: int, n: int) -> str:
    return f"R{readout}.U{n}"


# ---------------------------------------------------------------- locate


class Normalised:
    """``normalize_ws`` of a haystack plus a map from each normalised char to its raw index.

    Same normalisation as ``verify_quote`` (typographic quotes -> straight, whitespace runs ->
    one space, lowercase); lowercasing can change length (``İ``), hence the per-char map."""

    def __init__(self, text: str):
        self.text = text
        chars: list[str] = []
        idx: list[int] = []
        for i, ch in enumerate(text):
            ch = ch.translate(_QUOTES)
            if ch.isspace():
                if chars and chars[-1] == " ":
                    continue
                chars.append(" ")
                idx.append(i)
                continue
            for c in ch.lower():
                chars.append(c)
                idx.append(i)
        self.norm = "".join(chars)
        self.idx = idx

    def norm_pos(self, raw: int) -> int:
        """The first normalised position whose raw index is >= ``raw``."""
        lo, hi = 0, len(self.idx)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.idx[mid] < raw:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def find(self, sub: str, start: int = 0) -> tuple[int, int] | None:
        needle = normalize_ws(sub)
        if not needle:
            return None
        j = self.norm.find(needle, self.norm_pos(start))
        if j < 0:
            return None
        return self.idx[j], self.idx[j + len(needle) - 1] + 1


def locate(sub: str, text: str | Normalised, start: int = 0) -> tuple[int, int] | None:
    """Raw ``(start, end)`` of the first match of ``sub`` in ``text`` at or after raw offset
    ``start`` (``verify_quote`` normalisation, no minimum length), or ``None``."""
    n = text if isinstance(text, Normalised) else Normalised(text)
    return n.find(sub, start)


def locate_established(text: str, spans: Sequence[str]) -> tuple[list[tuple[int, int]], int]:
    """Locate a readout's established (v5c verified wrong) strings: each takes its first
    occurrence not already taken by an equal string, left to right; overlapping or nested
    matches are merged into one span. Returns ``(sorted disjoint spans, n_unlocated)``."""
    norm = Normalised(text)
    taken: dict[str, list[tuple[int, int]]] = {}
    found: list[tuple[int, int]] = []
    n_unlocated = 0
    for s in spans:
        key = normalize_ws(s)
        first = norm.find(s, 0)
        if first is None:
            n_unlocated += 1
            continue
        m: tuple[int, int] | None = first
        used = taken.setdefault(key, [])
        while m is not None and m in used:
            m = norm.find(s, m[0] + 1)
        m = m or first  # every occurrence already taken: the same span listed twice
        used.append(m)
        found.append(m)
    merged: list[tuple[int, int]] = []
    for a, b in sorted(found):
        if merged and a < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged, n_unlocated


# ---------------------------------------------------------------- splitter


def _is_abbrev(text: str, i: int) -> bool:
    """Does the ``.`` at ``i`` end a known abbreviation (not preceded by a letter)?"""
    low = text[: i + 1].lower()
    for a in ABBREVIATIONS:
        if low.endswith(a):
            j = i + 1 - len(a)
            if j == 0 or not text[j - 1].isalpha():
                return True
    return False


def _cuts(text: str) -> set[int]:
    """Cut positions (boundaries) of the free-text rules over the whole readout."""
    cuts: set[int] = {0, len(text)}
    n = len(text)
    line_start = 0
    while line_start <= n:
        nl = text.find("\n", line_start)
        line_end = n if nl < 0 else nl
        cuts.add(line_start)
        cuts.add(line_end)
        m = _MARKER.match(text, line_start, line_end)
        protect = m.end() if m else line_start
        for t in _TAG.finditer(text, line_start, line_end):
            cuts.add(t.start())
            cuts.add(t.end())
        straight = False
        curly = corner = 0
        for i in range(line_start, line_end):
            ch = text[i]
            if ch == '"':
                straight = not straight
                continue
            if ch == "“":
                curly += 1
                continue
            if ch == "”":
                curly = max(0, curly - 1)
                continue
            if ch == "「":
                corner += 1
                continue
            if ch == "」":
                corner = max(0, corner - 1)
                continue
            if i < protect or straight or curly or corner:
                continue
            nxt_ws = i + 1 >= n or text[i + 1].isspace()
            if ch in TERMINATORS:
                if not nxt_ws:
                    continue
                if i > 0 and text[i - 1].isdigit():
                    continue
                if ch == "." and _is_abbrev(text, i):
                    continue
                cuts.add(i + 1)
            elif ch in SEMICOLONS or (ch in COLONS and nxt_ws):
                cuts.add(i + 1)
        if nl < 0:
            break
        line_start = nl + 1
    return cuts


def _trim(text: str, a: int, b: int) -> tuple[int, int]:
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    return a, b


def split_readout(
    text: str, readout: int = 0, established: Sequence[str] = ()
) -> tuple[list[Unit], int]:
    """``(units, n_unlocated_established)`` for one readout."""
    est, n_unloc = locate_established(text, established)
    cuts = _cuts(text)
    cuts = {c for c in cuts if not any(a < c < b for a, b in est)}
    for a, b in est:
        cuts.add(a)
        cuts.add(b)
    est_set = set(est)
    bounds = sorted(cuts)
    units: list[Unit] = []
    for a, b in itertools.pairwise(bounds):
        s, e = _trim(text, a, b)
        if s >= e:
            continue
        units.append(
            Unit(
                unit_id(readout, len(units) + 1),
                readout,
                len(units) + 1,
                s,
                e,
                text[s:e],
                (s, e) in est_set or (a, b) in est_set,
            )
        )
    return units, n_unloc


def split_cell(
    samples: Sequence[str], established: Sequence[Sequence[str]] | None = None
) -> tuple[list[Unit], list[int]]:
    """Units of every readout of a cell (ids ``R{i}.U{n}``) and the per-readout
    ``n_unlocated_established``."""
    units: list[Unit] = []
    unloc: list[int] = []
    for i, s in enumerate(samples):
        u, n = split_readout(s, i, established[i] if established else ())
        units += u
        unloc.append(n)
    return units, unloc
