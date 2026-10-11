"""The readout contract every eval reads, plus ``decode_token`` and ``verify_quote``.

Ported from WorkspaceBench (``src/wsbench/readouts.py`` and the helpers of
``src/wsbench/evals/hallucination/judge.py``) @ 92d763e, MIT. One JSONL row per cell::

    {"id": "<item id>", "layer": 36, "pos": 33, "samples": ["...", "..."]}
    {"id": "<item id>", "layer": 36, "pos": 33, "tokens": ["Ġword", "..."], "scores": [10.8, 9.9]}

Either row may carry an optional ``"token": "<read-site token>"`` string. A file is
all-prose or all-tokens. Malformed lines are counted, never fatal.
"""

import hashlib
import json
import re
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Kind = Literal["prose", "tokens"]
SKIP_CATEGORIES = (
    "malformed",
    "duplicate",
    "mixed_kind",
    "unknown_id",
    "layer_not_selected",
    "pos_not_selected",
)


@dataclass(frozen=True)
class Cell:
    id: str
    layer: int
    pos: int
    samples: tuple[str, ...] | None  # prose lens
    tokens: tuple[str, ...] | None  # top-k token lens (vocab strings, best first)
    scores: tuple[float, ...] | None
    token: str | None = None  # the read-site token string, when the producer recorded it

    @property
    def kind(self) -> Kind:
        return "prose" if self.samples is not None else "tokens"

    @property
    def key(self) -> str:
        return f"{self.id}__L{self.layer:03d}__p{self.pos}"

    @property
    def empty(self) -> bool:
        if self.samples is not None:
            return not any(s.strip() for s in self.samples)
        return not self.tokens


@dataclass
class LoadReport:
    kind: Kind | None = None
    n_rows: int = 0
    skipped: dict[str, int] = field(default_factory=lambda: dict.fromkeys(SKIP_CATEGORIES, 0))
    n_empty: int = 0
    layers: list[int] = field(default_factory=list)


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x: Any) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool)


def _str_list(x: Any) -> bool:
    return isinstance(x, list) and all(isinstance(s, str) for s in x)


def _parse_row(row: Any) -> Cell | None:
    """A validated Cell, or None if the row is malformed."""
    if not isinstance(row, dict):
        return None
    id_, layer, pos = row.get("id"), row.get("layer"), row.get("pos")
    if not isinstance(id_, str) or not _is_int(layer) or not _is_int(pos):
        return None
    has_samples, has_tokens = "samples" in row, "tokens" in row
    if has_samples == has_tokens:
        return None
    token = row.get("token")
    if token is not None and not isinstance(token, str):
        return None
    if has_samples:
        if "scores" in row or not _str_list(row["samples"]):
            return None
        return Cell(id_, layer, pos, tuple(row["samples"]), None, None, token)
    tokens = row["tokens"]
    if not _str_list(tokens):
        return None
    scores: tuple[float, ...] | None = None
    if "scores" in row:
        sc = row["scores"]
        if not isinstance(sc, list) or len(sc) != len(tokens) or not all(_is_num(v) for v in sc):
            return None
        scores = tuple(float(v) for v in sc)
    return Cell(id_, layer, pos, None, tuple(tokens), scores, token)


def load_readouts(
    path: Path,
    *,
    ids: Collection[str] | None = None,
    layers: Collection[int] | None = None,
    positions: Mapping[str, Collection[int]] | None = None,
) -> tuple[list[Cell], LoadReport]:
    rep = LoadReport()
    id_set = set(ids) if ids is not None else None
    layer_set = set(layers) if layers is not None else None
    pos_sets = {k: set(v) for k, v in positions.items()} if positions is not None else None
    cells: list[Cell] = []
    seen: set[tuple[str, int, int]] = set()
    for line in Path(path).read_bytes().decode("utf-8", "replace").splitlines():
        if not line.strip():
            continue
        rep.n_rows += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            rep.skipped["malformed"] += 1
            continue
        cell = _parse_row(row)
        if cell is None:
            rep.skipped["malformed"] += 1
            continue
        if rep.kind is None:
            rep.kind = cell.kind
        elif cell.kind != rep.kind:
            rep.skipped["mixed_kind"] += 1
            continue
        if id_set is not None:
            if cell.id not in id_set:
                rep.skipped["unknown_id"] += 1
                continue
        elif pos_sets is not None and cell.id not in pos_sets:
            rep.skipped["unknown_id"] += 1
            continue
        if layer_set is not None and cell.layer not in layer_set:
            rep.skipped["layer_not_selected"] += 1
            continue
        if pos_sets is not None and cell.id in pos_sets and cell.pos not in pos_sets[cell.id]:
            rep.skipped["pos_not_selected"] += 1
            continue
        k = (cell.id, cell.layer, cell.pos)
        if k in seen:
            rep.skipped["duplicate"] += 1
            continue
        seen.add(k)
        if cell.empty:
            rep.n_empty += 1
        cells.append(cell)
    rep.layers = sorted({c.layer for c in cells})
    return cells, rep


def expected_cells(
    positions: Mapping[str, Collection[int]], layers: Collection[int]
) -> set[tuple[str, int, int]]:
    return {(id_, layer, pos) for id_, ps in positions.items() for layer in layers for pos in ps}


def missing_cells(
    cells: Iterable[Cell], expected: set[tuple[str, int, int]]
) -> list[tuple[str, int, int]]:
    present = {(c.id, c.layer, c.pos) for c in cells}
    return sorted(expected - present)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ quote verification
# Verbatim from wsbench ``evals/hallucination/judge.py``.

_WS = re.compile(r"\s+")
# normalising the typographic forms IS the point here
_QUOTES = str.maketrans(
    {"“": '"', "”": '"', "‘": "'", "’": "'", "«": '"', "»": '"'}
)


def normalize_ws(s: str) -> str:
    return _WS.sub(" ", s.translate(_QUOTES)).strip().lower()


def verify_quote(quote: str, haystacks: Iterable[str]) -> bool:
    """True when ``quote`` is a verbatim span (whitespace-, quote-mark- and case-normalised, at
    least 3 characters) of one of ``haystacks``."""
    q = normalize_ws(quote)
    if len(q) < 3:
        return False
    return any(q in normalize_ws(h) for h in haystacks)


# ------------------------------------------------------------------ byte-level BPE -> text


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte-level BPE table (Qwen's tokenizer uses the same): byte -> printable char."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs, strict=True)}


_U2B = {c: b for b, c in _bytes_to_unicode().items()}


def decode_token(tok: str) -> str:
    """Byte-level BPE vocabulary string -> text, char by char: byte-table chars back to their byte,
    anything else (e.g. a literal space) to its own UTF-8 bytes. Pre-decoded non-ASCII text that
    happens to use byte-table characters (Latin-1 letters) is mangled — pass vocabulary strings."""
    raw = b"".join(bytes([_U2B[ch]]) if ch in _U2B else ch.encode("utf-8") for ch in tok)
    return raw.decode("utf-8", errors="replace")
