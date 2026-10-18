"""The sibling-probe bank: items, coverage rule, leak / 8-gram / window-identity checks, and the
ReadSpec plan-file writers (plan 0003 §Phase B, §Phase C, §Phase E).

``data/sibling_probe/items.json``::

    {"version", "provenance", "counts": {...},
     "items": [{"id", "category", "gold", "source": "wsbench"|"pilot"|"generated",
                "messages": [system, user], "cue", "request", "request_idx",
                "named": [concept ids named in the user turn],
                "implication": {"answer", "pass"} | null,   # generated items only
                "response": str | null}]}

``data/sibling_probe/probe_texts.jsonl``: one row per text
``{"id", "category", "concept" (null for neutral), "kind": "explicit"|"implied"|"neutral",
"request_idx", "context", "messages"}``.

Every message list is ``[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user",
"content": cue + " " + request}]``: the request sentence follows the cue after ONE space, so the
``from_last_sentence_start`` read window (request sentence + chat-template tail) is
token-identical across all rows of a (category, request).
"""

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from hallucination_bench import DATA, paths
from hallucination_bench.evals.sibling_probe import SYSTEM_PROMPT
from hallucination_bench.evals.sibling_probe.concepts import Category, Matcher

BANK_DIR = DATA / "sibling_probe"
ITEMS_PATH = BANK_DIR / "items.json"
PROBE_TEXTS_PATH = BANK_DIR / "probe_texts.jsonl"
WSBENCH_UM_ITEMS = paths.WSBENCH_PUB / "evals" / "user_modeling" / "items.json"

CAPTURE_LAYERS = (28, 36, 42, 44, 52)  # both captures (eval + probe)
S3D_LAYERS = (28, 36, 44, 52)  # fanout_custom (O-lens)
EVAL_FAMILY_PREFIX = "sibling"
PROBE_FAMILY_PREFIX = "sibprobe"
POSITIONS = {"kind": "from_last_sentence_start"}
BANK_MIN, BANK_MAX, GEN_CAP_PER_GOLD = 20, 32, 3
NGRAM_N = 8
_ID_OK = re.compile(r"^[A-Za-z0-9._-]+$")


# ------------------------------------------------------------------ messages


def messages_for(cue: str, request: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{cue} {request}"},
    ]


def user_text(messages: Sequence[Mapping[str, str]]) -> str:
    return next(m["content"] for m in reversed(messages) if m["role"] == "user")


def split_cue(text: str, requests: Sequence[str]) -> tuple[str, int]:
    """(cue, request index) of a user turn ``cue + " " + request``; ValueError otherwise."""
    for i, r in enumerate(requests):
        if text.endswith(" " + r):
            return text[: -len(r) - 1], i
    raise ValueError(f"user turn does not end with ' ' + a request sentence: {text[-80:]!r}")


def normalize_cue(text: str) -> str:
    return " ".join(text.lower().replace("\u2019", "'").split())


def cue_ok(cue: str) -> bool:
    """A cue ends with sentence-final punctuation (no trailing quote / bracket)."""
    return bool(re.search(r"[A-Za-z0-9)][.!?]$", cue.strip())) and cue == cue.strip()


# ------------------------------------------------------------------ checks


def leaks(text: str, matcher: Matcher) -> set[str]:
    """Concepts of the matcher's category named in ``text`` (the leak check: generated items
    must name none)."""
    return matcher.mentions(text)


def word_ngrams(text: str, n: int = NGRAM_N) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower().replace("’", "'"))
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def ngram_index(texts: Iterable[str], n: int = NGRAM_N) -> set[tuple[str, ...]]:
    out: set[tuple[str, ...]] = set()
    for t in texts:
        out |= word_ngrams(t, n)
    return out


def shares_ngram(text: str, index: set[tuple[str, ...]], n: int = NGRAM_N) -> bool:
    return not word_ngrams(text, n).isdisjoint(index)


# ------------------------------------------------------------------ coverage rule


def bank_size(n_concepts: int) -> int:
    """Total items per category: max(20, 2 x concepts), capped at 32."""
    return min(BANK_MAX, max(BANK_MIN, 2 * n_concepts))


def allocate(
    concept_ids: Sequence[str],
    fixed: Mapping[str, int],
    total: int,
    *,
    cap: int = GEN_CAP_PER_GOLD,
    available: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """Generated items per gold concept: fill ``total - sum(fixed)`` slots one at a time, each to
    the concept with the fewest items so far (fixed + generated; ties by inventory order),
    never more than ``cap`` generated (or ``available[c]``) for one concept."""
    gen = dict.fromkeys(concept_ids, 0)
    order = {c: i for i, c in enumerate(concept_ids)}
    limit = {c: min(cap, (available or {}).get(c, cap)) for c in concept_ids}
    for _ in range(max(0, total - sum(fixed.get(c, 0) for c in concept_ids))):
        elig = [c for c in concept_ids if gen[c] < limit[c]]
        if not elig:
            break
        c = min(elig, key=lambda c: (fixed.get(c, 0) + gen[c], order[c]))
        gen[c] += 1
    return gen


def coverage_problems(items: Sequence[Mapping[str, Any]], cat: Category) -> list[str]:
    """The coverage rule on a finished category bank."""
    mine = [it for it in items if it["category"] == cat.name]
    errs = []
    by_gold = Counter(it["gold"] for it in mine)
    gen_by_gold = Counter(it["gold"] for it in mine if it["source"] == "generated")
    for c in cat.ids:
        if by_gold[c] < 1:
            errs.append(f"{cat.name}: concept {c} is the gold of no item")
        if gen_by_gold[c] > GEN_CAP_PER_GOLD:
            errs.append(f"{cat.name}: {gen_by_gold[c]} generated items for {c}")
    unknown = set(by_gold) - set(cat.ids)
    if unknown:
        errs.append(f"{cat.name}: golds outside the inventory {sorted(unknown)}")
    if len(mine) > max(BANK_MAX, sum(1 for it in mine if it["source"] != "generated")):
        errs.append(f"{cat.name}: {len(mine)} items > {BANK_MAX}")
    ids = [it["id"] for it in mine]
    if len(ids) != len(set(ids)):
        errs.append(f"{cat.name}: duplicate ids")
    return errs


# ------------------------------------------------------------------ render + window identity
# ``display_tokens`` and ``from_last_sentence_start`` are ports of wsbench-pub @92d763e
# ``wsbench.produce.render.display_tokens`` / ``wsbench.readplan._from_last_sentence_start``; a
# test checks them against the source when the checkout is present.


def display_tokens(tokenizer: Any, ids: Sequence[int]) -> list[str]:
    return [
        t.replace("Ġ", " ").replace("▁", " ") if isinstance(t, str) else ""
        for t in tokenizer.convert_ids_to_tokens(list(ids))
    ]


def _clean(tok: str) -> str:
    return tok.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\n")


def from_last_sentence_start(tokens: Sequence[str]) -> list[int]:
    n = len(tokens)
    ends = [i for i, t in enumerate(tokens) if t == "<|im_end|>"]
    starts = [
        i
        for i, t in enumerate(tokens)
        if t == "<|im_start|>" and i + 1 < n and tokens[i + 1] == "user"
    ]
    if not starts or not ends:
        return list(range(n))
    u0 = starts[-1]
    u1 = next((e for e in ends if e > u0), n)
    body = list(range(u0 + 3, u1))
    if not body:
        return list(range(u1, n))
    i = body[-1] - 1
    while i > body[0]:
        if re.search(r"[.!?]\s*$", _clean(tokens[i])):
            break
        i -= 1
    start = i + 1 if i > body[0] else body[0]
    return list(range(start, n))


def render_ids(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> list[int]:
    """The ``chat_context`` render: chat template, generation prompt, thinking off."""
    out = tokenizer.apply_chat_template(
        list(messages), tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    return list(out["input_ids"] if hasattr(out, "keys") else out)


def window_ids(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> list[int]:
    ids = render_ids(tokenizer, messages)
    pos = from_last_sentence_start(display_tokens(tokenizer, ids))
    return [ids[p] for p in pos]


def canonical_window_ids(tokenizer: Any, request: str) -> list[int]:
    """The request sentence tokenised after one space, plus the template tail (from the user
    turn's ``<|im_end|>`` through the end of the generation prompt)."""
    ref = render_ids(tokenizer, messages_for("Hello.", request))
    toks = display_tokens(tokenizer, ref)
    end = max(i for i, t in enumerate(toks) if t == "<|im_end|>")
    return tokenizer.encode(" " + request, add_special_tokens=False) + ref[end:]


class WindowChecker:
    """Caches the canonical window per request sentence."""

    def __init__(self, tokenizer: Any):
        self.tok = tokenizer
        self._canon: dict[str, list[int]] = {}

    def canonical(self, request: str) -> list[int]:
        if request not in self._canon:
            self._canon[request] = canonical_window_ids(self.tok, request)
        return self._canon[request]

    def ok(self, messages: Sequence[Mapping[str, str]], request: str) -> bool:
        return window_ids(self.tok, messages) == self.canonical(request)


# ------------------------------------------------------------------ loaders


def load_wsbench_items(category: str, path: Path = WSBENCH_UM_ITEMS) -> list[dict[str, Any]]:
    """The wsbench user_modeling selfdescribe Country / Occupation items, verbatim."""
    cls = {"country": "Country", "occupation": "Occupation"}[category]
    bank = json.loads(Path(path).read_text(encoding="utf-8"))["items"]
    return [it for it in bank if it["subfamily"] == "selfdescribe" and it["attr_class"] == cls]


def load_items(path: Path = ITEMS_PATH) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["items"]


def load_probe_texts(path: Path = PROBE_TEXTS_PATH) -> list[dict[str, Any]]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x]


# ------------------------------------------------------------------ ReadSpec writers


def readspec_row(
    family: str, rid: str, messages: Sequence[Mapping[str, str]], layers: Sequence[int], note=""
) -> dict[str, Any]:
    if not _ID_OK.match(rid):
        raise ValueError(f"id {rid!r} is not filename-safe")
    row: dict[str, Any] = {
        "family": family,
        "id": rid,
        "render": "chat_context",
        "positions": dict(POSITIONS),
        "layers": list(layers),
        "messages": [dict(m) for m in messages],
    }
    if note:
        row["note"] = note
    return row


def write_plan_files(
    rows: Iterable[Mapping[str, Any]],
    out_dir: Path,
    prefix: str,
    *,
    layers: Sequence[int] = CAPTURE_LAYERS,
) -> dict[str, int]:
    """One ``<prefix>_<category>.jsonl`` ReadSpec plan file per category (the input of
    ``wsbench_public_plan.py prompts``). Ids must be unique across ALL files of one capture."""
    by_cat: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for r in rows:
        if r["id"] in seen:
            raise ValueError(f"duplicate id {r['id']}")
        seen.add(r["id"])
        fam = f"{prefix}_{r['category']}"
        by_cat.setdefault(fam, []).append(readspec_row(fam, r["id"], r["messages"], layers))
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(f"{prefix}_*.jsonl"):
        old.unlink()
    for fam, specs in by_cat.items():
        with (out_dir / f"{fam}.jsonl").open("w", encoding="utf-8") as fh:
            for s in specs:
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    return {fam: len(v) for fam, v in by_cat.items()}


def write_s3d_specs(
    items: Iterable[Mapping[str, Any]], path: Path, *, layers: Sequence[int] = S3D_LAYERS
) -> int:
    """The JSON list of ReadSpec dicts for ``wsbench_public_modal.py::fanout_custom`` (same
    render and positions rule as the capture plan files, so ids and positions match)."""
    specs = [
        readspec_row(f"{EVAL_FAMILY_PREFIX}_{it['category']}", it["id"], it["messages"], layers)
        for it in items
    ]
    ids = [s["id"] for s in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate ids in s3d specs")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(specs, ensure_ascii=False))
    return len(specs)
