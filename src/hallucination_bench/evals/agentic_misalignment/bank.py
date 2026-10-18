"""The agentic_misalignment bank: ``data/agentic_misalignment/items.json``, built by
``scripts/build_am_bank.py`` from WorkspaceBench ``evals/agentic_misalignment/items.json``
(provenance in its ``meta``).

Item schema: ``{id, category, arm, group, descriptor, misalignment_rate, persona, system, text,
prefill, prompt_render, response, sites: [{pos, token, char[, split_char]}]}``. ``char`` is the
end offset of token ``pos`` in ``prompt_render`` (the Qwen chat render the model read), so the
marker goes into the prompt: :func:`marked_prompt`.
"""

import json
from collections.abc import Collection
from functools import cache
from typing import Any

from hallucination_bench import DATA
from hallucination_bench.evals._bank_common import expected_grid, positions, sites_by_item
from hallucination_bench.judges.v5c import READ_MARK

__all__ = [
    "BANK_COMMIT",
    "BANK_DIR",
    "BANK_SOURCE",
    "ITEMS_PATH",
    "N_SITES",
    "conversation",
    "expected_grid",
    "load_bank",
    "load_items",
    "marked_prompt",
    "positions",
    "render_prompt",
    "sites_by_item",
]

BANK_DIR = DATA / "agentic_misalignment"
ITEMS_PATH = BANK_DIR / "items.json"
N_ITEMS = 32
N_SITES = 50_452


def render_prompt(system: str, text: str) -> str:
    """The Qwen chat render of one item's prompt, ending at ``<|im_start|>assistant``."""
    head = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
    return head + f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant"


@cache
def load_bank() -> dict[str, Any]:
    return json.loads(ITEMS_PATH.read_text(encoding="utf-8"))


def load_items(ids: Collection[str] | None = None) -> list[dict[str, Any]]:
    items = load_bank()["items"]
    if ids is None:
        return items
    known = {it["id"] for it in items}
    unknown = sorted(set(ids) - known)
    if unknown:
        raise KeyError(f"unknown agentic_misalignment item ids: {unknown}")
    return [it for it in items if it["id"] in set(ids)]


def marked_prompt(item: dict[str, Any], char: int) -> str:
    """``prompt_render`` with ``⟦READ⟧`` inserted at the site's ``char``."""
    p = item["prompt_render"]
    if not 0 <= char <= len(p):
        raise ValueError(f"{item['id']}: char {char} outside the prompt (len {len(p)})")
    return p[:char] + READ_MARK + p[char:]


def conversation(item: dict[str, Any]) -> str:
    """What evidence quotes are located in: the unmarked prompt render, then the response."""
    return item["prompt_render"] + "\n" + item["response"]


_META = load_bank()["meta"]
BANK_COMMIT: str = _META["source_commit"]
BANK_SOURCE = f"{_META['source_repo']}@{BANK_COMMIT[:7]}:{_META['source_path']}"
