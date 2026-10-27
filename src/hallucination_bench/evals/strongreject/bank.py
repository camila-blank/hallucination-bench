"""The strongreject read bank: ``data/strongreject/items.json``, built by
``scripts/strongreject/build_bank.py`` from ``prompts.json`` + the graded outcomes (plan 0016
§Read set and sites).

Item schema: ``{id, forbidden_id, category, source, family, forbidden_prompt, jailbreak_prompt,
outcome, outcome_counts, prompt_render, n_positions, sites: [{pos, token, char, region}]}``.
``prompt_render`` is the Qwen chat render up to and including ``<|im_start|>assistant``;
``char`` is the end offset of token ``pos`` in it; ``region`` is ``turn`` (last 4 positions),
``tail`` (the rest of the last 64) or ``body`` (the stride-16 sample before).
"""

import json
import os
from collections.abc import Collection
from functools import cache
from pathlib import Path
from typing import Any

from hallucination_bench import DATA

BANK_DIR = DATA / "strongreject"
ITEMS_PATH = Path(os.environ.get("HBENCH_SR_ITEMS") or BANK_DIR / "items.json")
PROMPTS_PATH = BANK_DIR / "prompts.json"

TAIL = 64  # every position in the last TAIL read positions
STRIDE = 16  # plus every position p < n - TAIL with p % STRIDE == 0
TURN = 4  # <|im_end|> \n <|im_start|> assistant
REGIONS = ("body", "tail", "turn")


def site_positions(n: int) -> list[tuple[int, str]]:
    """The thinned read sites of an n-position prompt render: ``[(pos, region)]`` sorted."""
    if n < 1:
        raise ValueError("empty render")
    out = [(p, "body") for p in range(0, max(0, n - TAIL)) if p % STRIDE == 0]
    for p in range(max(0, n - TAIL), n):
        out.append((p, "turn" if p >= n - TURN else "tail"))
    return out


def render_prompt(text: str) -> str:
    """The Qwen chat render of a single user turn, ending at ``<|im_start|>assistant``. The Qwen
    template trims the message content (``poems`` / ``wikipedia_with_title`` end in whitespace),
    so this does too: it is the text the model read."""
    return f"<|im_start|>user\n{text.strip()}<|im_end|>\n<|im_start|>assistant"


@cache
def load_prompts() -> dict[str, Any]:
    return json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))


@cache
def load_bank() -> dict[str, Any]:
    return json.loads(ITEMS_PATH.read_text(encoding="utf-8"))


def load_items(ids: Collection[str] | None = None) -> list[dict[str, Any]]:
    items = load_bank()["items"]
    if ids is None:
        return items
    unknown = sorted(set(ids) - {it["id"] for it in items})
    if unknown:
        raise KeyError(f"unknown strongreject item ids: {unknown}")
    return [it for it in items if it["id"] in set(ids)]


def sites_by_item(items: list[dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    return {it["id"]: {s["pos"]: s for s in it["sites"]} for it in items}


def turn_pos(item: dict[str, Any]) -> int:
    """The final read position (the ``assistant`` token)."""
    return item["n_positions"] - 1
