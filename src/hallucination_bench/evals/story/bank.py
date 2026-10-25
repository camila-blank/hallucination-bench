"""The story bank: ``data/story/items.json``, built by ``scripts/build_story_bank.py`` from
global-workspace's ``hallucination_creative`` bank (provenance in its ``meta``). Same item schema
and public API as ``wsbench_chat.bank``."""

import json
import os
from functools import cache
from pathlib import Path
from typing import Any

from hallucination_bench import DATA
from hallucination_bench.evals._bank_common import (
    expected_grid,
    marked_response,
    positions,
    sites_by_item,
)

__all__ = [
    "BANK_COMMIT",
    "BANK_DIR",
    "BANK_SOURCE",
    "ITEMS_PATH",
    "expected_grid",
    "load_bank",
    "load_items",
    "marked_response",
    "positions",
    "sites_by_item",
]

BANK_DIR = DATA / "story"
# HBENCH_STORY_ITEMS: another subject model's on-policy story bank (same schema + meta keys),
# e.g. the DeepSeek-V4-Flash bank (built by gw-v4-flash-ar's scripts/v4_flash/story_v4_bank.py)
ITEMS_PATH = Path(os.environ.get("HBENCH_STORY_ITEMS") or BANK_DIR / "items.json")


@cache
def load_bank() -> dict[str, Any]:
    return json.loads(ITEMS_PATH.read_text(encoding="utf-8"))


def load_items() -> list[dict[str, Any]]:
    return load_bank()["items"]


_META = load_bank()["meta"]
BANK_COMMIT: str = _META["source_commit"]
BANK_SOURCE = f"{_META['source_repo']}@{BANK_COMMIT[:8]}:{_META['source_path']}/items.json"
