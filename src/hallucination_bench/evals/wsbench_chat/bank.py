"""The wsbench_chat bank: items, read sites, the marked response and the expected cell grid.

``data/wsbench_chat/items.json`` is a verbatim copy of WorkspaceBench
``evals/hallucination/items.json`` @ 92d763e (``{meta, items: [{id, stratum, source, prompt,
response, response_start, sites: [{pos, token, kind, char}]}]}``).
"""

import json
import os
from functools import cache
from pathlib import Path
from typing import Any

from hallucination_bench import DATA, WSBENCH_PUB_COMMIT
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

BANK_DIR = DATA / "wsbench_chat"
# HBENCH_WSBENCH_CHAT_ITEMS: another subject model's on-policy bank (same schema), e.g. the
# DeepSeek-V4-Flash copy at <wsbench-pub-v4>/evals/hallucination/items.json
ITEMS_PATH = Path(os.environ.get("HBENCH_WSBENCH_CHAT_ITEMS") or BANK_DIR / "items.json")
BANK_SOURCE = f"camilablank/wsbench-pub@{WSBENCH_PUB_COMMIT}:evals/hallucination/items.json"
BANK_COMMIT = WSBENCH_PUB_COMMIT


@cache
def load_bank() -> dict[str, Any]:
    return json.loads(ITEMS_PATH.read_text(encoding="utf-8"))


def load_items() -> list[dict[str, Any]]:
    return load_bank()["items"]
