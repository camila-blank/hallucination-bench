"""The wsbench_chat bank: items, read sites, the marked response and the expected cell grid.

``data/wsbench_chat/items.json`` is a verbatim copy of WorkspaceBench
``evals/hallucination/items.json`` @ 92d763e (``{meta, items: [{id, stratum, source, prompt,
response, response_start, sites: [{pos, token, kind, char}]}]}``).
"""

import json
from collections.abc import Collection
from functools import cache
from typing import Any

from hallucination_bench import DATA, WSBENCH_PUB_COMMIT
from hallucination_bench.judges.v5c import render_marked_response

BANK_DIR = DATA / "wsbench_chat"
ITEMS_PATH = BANK_DIR / "items.json"
BANK_SOURCE = f"camilablank/wsbench-pub@{WSBENCH_PUB_COMMIT}:evals/hallucination/items.json"


@cache
def load_bank() -> dict[str, Any]:
    return json.loads(ITEMS_PATH.read_text(encoding="utf-8"))


def load_items() -> list[dict[str, Any]]:
    return load_bank()["items"]


def sites_by_item(items: list[dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    return {it["id"]: {int(s["pos"]): s for s in it["sites"]} for it in items}


def positions(items: list[dict[str, Any]]) -> dict[str, list[int]]:
    return {it["id"]: [int(s["pos"]) for s in it["sites"]] for it in items}


def marked_response(item: dict[str, Any], site: dict[str, Any]) -> str:
    return render_marked_response(item["response"], int(site["char"]))


def expected_grid(
    items: list[dict[str, Any]], layers: Collection[int]
) -> set[tuple[str, int, int]]:
    return {
        (it["id"], layer, int(s["pos"])) for it in items for layer in layers for s in it["sites"]
    }
