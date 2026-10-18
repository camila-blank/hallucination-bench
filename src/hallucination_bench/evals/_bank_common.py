"""Bank helpers shared by every eval bank (``wsbench_chat``, ``story``): read sites, the marked
response and the expected cell grid. Every bank's items use the same schema: ``{id, prompt,
response, sites: [{pos, token, kind, char}]}``."""

from collections.abc import Collection
from typing import Any

from hallucination_bench.judges.v5c import render_marked_response


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
