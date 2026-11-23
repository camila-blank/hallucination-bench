# ruff: noqa: E501  (the one-line descriptions and the table rows are long strings)
"""The README's tables of families and taxonomies, written from the code so they cannot drift.

    python scripts/readme_tables.py            # print the three blocks
    python scripts/readme_tables.py --write    # replace them in README.md, between the markers

Weights, groups, verdicts and the values of each axis come from the index and the two taxonomy
modules; the number of prompts from the data files. Only the one-line descriptions are written
here. ``tests/test_readme_tables.py`` fails when a README block is not the generated one.
"""

import importlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hallucination_bench import index  # noqa: E402


def _module(*names: str):
    """The first of ``names`` that exists (the taxonomy modules under either of their names)."""
    for name in names:
        try:
            return importlib.import_module(f"hallucination_bench.{name}")
        except ModuleNotFoundError:
            continue
    raise ModuleNotFoundError(names)


claim_map = _module("claim_map_taxonomy", "taxonomy_v2")
spans = _module("span_taxonomy", "taxonomy")

# family -> (what it tests, the data directory with its page, its items file)
FAMILIES = {
    "main": (
        "Claims about a chat conversation the model is replying to.",
        "wsbench_chat_x3",
        "items.json",
    ),
    "am_claims": (
        "Claims a lens makes where it mentions misalignment in an agentic scenario.",
        "agentic_misalignment",
        "items.json",
    ),
    "am_situation": (
        "Readouts that misrepresent the agentic scenario as a whole.",
        "agentic_misalignment",
        "items.json",
    ),
    "sibling": (
        "User modelling: a wrong value of the right kind (the user is Korean, the readout says China).",
        "sibling_probe",
        "items.json",
    ),
    "story": ("Claims about a story the model is writing.", "story", "items.json"),
    "overrefusal": (
        "Readouts that claim a refusal on prompts the model never refuses.",
        "overrefusal",
        "items.json",
    ),
    "benign_roleplay": (
        "Readouts that see a jailbreak in harmless role-play prompts.",
        "benign_roleplay",
        "items_v1.json",
    ),
}
LEAVES = {
    "junk": "Says nothing concrete, or is about neither the conversation nor a subject it raises.",
    "correct": "The conversation states it, the reply goes on to say it, or the conversation entails it.",
    "detail": "Contradicted, and the wrong part is a specific: a name, number, date or wording.",
    "false_claim": "A contradicted proposition or situation: reversed outcome or stance, wrong agent, wrong cause.",
    "substitution_sibling": "Contradicted: a different object of the same class in place of the right one.",
    "substitution_category": "Contradicted: an object of a different category in place of the right one.",
    "model_state": "About the model's own beliefs, state, intent or awareness.",
    "unverifiable": "On topic, neither contradicted nor settled by the conversation.",
    "supplementary_true": "A world claim the conversation does not settle, true in the world.",
    "supplementary_false": "A world claim the conversation does not settle, false in the world.",
    "ambiguous_topic": "Unclear whether it refers to this conversation at all.",
    "continuation_miss": "A contradicted guess of the reply's literal next words, excused by a second check.",
}
AXES = {
    "grounding": ("How the span fails against the conversation.", "GROUNDINGS"),
    "type": ("Whether the meaning is wrong or only a specific value.", "TYPES"),
    "target": ("What the span is about.", "TARGETS"),
    "severity": ("How much it would mislead a reader.", "SEVERITIES"),
    "unsupported kind": ("For an unsupported span, why nothing supports it.", "UNSUPPORTED_KINDS"),
}
BLOCKS = ("families", "claim map", "span axes")


def _count(directory: str, name: str) -> int:
    raw = json.loads((REPO / "data" / directory / name).read_text(encoding="utf-8"))
    return len(raw["items"] if isinstance(raw, dict) else raw)


def _weight(x: float) -> str:
    return f"{x:g}"


def families() -> str:
    weights = index.FAMILY_WEIGHTS_V8
    lines = [
        "| family | weight in the index | what it tests | prompts | page |",
        "|---|---|---|---|---|",
    ]
    for name in index.FAMILIES_V8:
        what, directory, items = FAMILIES[name]
        lines.append(
            f"| `{name}` | {_weight(weights[name])} | {what} | {_count(directory, items):,} "
            f"| [data/{directory}](data/{directory}/README.md) |"
        )
    total = sum(weights.values())
    lines += [
        "",
        f"All {len(weights)} families are in the index; the weights sum to {_weight(total)}.",
    ]
    return "\n".join(lines)


def claim_map_table() -> str:
    main, story = index.MAIN_LEAF_WEIGHTS, index.STORY_LEAF_WEIGHTS
    names = [*claim_map.LEAF_NAMES, *claim_map.SCORING_EXTRA]
    lines = [
        "| leaf | meaning | group | verdict | weight (`main` / `story`) |",
        "|---|---|---|---|---|",
    ]
    for name in names:
        group = (
            claim_map.group_of(name) if hasattr(claim_map, "group_of") else claim_map.GROUP_OF[name]
        )
        lines.append(
            f"| `{name}` | {LEAVES[name]} | {group} | {claim_map.verdict_of(name)} "
            f"| {_weight(main.get(name, 0))} / {_weight(story.get(name, 0))} |"
        )
    return "\n".join(lines)


def span_axes() -> str:
    lines = ["| axis | what it says | values |", "|---|---|---|"]
    for axis, (what, table) in AXES.items():
        values = getattr(spans, table)
        shown = [v[0] if isinstance(v, tuple) else str(k) for k, v in values.items()]
        lines.append(f"| {axis} | {what} | {', '.join(f'`{v}`' for v in shown)} |")
    return "\n".join(lines)


TABLES = {"families": families, "claim map": claim_map_table, "span axes": span_axes}


def markers(block: str) -> tuple[str, str]:
    return f"<!-- {block}: start -->", f"<!-- {block}: end -->"


def replace(text: str) -> str:
    for block in BLOCKS:
        start, end = markers(block)
        head, rest = text.split(start, 1)
        _, tail = rest.split(end, 1)
        text = f"{head}{start}\n{TABLES[block]()}\n{end}{tail}"
    return text


if __name__ == "__main__":
    if "--write" in sys.argv[1:]:
        readme = REPO / "README.md"
        readme.write_text(replace(readme.read_text(encoding="utf-8")), encoding="utf-8")
    else:
        print("\n\n".join(TABLES[b]() for b in BLOCKS))
