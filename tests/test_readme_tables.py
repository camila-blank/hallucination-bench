"""The README's tables of families, claim-map leaves and span axes are the ones the code gives:
written by ``scripts/readme_tables.py --write``, never by hand."""

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _tables():
    path = REPO / "scripts" / "readme_tables.py"
    spec = importlib.util.spec_from_file_location("readme_tables", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_readme_blocks_are_the_generated_ones():
    gen = _tables()
    text = (REPO / "README.md").read_text(encoding="utf-8")
    assert text == gen.replace(text), "run: python scripts/readme_tables.py --write"


def test_every_family_leaf_and_axis_of_the_code_is_in_the_tables():
    gen = _tables()
    assert set(gen.FAMILIES) == set(gen.index.FAMILIES_V8)
    assert set(gen.LEAVES) == {*gen.claim_map.LEAF_NAMES, *gen.claim_map.SCORING_EXTRA}
    table = gen.claim_map_table()
    for leaf in gen.claim_map.LEAF_NAMES:
        row = next(ln for ln in table.splitlines() if ln.startswith(f"| `{leaf}` |"))
        assert f"| {gen.claim_map.verdict_of(leaf)} |" in row, leaf
    verdicts = {gen.claim_map.verdict_of(leaf) for leaf in gen.LEAVES}
    assert verdicts == set(gen.claim_map.VERDICTS)
    axes = gen.span_axes()
    for table_name in ("GROUNDINGS", "TYPES", "TARGETS", "UNSUPPORTED_KINDS"):
        for value in getattr(gen.spans, table_name):
            assert f"`{value}`" in axes, value
