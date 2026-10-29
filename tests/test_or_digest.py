"""Plan 0020: the wsbench digest / bpe ports equal the pinned source (wsbench @ 1baaf00)."""

import ast
import os
import subprocess
from pathlib import Path

import pytest

from hallucination_bench.evals.overrefusal import wsb_bpe, wsb_digest

PORTED_DEFS = (  # every top-level def / class copied from wsbench digest.py
    "Line",
    "_Group",
    "_one_line",
    "_group",
    "_render",
    "est_tokens",
    "bundle",
    "_round_robin",
    "_stride",
    "_norm",
    "_lower_ws",
    "_folded",
    "_span_in",
    "_prose_level",
    "_token_level",
    "match_level",
    "evidence_line",
    "_plain",
    "_key_hits",
    "support_of",
    "Grounded",
    "_cjk_int",
    "number_value",
    "_number_key_ok",
    "has_entries",
    "ground",
    "render_summary",
    "render_user",
)
CONSTS = (
    "DIGEST_PROMPT_VERSION",
    "DEFAULT_MODEL",
    "DEFAULT_EFFORT",
    "DEFAULT_MAX_ITEMS",
    "DEFAULT_MAX_RARE",
    "DEFAULT_TOKEN_CAP",
    "MAX_TOKENS",
    "DIGEST_TIMEOUT",
    "DEFAULT_VARIANT",
    "DEFAULT_RENDER",
    "DIGEST_SYSTEM",
    "DIGEST_USER",
    "SUMMARY_NOTE",
    "SUMMARY_NOTE_TOKENS",
    "EXTRACTIVE_LINE",
    "TOKEN_SEP",
    "DIGEST_SCHEMA",
)


def _pinned(rel: str) -> str:
    root = os.environ.get("HBENCH_WSBENCH_PUB")
    if not root or not Path(root).is_dir():
        pytest.skip("set HBENCH_WSBENCH_PUB to a workspace-bench checkout")
    r = subprocess.run(
        ["git", "-C", root, "show", f"{wsb_digest.WSBENCH_SHA}:{rel}"],
        capture_output=True,
        text=True,
    )
    if r.returncode:
        pytest.skip(f"{wsb_digest.WSBENCH_SHA} not in {root}")
    return r.stdout


def _defs(src: str) -> dict[str, str]:
    tree = ast.parse(src)
    return {n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef | ast.ClassDef)}


def test_digest_functions_and_literals_equal_the_pinned_source():
    src = _pinned("src/wsbench/digest.py")
    pinned, ported = _defs(src), _defs(Path(wsb_digest.__file__).read_text())
    for name in PORTED_DEFS:
        assert ported[name] == pinned[name], name
    ns: dict = {"schema_block": wsb_digest.schema_block}
    lines = src.split("\n")
    start = next(i for i, ln in enumerate(lines) if ln.startswith("DIGEST_PROMPT_VERSION"))
    end = next(i for i, ln in enumerate(lines) if ln.startswith("# Families that run native"))
    exec("\n".join(lines[start:end]), ns)  # the pinned constants block, as written
    tok = next(ln for ln in lines if ln.startswith("TOKEN_SEP"))
    exec(tok, ns)
    for name in CONSTS:
        assert getattr(wsb_digest, name) == ns[name], name
    # render_system is reimplemented on the caps: the same string as wsbench for the defaults
    want = (
        ns["DIGEST_SYSTEM"]
        .replace("{max_items}", str(ns["DEFAULT_MAX_ITEMS"]))
        .replace("{max_rare}", str(ns["DEFAULT_MAX_RARE"]))
    )
    assert wsb_digest.render_system() == want


def test_bpe_equals_the_pinned_source():
    src = _pinned("src/wsbench/bpe.py")
    pinned, ported = _defs(src), _defs(Path(wsb_bpe.__file__).read_text())
    assert set(pinned) == set(ported)
    for name in pinned:
        assert ported[name] == pinned[name], name


def test_bpe_decodes():
    assert wsb_bpe.decode_token("Ġrefuse") == " refuse"
    assert wsb_bpe.decode_token("æĪĳæĹłæ³ķ") == "我无法"
    assert "�" not in wsb_bpe.decode_token("æĪ")
