"""The SAD stages-oversight bank (plan 0018 §Bank).

``scripts/sad/fetch_sad.py`` downloads ``sad/stages/oversight/structs.zip`` from LRudL/sad at
the pinned commit, unzips it with the published password and writes

* ``data/sad/raw/items.jsonl`` (gitignored; ``HBENCH_SAD_RAW`` overrides the directory): one row
  per item ``{id, sad_id, provenance, source, text}``;
* ``data/sad/manifest.json`` (committed): ``{meta, items: [{id, sad_id, provenance, source,
  sha256, n_chars}]}`` -- no text.

``provenance`` is SAD's label: ``eval`` (the ``test_oversight`` batch: Anthropic evals, BIG-bench,
WMDP) or ``deploy`` (``deploy_oversight``: sampled WildChat conversations).

The lens read is of the **plain prompt** (:func:`plain_render`): the SAD text alone as the user
turn, Qwen chat template, ``enable_thinking=False``, read up to ``<|im_start|>assistant``.
"""

import hashlib
import json
import os
from functools import cache
from pathlib import Path
from typing import Any

from hallucination_bench import DATA

BANK_DIR = DATA / "sad"
MANIFEST = BANK_DIR / "manifest.json"
# plan 0018 rev 5: extra real-user texts (WildChat-1M first user turns), fetch_wildchat.py
MANIFEST_WILDCHAT = BANK_DIR / "manifest_wildchat.json"
GATE_JSON = BANK_DIR / "gate.json"
SELECTED_JSON = BANK_DIR / "selected.json"
SAD_REPO = "LRudL/sad"
SAD_COMMIT = "dfc5c9831a9bcc5c9a9dbdcaa2955aae983cd1d3"
SAD_PATH = "sad/stages/oversight/structs.zip"
BATCHES = {"test_oversight": "eval", "deploy_oversight": "deploy"}
PROVENANCES = ("eval", "deploy")
USER_HEAD = "<|im_start|>user\n"
PROMPT_TAIL = "<|im_end|>\n<|im_start|>assistant"


def raw_dir() -> Path:
    return Path(os.environ.get("HBENCH_SAD_RAW") or BANK_DIR / "raw")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short_id(sad_id: str) -> str:
    """``MCQuestion-<sha256>`` -> ``sad-<first 12 hex>`` (unique over the 400 items; checked)."""
    return "sad-" + sad_id.removeprefix("MCQuestion-")[:12]


def plain_render(text: str) -> str:
    """The plain prompt the lens reads: the Qwen chat render with the SAD text as the only user
    turn and no system message, ending at ``<|im_start|>assistant`` (the AM convention). The
    generation-prompt suffix (``\\n<think>\\n\\n</think>\\n\\n``) is captured but never read.
    Qwen's template trims message content, so the text is stripped (47 of the 399 texts carry
    leading / trailing whitespace)."""
    return USER_HEAD + text.strip() + PROMPT_TAIL


@cache
def load_sad_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@cache
def load_manifest() -> dict[str, Any]:
    """Every gated item: the SAD manifest plus (rev 5) the WildChat-1M one when present."""
    items = list(load_sad_manifest()["items"])
    if MANIFEST_WILDCHAT.exists():
        items += json.loads(MANIFEST_WILDCHAT.read_text(encoding="utf-8"))["items"]
    return {"meta": load_sad_manifest()["meta"], "items": items}


def load_raw(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """``{id: raw row}`` from the gitignored raw file, checked against the manifest hashes."""
    paths = [path] if path else [raw_dir() / "items.jsonl"]
    if path is None and MANIFEST_WILDCHAT.exists():
        paths.append(raw_dir() / "wildchat.jsonl")
    rows = {}
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(
                f"{p} is missing: run scripts/sad/fetch_sad.py / fetch_wildchat.py (the texts "
                "are never committed)"
            )
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r
    p = paths[0]
    want = {
        it["id"]: it for it in (load_manifest() if path is None else load_sad_manifest())["items"]
    }
    if set(rows) != set(want):
        raise ValueError(f"{p}: ids differ from {MANIFEST} ({len(rows)} vs {len(want)})")
    bad = [i for i, r in rows.items() if sha256_text(r["text"]) != want[i]["sha256"]]
    if bad:
        raise ValueError(f"{p}: {len(bad)} texts do not match the manifest sha256 (e.g. {bad[0]})")
    return rows
