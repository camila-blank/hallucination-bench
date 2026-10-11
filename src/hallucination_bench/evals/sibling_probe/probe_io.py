"""Inputs of the Modal probe job and loaders for its outputs (plan 0003 §Phase E).

``hbench sibling probes --prepare`` writes ``outputs/sibling/probes/inputs.json``: the concept
order per category (the column order of every probe output), the item / probe-text metadata and
the Qwen token ids of every concept's token aliases (unembedding check). The Modal job
(``scripts/sibling_probe/probes_modal.py``) writes its outputs to the same directory.
"""

import gzip
import json
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from hallucination_bench import REPO_ROOT
from hallucination_bench.evals.sibling_probe import bank as B
from hallucination_bench.evals.sibling_probe.concepts import load_inventory

PROBES_DIR = REPO_ROOT / "outputs" / "sibling" / "probes"
TOKENIZER = "Qwen/Qwen3.6-27B"


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def token_ids(tokenizer: Any) -> dict[str, dict[str, list[int]]]:
    out: dict[str, dict[str, list[int]]] = {}
    for cat in load_inventory().values():
        for c in cat.concepts:
            ids = []
            for t in c.token_aliases:
                enc = tokenizer.encode(t, add_special_tokens=False)
                if len(enc) != 1:
                    raise ValueError(f"{cat.name}/{c.id}: token alias {t!r} is {len(enc)} tokens")
                ids.append(int(enc[0]))
            out.setdefault(cat.name, {})[c.id] = ids
    return out


def prepare_inputs(path: Path = PROBES_DIR / "inputs.json", tokenizer: Any = None) -> dict:
    inv = load_inventory()
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    doc = {
        "concepts": {name: cat.ids for name, cat in inv.items()},
        "items": [
            {k: it[k] for k in ("id", "category", "gold", "request_idx")} for it in B.load_items()
        ],
        "probe_texts": [
            {k: t[k] for k in ("id", "category", "concept", "kind", "request_idx")}
            for t in B.load_probe_texts()
        ],
        "token_ids": token_ids(tokenizer),
        "git_sha": git_sha(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return doc


def iter_eval_scores(path: Path = PROBES_DIR / "eval_scores.jsonl.gz") -> Iterator[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def load_validity(path: Path = PROBES_DIR / "validity.jsonl") -> dict[tuple[str, str, int], dict]:
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[(r["category"], r["concept"], int(r["layer"]))] = r
    return out


OUTPUT_FILES: tuple[str, ...] = (
    "directions.safetensors",
    "thresholds.json",
    "validity.jsonl",
    "unembed.jsonl",
    "eval_scores.jsonl.gz",
    "summary.json",
    *(f"oof_{c}.npz" for c in load_inventory()),
)
