"""``data/mask/`` loaders, site selection and the readouts loader / grid check (plan 0017).

Files (see ``data/mask/README.md``): ``items.json`` (the 450 sampled MASK items),
``labels.json`` (stage T), ``rollouts.jsonl.gz`` (every sample; pinned token ids),
``read_set.json`` + ``sites.json`` + ``capture_rows.json`` (``scripts/mask/build_capture.py``).

External paths are environment variables with dev-box defaults (the ``paths.py`` convention,
kept local so this eval touches no shared module): ``HBENCH_MASK_READOUTS`` (readouts root,
``<root>/<arm>/mask.jsonl``).
"""

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hallucination_bench import DATA
from hallucination_bench.readouts import Cell, load_readouts

DATA_DIR = DATA / "mask"
READOUTS_ROOT = Path(os.environ.get("HBENCH_MASK_READOUTS") or "/workspace/hb-readouts-mask")
READOUTS_FILE = "mask.jsonl"
N_PROMPT_SITES = 8
N_RESPONSE_SITES = 8
SITE_TYPES = ("prompt", "response")
IM_START, IM_END = "<|im_start|>", "<|im_end|>"


def _load(name: str) -> Any:
    return json.loads((DATA_DIR / name).read_text())


def load_items() -> list[dict[str, Any]]:
    return _load("items.json")["items"]


def load_labels() -> dict[str, dict[str, Any]]:
    return {r["task_id"]: r for r in _load("labels.json")["items"]}


def load_sites() -> dict[str, list[dict[str, Any]]]:
    """``{task_id: [{pos, token, site_type, char}, ...]}`` of the read set."""
    return {r["task_id"]: r["sites"] for r in _load("sites.json")["items"]}


def _linspace_int(lo: int, hi: int, n: int) -> list[int]:
    if hi <= lo:
        return [lo]
    return sorted({round(lo + (hi - lo) * i / (n - 1)) for i in range(n)})


def select_sites(tokens: Sequence[str], n_prompt_ids: int) -> list[tuple[int, str]]:
    """``[(pos, site_type)]`` over a decoded token stream ``prompt + pinned response``.

    ``tokens`` are the decoded strings of the capture ``input_ids``; ``n_prompt_ids`` is the
    length of the rendered prompt (generation suffix included). Prompt sites: 8 evenly spaced
    from the first system-content token to the ``assistant`` token of the last ``<|im_start|>``
    ``assistant`` pair (always included); the ``\\n<think>\\n\\n</think>\\n\\n`` suffix is never
    read. Response sites: 8 evenly spaced over the pinned tokens, a trailing ``<|im_end|>``
    excluded."""
    if len(tokens) < 3 or tokens[0] != IM_START or tokens[1] != "system":
        raise ValueError("render does not start with <|im_start|>system")
    s = next(i for i in range(2, n_prompt_ids) if "\n" in tokens[i]) + 1
    a = max(
        i for i in range(1, n_prompt_ids) if tokens[i] == "assistant" and tokens[i - 1] == IM_START
    )
    if not s <= a:
        raise ValueError(f"empty prompt span ({s}, {a})")
    out = [(p, "prompt") for p in _linspace_int(s, a, N_PROMPT_SITES)]
    r0, r1 = n_prompt_ids, len(tokens) - 1
    while r1 >= r0 and tokens[r1] == IM_END:
        r1 -= 1
    if r1 >= r0:
        out += [(p, "response") for p in _linspace_int(r0, r1, N_RESPONSE_SITES)]
    return out


def resolve_arms(root: str | None, arms: str) -> list[tuple[str, Path]]:
    base = Path(root) if root else READOUTS_ROOT
    out = []
    for name in [a for a in arms.split(",") if a]:
        p = base / name / READOUTS_FILE
        if not p.exists():
            raise SystemExit(f"[{name}] readouts file not found: {p}")
        out.append((name, p))
    if not out:
        raise SystemExit("no arms")
    return out


def load_arm_cells(
    path: Path, sites: dict[str, list[dict[str, Any]]], layers: Sequence[int] | None = None
) -> tuple[list[Cell], list[int]]:
    """Every cell of the read-set grid at the file's layers (or ``layers``); a missing,
    off-grid or duplicate row is fatal (the whole file is checked before ``layers`` applies)."""
    positions = {tid: [s["pos"] for s in ss] for tid, ss in sites.items()}
    cells, rep = load_readouts(path)
    bad = {k: v for k, v in rep.skipped.items() if v}
    if bad:
        raise SystemExit(f"{path}: skipped rows {bad}")
    have = {(c.id, c.layer, c.pos) for c in cells}
    file_layers = sorted({c.layer for c in cells})
    want = {(t, la, p) for t, ps in positions.items() for p in ps for la in file_layers}
    miss, off = want - have, have - want
    if miss or off:
        raise SystemExit(
            f"{path}: grid mismatch: {len(miss)} missing (e.g. {sorted(miss)[:3]}), "
            f"{len(off)} off-grid (e.g. {sorted(off)[:3]})"
        )
    if layers:
        unknown = set(layers) - set(file_layers)
        if unknown:
            raise SystemExit(f"{path}: layers {sorted(unknown)} not in file {file_layers}")
        cells = [c for c in cells if c.layer in set(layers)]
    return cells, sorted({c.layer for c in cells})
