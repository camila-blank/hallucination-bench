"""The wsbench_chat bank: items, read sites, the marked response and the expected cell grid.

``data/wsbench_chat/items.json`` is a verbatim copy of WorkspaceBench
``evals/hallucination/items.json`` @ 92d763e (``{meta, items: [{id, stratum, source, prompt,
response, response_start, sites: [{pos, token, kind, char}]}]}``).
"""

import hashlib
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
    "BANKS",
    "BANK_COMMIT",
    "BANK_DIR",
    "BANK_SOURCE",
    "ITEMS_PATH",
    "X3_BANK_DIR",
    "X3_ITEMS_PATH",
    "bank_record",
    "current_bank",
    "expected_grid",
    "load_bank",
    "load_items",
    "marked_response",
    "positions",
    "reset_bank",
    "sites_by_item",
    "use_bank",
]

BANK_DIR = DATA / "wsbench_chat"
# HBENCH_WSBENCH_CHAT_ITEMS: another subject model's on-policy bank (same schema), e.g. the
# DeepSeek-V4-Flash copy at <wsbench-pub-v4>/evals/hallucination/items.json
_OVERRIDE = os.environ.get("HBENCH_WSBENCH_CHAT_ITEMS")
ITEMS_PATH = Path(_OVERRIDE or BANK_DIR / "items.json")
if _OVERRIDE:  # provenance of the overriding bank, never the Qwen default's
    _SHA = hashlib.sha256(ITEMS_PATH.read_bytes()).hexdigest()[:16]
    BANK_SOURCE = f"override:{ITEMS_PATH}"
    BANK_COMMIT = f"sha256:{_SHA}"
else:
    BANK_SOURCE = f"camilablank/wsbench-pub@{WSBENCH_PUB_COMMIT}:evals/hallucination/items.json"
    BANK_COMMIT = WSBENCH_PUB_COMMIT

# Plan 0028: the main bank at 3x prompts / <=16 punctuation-only sites (local until frozen;
# HBENCH_X3_BANK = its directory, as tests/test_main_x3.py). Selected by `hbench run|report
# --bank x3` (plan 0029: the v8 main family); the HBENCH_WSBENCH_CHAT_ITEMS override still works.
X3_BANK_DIR = Path(os.environ.get("HBENCH_X3_BANK") or DATA / "wsbench_chat_x3")
X3_ITEMS_PATH = X3_BANK_DIR / "items.json"
BANKS: tuple[str, ...] = ("wsb", "x3")
# the bank named by the CLI's --bank (None = flag omitted); replaying a recorded bank
# (run.replay_bank) points the module without setting it
SELECTED: str | None = None
_DEFAULT_STATE = (ITEMS_PATH, BANK_SOURCE, BANK_COMMIT)


def use_bank(name: str, *, explicit: bool = True) -> None:
    """Point this module at a named bank for the rest of the process (CLI ``--bank``):
    ``x3`` = plan 0028's bank (:data:`X3_ITEMS_PATH`); ``wsb`` = the WSB 149 x 8 default. An
    ``HBENCH_WSBENCH_CHAT_ITEMS`` override naming another file conflicts (SystemExit).
    ``explicit`` (the CLI flag) records the name in :data:`SELECTED`; a replay of an arm dir's
    recorded bank passes ``explicit=False``."""
    global ITEMS_PATH, BANK_SOURCE, BANK_COMMIT, SELECTED
    if name not in BANKS:
        raise ValueError(f"unknown wsbench_chat bank {name!r} (known: {BANKS})")
    path = X3_ITEMS_PATH if name == "x3" else BANK_DIR / "items.json"
    if _OVERRIDE and Path(_OVERRIDE).resolve() != path.resolve():
        raise SystemExit(
            f"--bank {name} conflicts with HBENCH_WSBENCH_CHAT_ITEMS={_OVERRIDE} (unset one)"
        )
    if name == "wsb":
        if not _OVERRIDE:
            ITEMS_PATH = path
            BANK_SOURCE = (
                f"camilablank/wsbench-pub@{WSBENCH_PUB_COMMIT}:evals/hallucination/items.json"
            )
            BANK_COMMIT = WSBENCH_PUB_COMMIT
    else:
        if not path.is_file():
            raise SystemExit(
                f"--bank x3: {path} not found (plan 0028's bank is local until frozen: build it "
                "with scripts/main_x3/build_bank.py or set HBENCH_X3_BANK=<dir>)"
            )
        ITEMS_PATH = path
        BANK_SOURCE = f"plan0028-x3:{path}"
        BANK_COMMIT = f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}"
    if explicit:
        SELECTED = name
    load_bank.cache_clear()


def reset_bank() -> None:
    """Back to the import-time bank (the WSB default or the env override); ``SELECTED`` is
    left as it is."""
    global ITEMS_PATH, BANK_SOURCE, BANK_COMMIT
    ITEMS_PATH, BANK_SOURCE, BANK_COMMIT = _DEFAULT_STATE
    load_bank.cache_clear()


def current_bank() -> str:
    """The named bank this module points at: ``x3`` when :data:`ITEMS_PATH` is plan 0028's
    bank, else ``wsb`` (the default and any ``HBENCH_WSBENCH_CHAT_ITEMS`` override, which arm
    dirs never recorded)."""
    if BANK_SOURCE.startswith("plan0028-x3:"):
        return "x3"
    return "wsb"


def bank_record() -> dict[str, str]:
    """The ``arm.json`` fields recording the bank (plan 0029 review): only a non-default bank
    is recorded (``bank``, ``bank_source``, ``bank_sha256`` of the items file), so wsb / v7 arm
    dirs keep their arm.json and a missing ``bank`` means wsb."""
    if current_bank() == "wsb":
        return {}
    return {
        "bank": current_bank(),
        "bank_source": BANK_SOURCE,
        "bank_sha256": hashlib.sha256(ITEMS_PATH.read_bytes()).hexdigest(),
    }


@cache
def load_bank() -> dict[str, Any]:
    return json.loads(ITEMS_PATH.read_text(encoding="utf-8"))


def load_items() -> list[dict[str, Any]]:
    return load_bank()["items"]
