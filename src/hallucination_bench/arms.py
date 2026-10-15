"""Plan 0010 §1: the arm registry (``data/arms.json``).

* ``gated_arms``: the persisted default gate set (``hbench gate --add-arm X`` appends). The cross
  index, class counts, the >= 4-pair cap, the matched null and ``item_sibling_names`` are always
  computed over this set.
* ``arms.<name>``: ``cells`` (overrides ``<cells-root>/<arm>/cells.jsonl``), ``readouts``,
  ``kind`` (``sampled`` | ``token``), ``family``, ``layers``, ``pool`` (sampled arms).
* ``pools.<pool>``: ``generator`` (``read`` | ``nla`` | ``lensagree``) and its ``args``,
  ``sampling``, ``orig_arm`` / ``orig_readouts`` (shape check), ``gpu``, ``samples_per_run``.
* ``env``, ``snapshots`` (the generator / capture script snapshots), ``acts_manifests``.

This module only reads JSON (it is imported by ``sibling_gate.units``, which the rebuild path
imports), so it stays light. ``python -m hallucination_bench.arms bash`` prints the bash views
``scripts/sibling_gate/run_all.sh`` uses (``POOLS``, ``ARM_OF``).
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

from hallucination_bench import DATA

REGISTRY_PATH = DATA / "arms.json"
KINDS = ("sampled", "token")
GENERATORS = ("read", "nla", "lensagree")

_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


class RegistryError(ValueError):
    """A registry / config problem (``hbench gate`` exits 3)."""


def load(path: Path | None = None) -> dict[str, Any]:
    """The registry dict (a fresh deep copy; cached by path + mtime)."""
    p = Path(path) if path is not None else REGISTRY_PATH
    mt = p.stat().st_mtime
    hit = _CACHE.get(str(p))
    if hit is None or hit[0] != mt:
        reg = json.loads(p.read_text())
        validate(reg)
        _CACHE[str(p)] = (mt, reg)
        hit = _CACHE[str(p)]
    return copy.deepcopy(hit[1])


def save(reg: dict[str, Any], path: Path | None = None) -> None:
    validate(reg)
    p = Path(path) if path is not None else REGISTRY_PATH
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(reg, indent=1, ensure_ascii=False) + "\n")
    tmp.replace(p)
    _CACHE.pop(str(p), None)


def validate(reg: dict[str, Any]) -> None:
    for k in ("gated_arms", "arms", "pools", "env", "snapshots"):
        if k not in reg:
            raise RegistryError(f"arms.json lacks {k!r}")
    for a in reg["gated_arms"]:
        if a not in reg["arms"]:
            raise RegistryError(f"gated arm {a!r} is not registered under arms")
    for a, row in reg["arms"].items():
        if row.get("kind") not in KINDS:
            raise RegistryError(f"{a}: kind must be one of {KINDS}")
        p = row.get("pool")
        if p is not None and p not in reg["pools"]:
            raise RegistryError(f"{a}: pool {p!r} is not registered")
    for p, row in reg["pools"].items():
        if row.get("generator") not in GENERATORS:
            raise RegistryError(f"pool {p}: generator must be one of {GENERATORS}")


# ---------------------------------------------------------------- views


def _reg(reg: dict[str, Any] | None) -> dict[str, Any]:
    return reg if reg is not None else load()


def gated_arms(reg: dict[str, Any] | None = None) -> tuple[str, ...]:
    return tuple(_reg(reg)["gated_arms"])


def arm(name: str, reg: dict[str, Any] | None = None) -> dict[str, Any] | None:
    return _reg(reg)["arms"].get(name)


def family_of(name: str, reg: dict[str, Any] | None = None) -> str:
    """The registered family; an unregistered arm: plan 0009 §3e prefixes (s3d-*, mlnla-*,
    nla-*), else the arm itself."""
    row = arm(name, reg)
    if row is not None and row.get("family"):
        return str(row["family"])
    for p in ("s3d", "mlnla", "nla"):
        if name.startswith(p + "-"):
            return p
    return name


def kind_of(name: str, reg: dict[str, Any] | None = None) -> str | None:
    row = arm(name, reg)
    return None if row is None else row["kind"]


def pool_of(name: str, reg: dict[str, Any] | None = None) -> str | None:
    """The arm's pool over every REGISTERED arm (not only the gated set)."""
    row = arm(name, reg)
    return None if row is None else row.get("pool")


def prose_arms(reg: dict[str, Any] | None = None) -> tuple[str, ...]:
    r = _reg(reg)
    return tuple(a for a in r["gated_arms"] if r["arms"][a]["kind"] == "sampled")


def token_arms(reg: dict[str, Any] | None = None) -> tuple[str, ...]:
    r = _reg(reg)
    return tuple(a for a in r["gated_arms"] if r["arms"][a]["kind"] == "token")


def deterministic_arms(reg: dict[str, Any] | None = None) -> frozenset[str]:
    """Every registered token arm (token bags: no sampling, rung 2 directly)."""
    r = _reg(reg)
    return frozenset(a for a, row in r["arms"].items() if row["kind"] == "token")


def pools_of_gated(reg: dict[str, Any] | None = None) -> dict[str, tuple[str, ...]]:
    """Pool -> its gated arms (registry pool order; pools without a gated arm are omitted)."""
    r = _reg(reg)
    out: dict[str, tuple[str, ...]] = {}
    for p in r["pools"]:
        arms = tuple(a for a in r["gated_arms"] if r["arms"][a].get("pool") == p)
        if arms:
            out[p] = arms
    return out


def pool_arms(pool: str, reg: dict[str, Any] | None = None) -> tuple[str, ...]:
    """Every registered arm of ``pool`` (gated first, in gated order)."""
    r = _reg(reg)
    g = [a for a in r["gated_arms"] if r["arms"][a].get("pool") == pool]
    rest = [a for a, row in r["arms"].items() if row.get("pool") == pool and a not in g]
    return (*g, *rest)


def arm_of_pool(reg: dict[str, Any] | None = None) -> dict[str, str]:
    """``run_all.sh`` ``ARM_OF``: pool -> the arm whose readouts the shape check compares."""
    r = _reg(reg)
    return {p: r["pools"][p]["orig_arm"] for p in pools_of_gated(r)}


def cells_path(name: str, cells_root: Path, reg: dict[str, Any] | None = None) -> Path:
    """The registered ``cells`` (always wins), else ``<cells_root>/<arm>/cells.jsonl``."""
    row = arm(name, reg) or {}
    c = row.get("cells")
    return Path(c) if c else Path(cells_root) / name / "cells.jsonl"


def with_added(reg: dict[str, Any], names: list[str]) -> dict[str, Any]:
    """A copy of ``reg`` with ``names`` appended to ``gated_arms`` (registered arms only)."""
    out = copy.deepcopy(reg)
    for n in names:
        if n not in out["arms"]:
            raise RegistryError(f"--add-arm {n}: not registered under arms in data/arms.json")
        if n not in out["gated_arms"]:
            out["gated_arms"].append(n)
    return out


def bash_views(reg: dict[str, Any] | None = None) -> str:
    r = _reg(reg)
    ao = arm_of_pool(r)
    return "\n".join(
        [
            f'POOLS="{" ".join(pools_of_gated(r))}"',
            "declare -A ARM_OF=(" + " ".join(f"[{p}]={a}" for p, a in ao.items()) + ")",
        ]
    )


if __name__ == "__main__":  # pragma: no cover - shell helper
    if sys.argv[1:] == ["bash"]:
        print(bash_views())
    else:
        raise SystemExit("usage: python -m hallucination_bench.arms bash")
