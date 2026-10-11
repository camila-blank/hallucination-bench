"""Eval registry: name -> module. Each eval module exposes ``run_arms``, ``rebuild_arm`` and a
``score`` submodule; later plans add more evals here."""

import importlib
from types import ModuleType

EVALS: dict[str, str] = {"wsbench_chat": "hallucination_bench.evals.wsbench_chat"}


def get(name: str) -> ModuleType:
    if name not in EVALS:
        raise KeyError(f"unknown eval {name!r}; known: {sorted(EVALS)}")
    return importlib.import_module(EVALS[name])
