"""Eval registry: name -> module. Each eval module exposes ``run_arms``, ``rebuild_arm`` and a
``score`` submodule; later plans add more evals here.

``story`` (plan 0003) contributes only a bank: it runs ``wsbench_chat``'s pipeline with
``Models.eval = "story"``, which picks the bank through :func:`bank_for` and the output root
``<out>/<eval>/<model>/``."""

import importlib
from types import ModuleType

EVALS: dict[str, str] = {
    "wsbench_chat": "hallucination_bench.evals.wsbench_chat",
    "story": "hallucination_bench.evals.story",
}


def get(name: str) -> ModuleType:
    if name not in EVALS:
        raise KeyError(f"unknown eval {name!r}; known: {sorted(EVALS)}")
    return importlib.import_module(EVALS[name])


def bank_for(name: str) -> ModuleType:
    """The bank module of eval ``name`` (``load_items``, ``positions``, ``sites_by_item``,
    ``marked_response``, ``expected_grid``, ``BANK_SOURCE``, ``BANK_COMMIT``)."""
    if name not in EVALS:
        raise KeyError(f"unknown eval {name!r}; known: {sorted(EVALS)}")
    return importlib.import_module(f"{EVALS[name]}.bank")
