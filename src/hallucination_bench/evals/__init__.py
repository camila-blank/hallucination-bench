"""Eval registry: name -> module. Each eval module exposes ``run_arms``, ``rebuild_arm`` and a
``score`` submodule; later plans add more evals here.

``story`` (plan 0007) contributes only a bank: it runs ``wsbench_chat``'s pipeline with
``Models.eval = "story"``, which picks the bank through :func:`bank_for` and the output root
``<out>/<eval>/<model>/``.

``agentic_misalignment`` (plan 0008) has its **own driver**, not the shared chat pipeline: it
exposes ``run.run_am`` / ``run.rebuild`` and ``score`` (no ``run_arms`` / ``rebuild_arm``), and
its bank has prompt-position sites with the marker in the prompt (``bank.marked_prompt``), not
``bank_for``'s chat-style ``marked_response`` sites. ``hbench run|report --eval
agentic_misalignment`` dispatch to it.

``overrefusal`` (plan 0015) also has its own driver and its own subcommands (``hbench
overrefusal ...``, registered through ``cli.EVAL_CLIS``); it has no ``bank`` submodule, so
:func:`bank_for` does not apply to it."""

import importlib
from types import ModuleType

EVALS: dict[str, str] = {
    "wsbench_chat": "hallucination_bench.evals.wsbench_chat",
    "story": "hallucination_bench.evals.story",
    "agentic_misalignment": "hallucination_bench.evals.agentic_misalignment",
    "overrefusal": "hallucination_bench.evals.overrefusal",
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
