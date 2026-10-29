"""External locations (plan 0012 D3): each is an ``HBENCH_*`` environment variable whose default is
the dev-box path, so nothing changes there and another machine sets the variables it needs.

Paths inside this repo are not here: use ``hallucination_bench.REPO_ROOT`` / ``DATA``. Shell
scripts read the same variables as ``${HBENCH_X:-<default>}``."""

import os
from pathlib import Path


def _env_path(var: str, default: str) -> Path:
    return Path(os.environ.get(var) or default)


# WorkspaceBench paper readouts: <root>/<arm>/{hallucination,agentic_misalignment}.jsonl
READOUTS_ROOT = _env_path("HBENCH_READOUTS_ROOT", "/workspace/wsbench-paper-data/readouts")
# a workspace-bench (github.com/camilablank/workspace-bench) checkout: items, judge prompts
WSBENCH_PUB = _env_path("HBENCH_WSBENCH_PUB", "/workspace/root/wsbench-pub")
# the DeepSeek-V4-Flash copy of workspace-bench (V4 readouts under outputs/readouts/<arm>/)
WSBENCH_PUB_V4 = _env_path("HBENCH_WSBENCH_PUB_V4", "/workspace/root/wsbench-pub-v4-opus")
# a global-workspace checkout (hallucination_creative bank for the story eval)
GLOBAL_WORKSPACE = _env_path("HBENCH_GLOBAL_WORKSPACE", "/workspace/global-workspace")
# the checkout that ran plans 0003-0005 stage 1 (gitignored outputs/ the UM ladder reads)
SIBLING_SRC_ROOT = _env_path("HBENCH_SIBLING_SRC_ROOT", "/workspace/hb-sibling-probes")
# a global-workspace checkout with the Ali probe bank (plan 0005)
ALI_PROBE_ROOT = _env_path("HBENCH_ALI_PROBE_ROOT", "/workspace/gw-ali-probe")
# WorkspaceBench combo-arm readouts (plan 0005 bank build)
COMBOS_READOUTS = _env_path(
    "HBENCH_COMBOS_READOUTS", "/workspace/gw-wsb-combos/outputs/wsbench_combos/readouts"
)
# report outputs of the safety evals (plans 0015-0017), read by scripts/plot_safety_summary.py;
# the index (plan 0019) reads OR's index_counts.jsonl only when the variable is SET (opt-in;
# strongreject and mask are not on the index)
SAFETY_OR_OUT = _env_path(
    "HBENCH_SAFETY_OR_OUT", "/workspace/hb-overrefusal/outputs/overrefusal/claude-sonnet-5-5"
)
SAFETY_SR_OUT = _env_path(
    "HBENCH_SAFETY_SR_OUT", "/workspace/hb-strongreject/outputs/strongreject/claude-opus-5-5"
)
SAFETY_MASK_OUT = _env_path(
    "HBENCH_SAFETY_MASK_OUT", "/workspace/hb-mask/outputs/mask/claude-opus-5-5"
)

__all__ = [
    "ALI_PROBE_ROOT",
    "COMBOS_READOUTS",
    "GLOBAL_WORKSPACE",
    "READOUTS_ROOT",
    "SAFETY_MASK_OUT",
    "SAFETY_OR_OUT",
    "SAFETY_SR_OUT",
    "SIBLING_SRC_ROOT",
    "WSBENCH_PUB",
    "WSBENCH_PUB_V4",
]
