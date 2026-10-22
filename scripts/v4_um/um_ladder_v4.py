"""The plan 0006 / 0011 UM ladder driver (``scripts/sibling_probe/um_ladder.py``, unchanged) re-pointed
at DeepSeek-V4-Flash. Every stage is the landed driver's own ``cmd_*``; this wrapper only
overrides module globals:

* methods = the V4 arms (``v4-s3don-sft``, ``jlens`` = v4-jlens-std-n1000, ``v4-s3don-rl600`` when
  its readouts exist / ``--arms``); O-lens arms split their "- " bullets like s3d;
* layers: lens grid 15,18,...,42; probe layers + 41 (the J-lens's deepest block); distractor
  layer 30 (the depth of Qwen's L44); the cross-method window |dL| <= 4 is kept (one V4 grid step);
* paths: inputs from ``outputs/v4/um/src`` (V4 responses, V4 readouts), every output under
  ``outputs/v4/um[/<domain>]`` (concepts / probe texts / distractors in its ``data/``), the Opus
  generation cache in ``outputs/v4/um/gen``; nothing under ``data/`` is written;
* probe-text window check = the V4 template (``v4tok``), not Qwen's.

    env -u ANTHROPIC_API_KEY HF_HUB_OFFLINE=1 .venv/bin/python scripts/v4_um/um_ladder_v4.py \
        --category holiday mentions
"""

import os
import sys
from pathlib import Path

HB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HB / "scripts" / "sibling_probe"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import um_ladder as DRV  # noqa: E402  (the landed driver)

from hallucination_bench.evals.sibling_probe import bank as B  # noqa: E402
from hallucination_bench.evals.sibling_probe import um_ladder as U  # noqa: E402

V4UM = Path(os.environ.get("V4UM_ROOT", HB / "outputs" / "v4" / "um"))
SRC = V4UM / "src"
READ_LAYERS = (15, 18, 21, 24, 27, 30, 33, 36, 39, 42)
JTARGET = 41
ARM_FILES = {  # method id -> readout dir under src/outputs/sibling/readouts/
    "v4-s3don-sft": "v4-s3don-sft",
    "jlens": "v4-jlens-std-n1000",
    "v4-s3don-rl600": "v4-s3don-rl600",
}
LABELS = {"v4-s3don-sft": "V4 O-lens SFT", "jlens": "V4 J-lens", "v4-s3don-rl600": "V4 O-lens RL600"}


def configure_v4(cat: str, arms: list[str]) -> None:
    U.METHODS = tuple(arms)
    U.METHOD_LABEL = {m: LABELS[m] for m in arms}
    U.GRID_LAYERS = READ_LAYERS
    U.PROBE_LAYERS = tuple(sorted({*READ_LAYERS, JTARGET}))
    U.ADJ_LAYERS = ()
    U.DISTRACTOR_LAYER = 30
    U.NLA42 = "__no_nla__"  # eff_layer(method, L) = L for every V4 arm
    orig = U.cell_samples

    def cell_samples(method: str, row):  # O-lens arms pack 4 samples as "- " bullets (= s3d)
        return orig("s3d" if method.startswith("v4-s3don") else method, row)

    U.cell_samples = cell_samples

    out = V4UM if cat == "country" else V4UM / cat
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    DRV.OUT = out
    DRV.DATA = data
    DRV.READOUTS = {m: f"{ARM_FILES[m]}/sibling_{cat}.jsonl" for m in arms}
    DRV.CONCEPTS_PATH = data / f"um_{cat}_concepts.json"
    DRV.PROBE_TEXTS_PATH = data / f"um_{cat}_probe_texts.jsonl"
    DRV.DISTRACTORS_PATH = data / f"um_{cat}_distractors.json"
    DRV.PROBE_RUN = f"v4-um-{cat.replace('_', '-')}" + os.environ.get("V4_RUN_SUFFIX", "")
    DRV.PROBE_ACTS = f"um-v4:/pack/{cat}"
    DRV.EVAL_ACTS = {str(L): "um-v4:/eval_acts" for L in U.PROBE_LAYERS}
    DRV.PROBES_DIR = out / "probes"
    DRV.PROBES_ADJ_DIR = out / "probes_adj"  # never built for V4 (nearby tier not in um3)
    DRV.LABEL_DIR = out / "label"
    DRV.CAPTURE_PREFIX = f"umprobe_v4_{cat}"

    import gen_items as G  # type: ignore[import-not-found]

    G.OUT = V4UM / "gen"
    G.OUT.mkdir(parents=True, exist_ok=True)

    class V4WindowChecker:
        def __init__(self, _tok=None) -> None:
            import v4tok

            self.v4 = v4tok.get()

        def ok(self, messages, request: str) -> bool:
            return self.v4.window_ok(messages, request)

    B.WindowChecker = V4WindowChecker


def main() -> None:
    arms = os.environ.get("V4_ARMS", "v4-s3don-sft,jlens").split(",")
    argv = sys.argv[1:]
    if "--arms" in argv:
        i = argv.index("--arms")
        arms = argv[i + 1].split(",")
        del argv[i : i + 2]
    if "--src-root" not in argv:
        argv = ["--src-root", str(SRC), *argv]
    cat = argv[argv.index("--category") + 1] if "--category" in argv else "country"
    orig_setup = DRV.setup_domain

    def setup(c: str) -> None:
        orig_setup(c)
        configure_v4(c, arms)

    DRV.setup_domain = setup
    sys.argv = [sys.argv[0], *argv]
    print(f"[v4] category {cat}; arms {arms}; out {V4UM if cat == 'country' else V4UM / cat}",
          flush=True)
    DRV.main()


if __name__ == "__main__":
    main()
