"""The plan 0006 / 0011 UM ladder driver (``scripts/sibling_probe/um_ladder.py``, unchanged) with
ONE extra lens arm appended to the method pool, for adding a new arm to the hallucination index.

Every stage is the landed driver's own ``cmd_*``; this wrapper only overrides module globals:

* methods = the frozen four (s3d, mlnla-rl600, jlens, nla-rl-L42) + ``$UM_ARM``; the new arm's
  samples are split on "- " bullets like s3d's (an O-lens readout is a bullet list);
* readouts: ``$UM_ARM`` reads ``<src-root>/outputs/sibling/readouts/$UM_ARM/sibling_eval.jsonl``
  (all three domains in one file, as s3d); the frozen four read the official files;
* outputs: every ``OUT``-derived path points at ``$UM_ROOT`` (a COPY of the official
  ``outputs/sibling/um`` root, never the original): the frozen concept set C, probe texts,
  probes, distractors and probe scores are reused as they are (never re-run ``concepts``,
  ``probe-texts`` or ``distractors``), so a new-arm commit to a concept outside C is untestable.

Run per domain: ``mentions`` -> ``judge`` -> ``label`` (the frozen arms' judge calls are cache
hits in the copied root). Adding an arm relabels the frozen arms' units in the copy (cross-method
support); only the new arm's units are exported (``export_um_arm.py``).

    UM_ARM=final-olens-rl600-x4 UM_ROOT=<copy> python scripts/final_olens/um_ladder_arm.py \\
        --src-root <src> --category holiday mentions
"""

import os
import sys
from pathlib import Path

HB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HB / "scripts" / "sibling_probe"))

import um_ladder as DRV  # noqa: E402  (the landed driver)

from hallucination_bench.evals.sibling_probe import um_ladder as U  # noqa: E402

# the only stages this wrapper runs: every other driver stage writes the repo's frozen data/
# (todo, concepts, probe-texts, distractors) or plots/ (figures), or reads probe inputs that must
# stay the official run's (capture-rows, probe-inputs, calib)
ALLOWED = ("summarize", "mentions", "judge", "label")
ARM = os.environ["UM_ARM"]
ROOT = Path(os.environ["UM_ROOT"]).resolve()
REQUIRED = ("probe_inputs.json", "commit.jsonl", "mentions.jsonl", "probes/score/summary.json")


def check_root(root: Path) -> None:
    """``root`` must be a copy of the official um root outside the repo's outputs/ and data/."""
    for d in (HB / "outputs", HB / "data"):
        if root == d.resolve() or d.resolve() in root.parents:
            sys.exit(f"UM_ROOT {root} is inside {d}: use a copy of the um root outside the repo")
    for dom in ("", "capital_city", "holiday"):
        miss = [f for f in REQUIRED if not (root / dom / f).exists()]
        if miss:
            sys.exit(f"UM_ROOT {root} is not a copy of the official um root: {dom or 'country'} "
                     f"lacks {miss}")  # fmt: skip


def check_stage(argv: list[str]) -> None:
    stage = next((a for a in argv[1:] if not a.startswith("-") and a in DRV_STAGES), None)
    if stage not in ALLOWED:
        sys.exit(f"um_ladder_arm runs only {ALLOWED}, not {stage!r}")


DRV_STAGES = (
    *ALLOWED, "todo", "calib", "concepts", "probe-texts", "capture-rows", "probe-inputs",
    "distractors", "figures",
)  # fmt: skip
if ARM in U.METHODS:
    sys.exit(f"{ARM} is already a ladder method")

U.METHODS = (*U.METHODS, ARM)
U.METHOD_LABEL = {**U.METHOD_LABEL, ARM: ARM}
_orig_samples = U.cell_samples


def _cell_samples(method, row):  # type: ignore[no-untyped-def]
    return _orig_samples("s3d" if method == ARM else method, row)


U.cell_samples = _cell_samples
_orig_setup = DRV.setup_domain


def setup_domain(cat: str) -> None:
    DRV.OUT = ROOT
    DRV.PROBES_DIR = ROOT / "probes"
    DRV.PROBES_ADJ_DIR = ROOT / "probes_adj"
    DRV.LABEL_DIR = ROOT / "label"
    _orig_setup(cat)  # non-country: OUT/<cat>, its probes / label dirs and READOUTS
    DRV.READOUTS = {**DRV.READOUTS, ARM: f"{ARM}/sibling_eval.jsonl"}


DRV.setup_domain = setup_domain

if __name__ == "__main__":
    check_stage(sys.argv)
    check_root(ROOT)
    DRV.main()
