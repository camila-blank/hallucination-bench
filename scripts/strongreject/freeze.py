"""Freeze the final strongreject set (plan 0016 rev 5): write data/strongreject/outcomes.json
(per-prompt ground truth without any rollout or grader text) and data/strongreject/MANIFEST.sha256
(sha256 of every file the eval reads: sources, prompts, outcomes, items, capture rows).
``tests/test_strongreject.py`` checks the manifest. Re-run only on a deliberate new version.

    uv run --no-sync python scripts/strongreject/freeze.py [--outcomes <outputs/.../outcomes.json>]
"""

import argparse
import hashlib
import json
from pathlib import Path

from hallucination_bench import DATA, REPO_ROOT

D = DATA / "strongreject"
KEEP = (
    "id",
    "forbidden_id",
    "category",
    "family",
    "outcome",
    "n_samples",
    "n_graded",
    "n_refused",
    "n_explicit_refusal",
    "n_adopted",
    "n_complied_nonzero",
    "mean_score",
    "adopted",
    "graders",
    "n_fallback",
    "finish",
)
FILES = (
    "source/strongreject_dataset.csv",
    "source/wrapping_jailbreaks.json",
    "source/judge_templates.json",
    "prompts.json",
    "outcomes.json",
    "items.json",
    "capture_rows.json",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--outcomes", default=str(REPO_ROOT / "outputs/strongreject/claude-opus-5-5/outcomes.json")
    )
    a = ap.parse_args()
    src = json.loads(Path(a.outcomes).read_text())
    slim = {
        "meta": {k: v for k, v in src["meta"].items() if k != "spend"},
        "families": src["families"],
        "prompts": [
            {
                **{k: r[k] for k in KEEP if k in r},
                "refusal": [g and g["refusal"] for g in r["grades"]],
                "explicit_refusal": [g and g["explicit_refusal"] for g in r["grades"]],
                "score": [g and g["score"] for g in r["grades"]],
            }
            for r in src["prompts"]
        ],
    }
    (D / "outcomes.json").write_text(json.dumps(slim, indent=1) + "\n", encoding="utf-8")
    lines = [f"{hashlib.sha256((D / f).read_bytes()).hexdigest()}  {f}" for f in FILES]
    (D / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
