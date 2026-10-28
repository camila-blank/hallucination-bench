"""Judge mix per arm (plan 0016 review fix): for every current cell, which model answered the
detector row the rebuild actually uses (the newest ok row for the cell's key + fingerprint), and
the same for every Stage E mention row it uses. No API calls.

    uv run --no-sync python scripts/strongreject/judge_mix.py [--arms a,b,c]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from hallucination_bench.evals.strongreject import run as R


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="s3d-rl600,mlnla-rl600,jlens")
    ap.add_argument("--readouts-root", default="/workspace/hb-readouts-strongreject")
    a = ap.parse_args()
    judge = R.Judge()
    out = {}
    for arm in a.arms.split(","):
        A = R.open_arm(
            arm,
            Path(a.readouts_root) / arm / R.READOUTS_FILE,
            Path("outputs"),
            judge,
            allow_missing=True,
            readonly=True,
        )
        d, e = Counter(), Counter()
        for c in A.cells:
            j = R.d_job(c, judge)
            if R.d_empty(j):
                continue
            row = A.cache.get(j.key, j.fp)
            if row is None:
                continue
            d[(row.get("meta") or {}).get("answered_by", judge.model)] += 1
            for ej in R.e_jobs(A, c, row["result"], judge):
                er = A.cache.get(ej.key, ej.fp)
                if er is not None:
                    e[(er.get("meta") or {}).get("answered_by", judge.model)] += 1
        out[arm] = {
            "detect": dict(d),
            "detect_sonnet_share": d[judge.fallback] / max(1, sum(d.values())),
            "echo": dict(e),
            "echo_sonnet_share": e[judge.fallback] / max(1, sum(e.values())),
        }
        print(arm, json.dumps(out[arm]))
    dst = R.model_root(Path("outputs"), judge) / "judge_mix.json"
    dst.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
