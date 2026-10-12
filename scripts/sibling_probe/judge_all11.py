"""Plan 0004 step 1, set A: the claim map on the 11-layer arms (new cells only).

The caches of ``s3d-rl600-all11`` / ``jlens-all11`` are seeded once from the plan-0003 4-layer
arms (identical cell keys and prompts, so their v2.7 rows are reused); ``mlnla-rl600-all11``
starts empty. No coverage sample (plan 0004 uses the prefilter only); scope / coverage calls are
skipped (``extras=False``); J-lens summarises only the judged (prefiltered) cells.

    uv run python scripts/sibling_probe/judge_all11.py --arms s3d-rl600-all11,jlens-all11
    uv run python scripts/sibling_probe/judge_all11.py --arms ... --rebuild   # cache only
"""

import argparse
import asyncio
import shutil

from hallucination_bench.evals.sibling_probe import judge as J

SEED = {"s3d-rl600-all11": "s3d-rl600", "jlens-all11": "jlens"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="s3d-rl600-all11,jlens-all11,mlnla-rl600-all11")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--concurrency", type=int, default=32)
    a = ap.parse_args()
    arms = [x for x in a.arms.split(",") if x]
    # plan 0004 uses the plan-0003 prefilter only: no 5% coverage sample on the new arms
    J.coverage_sample = lambda *_a, **_k: set()
    for arm in arms:
        src = SEED.get(arm)
        dst = J.JUDGE_ROOT / arm / "cache.jsonl"
        if src and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(J.JUDGE_ROOT / src / "cache.jsonl", dst)
            print(f"[seed] {dst} <- {src}")
    asyncio.run(
        J.judge_arms(
            arms,
            rebuild=a.rebuild,
            concurrency=a.concurrency,
            summarize_all=False,
            extras=False,
        )
    )


if __name__ == "__main__":
    main()
