"""Detector judge-model agreement (plan 0016 rev 3): Opus 5.5 refused most O-lens detector
batches, so most O-lens cells were labelled by the Sonnet 5.5 fallback while J-lens cells were
mostly Opus. This re-labels a seeded sample of cells that OPUS answered with Sonnet (single-cell
calls, same prompt) and reports per-label cell-level agreement (Cohen's kappa).

    env -u ANTHROPIC_API_KEY uv run --no-sync python scripts/strongreject/judge_agreement.py \\
        s3d-rl600 [--n 300]
Writes outputs/strongreject/<judge>/<arm>/judge_agreement.json (+ its own cache).
"""

import argparse
import asyncio
import json
import random
from pathlib import Path

from hallucination_bench.cache import Cache, fingerprint
from hallucination_bench.evals.strongreject import detect
from hallucination_bench.evals.strongreject import run as R
from hallucination_bench.llm import LLM


def kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--other", default="claude-sonnet-5-5")
    ap.add_argument("--readouts-root", default="/workspace/hb-readouts-strongreject")
    a = ap.parse_args()
    judge = R.Judge()
    path = Path(a.readouts_root) / a.arm / R.READOUTS_FILE
    arm = R.open_arm(a.arm, path, Path("outputs"), judge, readonly=True)
    pool = []
    for c in arm.cells:
        j = R.d_job(c, judge)
        row = arm.cache.get(j.key, j.fp)
        if row is None or R.d_empty(j):
            continue
        if (row.get("meta") or {}).get("answered_by", judge.model) == judge.model:
            pool.append((c, j, row["result"]))
    random.Random(0).shuffle(pool)
    pool = pool[: a.n]
    cache = Cache(arm.root / "judge_agreement_cache.jsonl")
    llm = LLM(concurrency=32, effort=judge.effort)

    async def one(c, j):
        fp = fingerprint(detect.PROMPT_VERSION, a.other, judge.effort, detect.SYSTEM, j.canon)
        row = cache.get(j.key, fp)
        if row is not None:
            return row["result"]
        s, u = detect.render_batch([j.canon])
        o = await llm.call(model=a.other, system=s, user=u, schema=detect.SCHEMA)
        bp = detect.parse_batch(o.result, [j.readouts], [j.hay])
        r = None if bp.failed else bp.cells[0].to_result()
        cache.put(j.key, fp, {"result": r, "meta": {"usage": o.usage.to_dict()}})
        return r

    others = await asyncio.gather(*[one(c, j) for c, j, _ in pool])
    await llm.aclose()
    out: dict = {"arm": a.arm, "n_sampled": len(pool), "other": a.other, "labels": {}}
    ok = [(p, o) for p, o in zip(pool, others, strict=True) if o is not None]
    out["n_compared"] = len(ok)
    for lab in detect.LABEL_NAMES:
        x = [any(m["label"] == lab for m in p[2]["mentions"]) for p, _ in ok]
        y = [any(m["label"] == lab for m in o["mentions"]) for _, o in ok]
        out["labels"][lab] = {
            "opus_rate": sum(x) / len(x) if x else None,
            "other_rate": sum(y) / len(y) if y else None,
            "agreement": sum(p == q for p, q in zip(x, y, strict=True)) / len(x) if x else None,
            "kappa": kappa(x, y),
        }
    dst = arm.root / "judge_agreement.json"
    dst.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
