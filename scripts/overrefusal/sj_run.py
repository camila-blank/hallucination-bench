"""Plan 0020 driver: digest -> strict judge -> off-topic judge over (reader, item, layer) units.

    uv run --no-sync python scripts/overrefusal/sj_run.py --tag calib --layers 20,44,60 \
        --calib-per-pop 10   # calibration: per reader x layer, 10 items per population
    uv run --no-sync python scripts/overrefusal/sj_run.py --tag full   # 3 x 11 x 427

Readouts: ``$HBENCH_OR_READOUTS_V3`` (11 layers). Outputs under
``$HBENCH_OR_OUT/overrefusal/claude-opus-5-5/``: caches ``digest/ sjudge/ offtopic/<reader>.jsonl``
and records ``sj_records_<tag>.jsonl``.
"""

import argparse
import asyncio
import random
from collections import defaultdict
from pathlib import Path

from hallucination_bench import paths
from hallucination_bench.evals.overrefusal import data, run, sjudge
from hallucination_bench.readouts import load_readouts

READERS = ("s3d-rl600", "mlnla-rl600-all11", "jlens")
JUDGE_ROOT = "claude-opus-5-5"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--layers", default="20,24,28,32,36,40,44,48,52,56,60")
    ap.add_argument("--readers", default=",".join(READERS))
    ap.add_argument("--calib-per-pop", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--calib-pops", default=",".join(data.POPULATIONS))
    ap.add_argument("--exclude-records", default="", help="skip items already in these records")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--stages", default="digest,judge,offtopic")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    items = data.load_items()
    if a.calib_per_pop:
        import json

        skip = set()
        for f in filter(None, a.exclude_records.split(",")):
            skip |= {json.loads(x)["item"] for x in Path(f).read_text().splitlines() if x}
        rng = random.Random(a.seed)
        pick = []
        for pop in a.calib_pops.split(","):
            its = sorted(
                (it for it in items if it["population"] == pop and it["id"] not in skip),
                key=lambda x: x["id"],
            )
            pick += rng.sample(its, a.calib_per_pop)
        items = pick
    by_id = {it["id"]: it for it in items}
    root = run.eval_root(paths.OR_OUT, JUDGE_ROOT)
    readers = a.readers.split(",")
    caches = sjudge.open_caches(root, readers)
    units = []
    for rd in readers:
        cells, _ = load_readouts(
            Path(paths.OR_READOUTS_V3) / rd / run.READOUTS_FILE,
            positions=data.positions(items),
            layers=layers,
        )
        grp: dict[tuple[str, int], list] = defaultdict(list)
        for c in cells:
            grp[(c.id, c.layer)].append(c)
        for iid, it in sorted(by_id.items()):
            for la in layers:
                units.append(sjudge.Unit(rd, it, la, grp.get((iid, la), [])))
    print(
        f"{len(units)} units ({len(readers)} readers x {len(layers)} layers x {len(items)} items)"
    )
    recs = asyncio.run(
        sjudge.run_units(units, caches, concurrency=a.concurrency, stages=a.stages.split(","))
    )
    for d in caches.values():
        for c in d.values():
            c.close()
    out = root / f"sj_records_{a.tag}.jsonl"
    sjudge.write_records(recs, out)
    n = defaultdict(int)
    for r in recs:
        n[(r["digest"]["status"], r.get("verdict"))] += 1
    print(f"wrote {out}: {dict(n)}")


if __name__ == "__main__":
    main()
