"""Plan 0020 rev 7 (exploratory): `orclaim-v1` statement labels + the claim-unit metric.

    uv run --no-sync python scripts/overrefusal/sj_claims.py --tag calib   # calibration sample
    uv run --no-sync python scripts/overrefusal/sj_claims.py --tag full    # 160 x 3 x 11

Reads the cached digests in ``sj_records_full.jsonl`` (never_refused only; no re-digesting) under
``$HBENCH_OR_OUT/overrefusal/claude-opus-5-5/``; cache ``claims/<reader>.jsonl``. Writes
``sj_claims_<tag>.json`` (metric, consistency with orsj-v3, health) and
``sj_claims_rows_<tag>.jsonl``; ``--tag full`` also writes ``index_counts_claims.jsonl`` (plan
0019 columns; the existing ``index_counts.jsonl`` is untouched).
"""

import argparse
import asyncio
import json
import random
from typing import Any

from hallucination_bench import paths
from hallucination_bench.evals.overrefusal import data, index_export, run, sclaim, sjudge

JUDGE_ROOT = "claude-opus-5-5"
READERS = ("s3d-rl600", "mlnla-rl600-all11", "jlens")
CALIB_LAYERS = (20, 44, 60)


def calib_sample(recs: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """>= 30 digests over 3 readers x L20/L44/L60, >= 10 with an orsj-v3 refusal: every refusal
    digest of the sparse readers, 2 per layer for the O-lens, plus 2 non-refusal per (reader,
    layer)."""
    rng = random.Random(seed)
    pick: list[dict[str, Any]] = []
    for rd in READERS:
        for la in CALIB_LAYERS:
            cell = sorted(
                (r for r in recs if r["reader"] == rd and r["layer"] == la),
                key=lambda r: r["item"],
            )
            ref = [r for r in cell if r["verdict"] == "refusal"]
            non = [
                r for r in cell if r["verdict"] == "no_refusal" and r["digest"]["status"] == "ok"
            ]
            pick += rng.sample(ref, min(2, len(ref)) if rd == READERS[0] else len(ref))
            pick += rng.sample(non, min(2, len(non)))
    return pick


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=70)
    ap.add_argument("--concurrency", type=int, default=64)
    a = ap.parse_args()
    root = run.eval_root(paths.OR_OUT, JUDGE_ROOT)
    recs = [
        r
        for r in data.read_jsonl(root / "sj_records_full.jsonl")
        if r["population"] == "never_refused"
    ]
    if a.tag != "full":
        recs = calib_sample(recs, a.seed)
    print(f"{len(recs)} digests ({a.tag})")
    caches = sclaim.open_caches(root, READERS)
    rows = asyncio.run(sclaim.label_records(recs, caches, concurrency=a.concurrency))
    for c in caches.values():
        c.close()
    items = data.load_items()
    by = {it["id"]: it for it in items}
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "sj_report", Path(__file__).with_name("sj_report.py")
    )
    assert spec and spec.loader
    rep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rep)
    gates = rep.load_gates()
    sclaim.annotate(rows, by, index_export.probe_presence(paths.OR_OUT), gates)
    res = {
        "versions": {"claim_labeller": sclaim.CL_VERSION, "strict_judge": sjudge.SJ_VERSION},
        "unit": "claims",
        "metrics": sclaim.metrics(rows),
        "consistency_with_orsj": sclaim.consistency(rows),
        "n_digests": len(rows),
        "n_label_failed": sum(r.get("labels") is None and r["digest_status"] == "ok" for r in rows),
    }
    (root / f"sj_claims_{a.tag}.json").write_text(json.dumps(res, indent=1) + "\n")
    with (root / f"sj_claims_rows_{a.tag}.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    if a.tag == "full":
        index_export.write(sclaim.index_rows(rows, items), root / "index_counts_claims.jsonl")
    print(json.dumps({k: res[k] for k in ("consistency_with_orsj", "n_label_failed")}))
    for rd, by_l in res["metrics"].items():
        for la, v in sorted(by_l.items(), key=lambda kv: int(kv[0])):
            m, r_ = v["claim_validated"]["mean"], v["claim_raw"]["mean"]
            print(f"{rd:18} L{la:>2} val={m} raw={r_} cpi={v['claims_per_item']:.1f} {v['health']}")


if __name__ == "__main__":
    main()
