"""Plan 0020 step 4: metrics 1/2, recall diagnostics and the index export from the full run.

    uv run --no-sync python scripts/overrefusal/sj_report.py --tag full

Reads ``sj_records_<tag>.jsonl`` under ``$HBENCH_OR_OUT/overrefusal/claude-opus-5-5/`` and writes
``sj_metrics_<tag>.json`` plus ``index_counts.jsonl`` / ``index_meta.json`` there (``--no-index``
to skip the export). Gates: ``refusal-dm-v2/L*`` from ``data/overrefusal/hb_scores.json``.
"""

import argparse
import json

from hallucination_bench import paths
from hallucination_bench.evals.overrefusal import data, index_export, probe_report, run, sj_metrics

JUDGE_ROOT = "claude-opus-5-5"


def load_gates() -> dict[int, dict[str, bool]]:
    g = json.loads(data.HB_SCORES.read_text())["gates"]
    pre = probe_report.PRIMARY + "/L"
    return {int(k[len(pre) :]): v for k, v in g.items() if k.startswith(pre)}


def fmt(c: dict) -> str:
    return "  -  " if c["mean"] is None else f"{100 * c['mean']:5.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--no-index", action="store_true")
    a = ap.parse_args()
    root = run.eval_root(paths.OR_OUT, JUDGE_ROOT)
    recs = data.read_jsonl(root / f"sj_records_{a.tag}.jsonl")
    items = data.load_items()
    by = {it["id"]: it for it in items}
    gates = load_gates()
    pres = index_export.probe_presence(paths.OR_OUT)
    sj_metrics.annotate(recs, by, pres, gates)
    m = sj_metrics.metrics(recs)
    (root / f"sj_metrics_{a.tag}.json").write_text(json.dumps(m, indent=1) + "\n")
    if not a.no_index:
        sj_metrics.export_index(recs, items, root, gates)
    cols = ("m1_raw", "m1_probe_validated", "m2_off_topic_all", "m2_off_topic_never_refused")
    for rd, by_l in m.items():
        print(f"\n{rd}  (% items: m1 raw|valid, m2 all|never, recall hard|always)")
        for la, v in sorted(by_l.items(), key=lambda kv: int(kv[0])):
            rec = v["recall"]
            print(
                f"  L{la:>2} "
                + " ".join(fmt(v[c]) for c in cols)
                + f" {fmt(rec['hard_refused'])} {fmt(rec['always_refused'])}  {v['health']}"
            )


if __name__ == "__main__":
    main()
