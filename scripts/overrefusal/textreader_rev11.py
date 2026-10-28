"""Plan 0015 rev 11 driver: text-reader floors on the 141 frozen pairs (runs A / B Opus with a
Sonnet fallback for API-refused samples, run S Sonnet), judged by the lens detector / verifier.
Writes outputs/overrefusal/textreader_rev11.json and, with --family-dir, the family's
hits/text_reader_*.json. Loads cf_influence.scoring from $HBENCH_GW_SCORING (default: the
refusal-prediction worktree)."""

import argparse
import asyncio
import importlib.util
import json
import os
from pathlib import Path

from hallucination_bench.cache import Cache
from hallucination_bench.evals.overrefusal import data
from hallucination_bench.evals.overrefusal import textreader as T
from hallucination_bench.evals.overrefusal import wsb_export as w

_GW = "/workspace/gw-wsb-refusal-prediction/src/global_workspace/olens_suite/cf_influence"
spec = importlib.util.spec_from_file_location(
    "scoring", os.environ.get("HBENCH_GW_SCORING") or f"{_GW}/scoring.py"
)
scoring = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scoring)

OUT = Path("outputs")
JUDGE = "claude-sonnet-5-5"
OPUS, SONNET = "claude-opus-5-5", "claude-sonnet-5-5"


def stats(pairs, tr):
    P = [scoring.Pair(p["pair_id"], p["cluster"]) for p in pairs]
    fi = {p["pair_id"]: bool(tr[p["I"]]["flag"]) for p in pairs}
    fn = {p["pair_id"]: bool(tr[p["N"]]["flag"]) for p in pairs}
    s = scoring.score_cell(P, fi, fn)
    lo, hi = scoring.cluster_bootstrap_ci(P, fi, fn)
    a, alo, ahi = T.afc_ci(pairs, tr)
    return {
        "n": s.n_pairs,
        "pass": s.pass_rate,
        "rev": s.rev_rate,
        "net": s.net,
        "tpr": s.tpr,
        "fpr": s.fpr,
        "ci95": [lo, hi],
        "afc": a,
        "afc_ci95": [alo, ahi],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family-dir", default="")
    args = ap.parse_args()
    items = {it["id"]: it for it in data.load_items()}
    pairs = w.build_pairs(list(items.values()))
    ids = sorted({i for p in pairs for i in (p["I"], p["N"])})
    renders = {i: items[i]["prompt_render"] for i in ids}
    cache = Cache(Path("outputs/overrefusal/textreader_gen_cache.jsonl"))
    res: dict = {"n_pairs": len(pairs), "runs": {}}
    hits: dict = {}
    for tag, model, fb in (("trA", OPUS, SONNET), ("trB", OPUS, SONNET), ("trS", SONNET, None)):
        samp = asyncio.run(T.generate(renders, cache, model=model, tag=tag, fallback=fb))
        texts = {i: [x["text"] for x in v] for i, v in samp.items()}
        tr = T.judge(texts, OUT, JUDGE, f"textreader-{tag}")
        tr.pop("_spend_judge")
        n_ref = {i: sum(x["refused"] for x in v) for i, v in samp.items()}
        routes = {i: [x["route"] for x in v] for i, v in samp.items()}
        run = {
            "model": model,
            "fallback": fb,
            "with_fallback": stats(pairs, tr),
            "n_api_refused": sum(n_ref.values()),
            "n_fallback_used": sum(r == SONNET for v in routes.values() for r in v) if fb else 0,
            "n_empty": sum(not t.strip() for v in texts.values() for t in v),
        }
        if fb:
            # (b) sensitivity: judge the primary-only texts; an API refusal = a refusal sample
            prim = {
                i: [x["text"] if x["route"] == model else "" for x in v] for i, v in samp.items()
            }
            tp = T.judge(prim, OUT, JUDGE, f"textreader-{tag}")
            tp.pop("_spend_judge")
            sens = {}
            for i in ids:
                sc = tp[i]["score"] + n_ref[i]
                sens[i] = {"score": sc, "flag": sc >= T.QUORUM}
            run["refusal_as_flag"] = stats(pairs, sens)
            hits[f"{tag}_refusal_as_flag"] = sens
        res["runs"][tag] = run
        hits[tag] = {
            i: {
                "score": tr[i]["score"],
                "flag": tr[i]["flag"],
                "routes": routes[i],
                "n_api_refused": n_ref[i],
                "refusals": tr[i]["refusals"],
            }
            for i in ids
        }
        print(tag, json.dumps({k: v for k, v in run.items() if k != "model"}, default=str))
    cache.close()
    Path("outputs/overrefusal/textreader_rev11.json").write_text(json.dumps(res, indent=1))
    if args.family_dir:
        hb_to_key = {}
        for p in pairs:
            hb_to_key[p["I"]] = f"{p['pair_id']}|I"
            hb_to_key[p["N"]] = f"{p['pair_id']}|N"
        d = Path(args.family_dir) / "hits"
        for name, h in hits.items():
            keyed = {hb_to_key[i]: v for i, v in h.items()}
            (d / f"text_reader_{name}.json").write_text(json.dumps(keyed, indent=0) + "\n")
        (Path(args.family_dir) / "hits" / "text_reader_summary.json").write_text(
            json.dumps(res, indent=1) + "\n"
        )


if __name__ == "__main__":
    main()
