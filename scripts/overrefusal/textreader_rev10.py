"""Plan 0015 rev 10 driver: text-reader filter (run A, Opus) over the 327 bank N / I items, then
fresh floors (run B Opus, run S Sonnet) on filtered pairs if >= 50 N survive. Writes
outputs/overrefusal/textreader_rev10.json. Loads cf_influence.scoring from a global-workspace
checkout ($HBENCH_GW_SCORING; default: the refusal-prediction worktree)."""

import asyncio
import collections
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
J = "claude-sonnet-5-5"
items = data.load_items()
pool = [it for it in items if it["population"] in ("never_refused", "hard_refused")]
by = {it["id"]: it for it in items}
gen_cache = Cache(Path("outputs/overrefusal/textreader_gen_cache.jsonl"))


def run_reader(ids, model, tag):
    samp = asyncio.run(
        T.generate({i: by[i]["prompt_render"] for i in ids}, gen_cache, model=model, tag=tag)
    )
    return T.judge(samp, OUT, J, f"textreader-{tag}")


A = run_reader([it["id"] for it in pool], "claude-opus-5-5", "trA")
spend = {"trA_judge": A.pop("_spend_judge")["usd"]}
surv = []
c = collections.Counter()
for it in pool:
    a = A[it["id"]]
    wrong = a["flag"] if it["population"] == "never_refused" else not a["flag"]
    c[(it["population"], "flagged" if a["flag"] else "not", "survive" if wrong else "drop")] += 1
    if wrong:
        surv.append(it)
print("filter", dict(c))
ns = sum(it["population"] == "never_refused" for it in surv)
isv = sum(it["population"] == "hard_refused" for it in surv)
print("survivors N", ns, "I", isv)
res = {
    "filter_counts": {"|".join(k): v for k, v in c.items()},
    "survivors": {"N": ns, "I": isv},
    "trA": A,
}
pairs = w.build_pairs(surv)
res["pairs"] = pairs
print("filtered pairs", len(pairs))
if ns >= 50 and len(pairs) >= 50:
    P = [scoring.Pair(p["pair_id"], p["cluster"]) for p in pairs]
    ids = sorted({i for p in pairs for i in (p["I"], p["N"])})
    for tag, model in (("trB", "claude-opus-5-5"), ("trS", "claude-sonnet-5-5")):
        R = run_reader(ids, model, tag)
        spend[tag + "_judge"] = R.pop("_spend_judge")["usd"]
        fi, fn, afc = T.pair_scores(pairs, R)
        s = scoring.score_cell(P, fi, fn)
        ci = scoring.cluster_bootstrap_ci(P, fi, fn)
        res[tag] = {"net": s.net, "ci95": ci, "tpr": s.tpr, "fpr": s.fpr, "afc": afc, "items": R}
        print(tag, round(s.net, 3), ci, "afc", round(afc, 3))
res["spend_judge"] = spend
Path("outputs/overrefusal/textreader_rev10.json").write_text(json.dumps(res, default=str))
gen_cache.close()
