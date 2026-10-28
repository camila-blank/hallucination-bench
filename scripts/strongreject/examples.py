"""Print sampled counted compliance mentions (and refusal mentions) of one arm for hand reading
(plan 0016 §Results): item, family, outcome, layer, position, region, label, quote, Stage E
verdict, the readout (truncated) and the prompt text before the read site (tail).

    uv run --no-sync python scripts/strongreject/examples.py <arm> [--label harmful_compliance]
        [--n 12] [--seed 0] [--all-verdicts]
"""

import argparse
import json
import random

from hallucination_bench import REPO_ROOT
from hallucination_bench.evals.strongreject import bank, detect, echo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--label", default="harmful_compliance")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--judge", default="claude-opus-5-5")
    ap.add_argument("--all-verdicts", action="store_true", help="include unconfirmed mentions")
    ap.add_argument("--readouts-root", default="/workspace/hb-readouts-strongreject")
    a = ap.parse_args()
    items = {it["id"]: it for it in bank.load_items()}
    root = REPO_ROOT / "outputs/strongreject" / a.judge / a.arm
    pool = []
    for line in (root / "records.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        for m in r["mentions"]:
            if m["label"] != a.label:
                continue
            e = m.get("echo")
            if m["label"] in echo.COMPLIANCE_LABELS and not a.all_verdicts and not echo.counted(e):
                continue
            pool.append((r, m))
    random.Random(a.seed).shuffle(pool)
    pick = pool[: a.n]
    want = {(r["id"], r["layer"], r["pos"]) for r, _ in pick}
    texts = {}
    for line in open(f"{a.readouts_root}/{a.arm}/strongreject.jsonl", encoding="utf-8"):
        row = json.loads(line)
        k = (row["id"], row["layer"], row["pos"])
        if k in want:
            texts[k] = row["samples"][0] if "samples" in row else detect.bag_text(row["tokens"])
    print(f"{a.arm} {a.label}: {len(pool)} mentions; showing {len(pick)}\n")
    for r, m in pick:
        it = items[r["id"]]
        site = next(s for s in it["sites"] if s["pos"] == r["pos"])
        pre = it["prompt_render"][: site["char"]][-160:].replace("\n", "\\n")
        txt = (texts.get((r["id"], r["layer"], r["pos"])) or "")[:400].replace("\n", " ")
        print(
            f"- {r['id']} [{it['family']}, {it['outcome']}] L{r['layer']} p{r['pos']} "
            f"({r['region']}) quote={m['quote']!r} stageE={m.get('echo')}"
        )
        print(f"    prefix: ...{pre}")
        print(f"    readout: {txt}\n")


if __name__ == "__main__":
    main()
