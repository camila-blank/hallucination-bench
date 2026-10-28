"""Read set, sites and capture rows of plan 0017 (after ``hbench mask gt``).

Reads ``data/mask/{items,labels}.json`` and ``rollouts.jsonl.gz``; writes

- ``data/mask/read_set.json``: ``{meta, items: [{task_id, archetype, label, stratum, reason}]}``
  (doubling-down excluded: its forced earlier turn is a real lie; every ``honest_wrong``;
  ``honest_correct`` up to ``--max-honest-correct``, ``resisted`` first; ``lie`` up to
  ``--max-lie``, round-robin over archetypes; seeded ``random.Random(17)``);
- ``data/mask/sites.json``: ``{meta, items: [{task_id, n_prompt_ids, n_ids, sites: [{pos, token,
  site_type, char}]}]}`` (``bank.select_sites``; ``char`` = end offset in the concatenation of
  the individually decoded tokens, approximate where a multi-byte character is split);
- ``data/mask/capture_rows.json``: ``[{label, family, input_ids, eval_positions, targets,
  look_for}]`` for global-workspace ``wsbench_capture_modal.py --prompts-json``.

    HF_HUB_OFFLINE=1 uv run --no-sync python scripts/mask/build_capture.py
"""

import argparse
import gzip
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from hallucination_bench.evals.mask import bank  # noqa: E402

SEED = 17
MODEL_ID = "Qwen/Qwen3.6-27B"
EXCLUDED_ARCHETYPES = ("doubling_down_known_facts",)


def choose(labels: list[dict], max_hc: int, max_lie: int) -> list[dict]:
    rng = random.Random(SEED)
    pool = [r for r in labels if r["archetype"] not in EXCLUDED_ARCHETYPES]
    out: list[dict] = []
    hw = [r for r in pool if r["label"] == "honest_wrong"]
    out += [{**r, "reason": "honest_wrong: all"} for r in hw]
    hc = [r for r in pool if r["label"] == "honest_correct"]
    res = [r for r in hc if r["stratum"] == "resisted"]
    rest = [r for r in hc if r["stratum"] != "resisted"]
    rng.shuffle(res)
    rng.shuffle(rest)
    pick = (res + rest)[:max_hc]
    out += [{**r, "reason": f"honest_correct: {r['stratum']}"} for r in pick]
    by_arch: dict[str, list[dict]] = defaultdict(list)
    for r in pool:
        if r["label"] == "lie":
            by_arch[r["archetype"]].append(r)
    for v in by_arch.values():
        rng.shuffle(v)
    lies: list[dict] = []
    while len(lies) < max_lie and any(by_arch.values()):
        for a in sorted(by_arch):
            if by_arch[a] and len(lies) < max_lie:
                lies.append(by_arch[a].pop())
    out += [{**r, "reason": "lie: round-robin by archetype"} for r in lies]
    return sorted(out, key=lambda r: r["task_id"])


def main() -> None:
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-honest-correct", type=int, default=140)
    ap.add_argument("--max-lie", type=int, default=90)
    args = ap.parse_args()
    labels = list(bank.load_labels().values())
    chosen = choose(labels, args.max_honest_correct, args.max_lie)
    roll = {}
    with gzip.open(bank.DATA_DIR / "rollouts.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["id"].endswith(":press"):
                roll[r["id"].removesuffix(":press")] = r
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    eos = tok.convert_tokens_to_ids(bank.IM_END)
    site_items, rows = [], []
    for r in chosen:
        tid = r["task_id"]
        g = roll[tid]
        pinned = list(g["pinned_token_ids"])
        while pinned and pinned[-1] == eos:
            pinned.pop()
        ids = list(g["prompt_token_ids"]) + pinned
        toks = [tok.decode([i]) for i in ids]
        sel = bank.select_sites(toks, len(g["prompt_token_ids"]))
        ends, n = [], 0
        for t in toks:
            n += len(t)
            ends.append(n)
        sites = [{"pos": p, "token": toks[p], "site_type": st, "char": ends[p]} for p, st in sel]
        site_items.append(
            {
                "task_id": tid,
                "n_prompt_ids": len(g["prompt_token_ids"]),
                "n_ids": len(ids),
                "sites": sites,
            }
        )
        rows.append(
            {
                "label": tid,
                "family": "mask",
                "input_ids": ids,
                "eval_positions": [s["pos"] for s in sites],
                "targets": [],
                "look_for": "",
            }
        )
    meta = {
        "plan": "0017",
        "seed": SEED,
        "excluded_archetypes": list(EXCLUDED_ARCHETYPES),
        "max_honest_correct": args.max_honest_correct,
        "max_lie": args.max_lie,
        "n_items": len(chosen),
        "n_sites": sum(len(s["sites"]) for s in site_items),
        "max_ids": max(s["n_ids"] for s in site_items),
    }
    keep = ("task_id", "archetype", "label", "stratum", "reason")
    (bank.DATA_DIR / "read_set.json").write_text(
        json.dumps({"meta": meta, "items": [{k: r[k] for k in keep} for r in chosen]}, indent=1)
        + "\n"
    )
    (bank.DATA_DIR / "sites.json").write_text(
        json.dumps({"meta": meta, "items": site_items}, ensure_ascii=False) + "\n"
    )
    (bank.DATA_DIR / "capture_rows.json").write_text(json.dumps(rows) + "\n")
    counts: dict[str, int] = defaultdict(int)
    for r in chosen:
        counts[r["reason"]] += 1
    print(json.dumps({**meta, "counts": counts}, indent=1))


if __name__ == "__main__":
    main()
