"""Write data/strongreject/{items.json, capture_rows.json} (plan 0016 §Read set and sites).

Read set: every `successful` prompt (cap 60) + up to 200 - n_successful (cap 150) `unsuccessful`
prompts drawn round-robin over families in random.Random(0) order. Sites (rev 5): every read
position (``bank.site_positions``; regions turn / tail / body), where the read
positions are tokens 0..n-1 of the Qwen chat template render (enable_thinking=False) up to and
including ``<|im_start|>assistant``. Needs the local Qwen3.6-27B tokenizer (HF_HUB_OFFLINE=1).

    HF_HUB_OFFLINE=1 uv run --no-sync python scripts/strongreject/build_bank.py \\
        [--outcomes outputs/strongreject/claude-opus-5-5/outcomes.json]
"""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from hallucination_bench import DATA, REPO_ROOT
from hallucination_bench.evals.strongreject import bank

MODEL_ID = "Qwen/Qwen3.6-27B"
MAX_SUCCESS = 60
MAX_SOFT = 40  # rev 3: soft_fail (rubric-refused, not an explicit refusal) = the engaged contrast
MAX_TOTAL = 200
MAX_UNSUCCESS = 150
MAX_POSITIONS = 1024  # the capture's --max-positions
OUT_ITEMS = DATA / "strongreject" / "items.json"
OUT_ROWS = DATA / "strongreject" / "capture_rows.json"


def select(prompts: list[dict], oc: dict[str, dict]) -> list[dict]:
    rng = random.Random(0)
    succ = [p for p in prompts if oc[p["id"]]["outcome"] == "successful"]
    rng.shuffle(succ)
    succ = succ[:MAX_SUCCESS]
    soft = [p for p in prompts if oc[p["id"]]["outcome"] == "soft_fail"]
    rng.shuffle(soft)
    soft = soft[:MAX_SOFT]
    n_uns = min(MAX_UNSUCCESS, MAX_TOTAL - len(succ) - len(soft))
    by_fam: dict[str, list[dict]] = defaultdict(list)
    for p in prompts:
        if oc[p["id"]]["outcome"] == "unsuccessful":
            by_fam[p["family"]].append(p)
    fams = sorted(by_fam)
    for f in fams:
        rng.shuffle(by_fam[f])
    uns: list[dict] = []
    while len(uns) < n_uns and any(by_fam[f] for f in fams):
        for f in fams:
            if by_fam[f] and len(uns) < n_uns:
                uns.append(by_fam[f].pop())
    chosen = {p["id"] for p in succ + soft + uns}
    return [p for p in prompts if p["id"] in chosen]


def build_item(tok, p: dict, oc: dict) -> dict:
    text = p["jailbreak_prompt"]
    templ = tok.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    render = bank.render_prompt(text)
    assert templ.startswith(render), (p["id"], templ[:80])
    enc = tok(templ, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    assert len(ids) <= MAX_POSITIONS, (p["id"], len(ids))
    n = sum(1 for _, e in offs if e <= len(render))
    toks = [tok.decode([i]) for i in ids]
    assert toks[n - 1] == "assistant" and toks[n - 2] == "<|im_start|>", (p["id"], toks[n - 3 : n])
    assert offs[n - 1][1] == len(render), p["id"]
    sites = []
    for pos, region in bank.site_positions(n):
        char = offs[pos][1]
        t = toks[pos]
        site = {"pos": pos, "token": t, "char": char, "region": region}
        if "�" in t:  # a piece of a multi-byte character split across tokens
            site["split_char"] = True
        else:
            assert render[:char].endswith(t), (p["id"], pos, t, render[max(0, char - 20) : char])
        sites.append(site)
    assert sites[-1]["pos"] == n - 1
    o = oc[p["id"]]
    counts = {k: o[k] for k in o if k.startswith("n_") or k == "mean_score"}
    return {
        **{k: p[k] for k in ("id", "forbidden_id", "category", "source", "family")},
        "forbidden_prompt": p["forbidden_prompt"],
        "jailbreak_prompt": text,
        "outcome": o["outcome"],
        "outcome_counts": counts,
        "prompt_render": render,
        "n_positions": n,
        "n_template_tokens": len(ids),
        "sites": sites,
    }


def main() -> None:
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--outcomes",
        default=str(REPO_ROOT / "outputs/strongreject/claude-opus-5-5/outcomes.json"),
    )
    args = ap.parse_args()
    oc_doc = json.loads(Path(args.outcomes).read_text())
    oc = {r["id"]: r for r in oc_doc["prompts"]}
    prompts = bank.load_prompts()["prompts"]
    assert set(oc) == {p["id"] for p in prompts}, "outcomes do not cover prompts.json"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    chosen = select(prompts, oc)
    items = [build_item(tok, p, oc) for p in chosen]
    n_sites = sum(len(it["sites"]) for it in items)
    meta = {
        "plan": "0016",
        "n_items": len(items),
        "n_sites": n_sites,
        "outcomes": dict(Counter(it["outcome"] for it in items)),
        "families": dict(Counter(f"{it['family']}:{it['outcome']}" for it in items)),
        "sites": {"all_positions": True, "tail": bank.TAIL, "turn": bank.TURN},
        "selection": {
            "max_successful": MAX_SUCCESS,
            "max_soft_fail": MAX_SOFT,
            "max_total": MAX_TOTAL,
            "max_unsuccessful": MAX_UNSUCCESS,
            "seed": 0,
        },
        "outcomes_file_sha256": hashlib.sha256(Path(args.outcomes).read_bytes()).hexdigest(),
        "grader": {k: oc_doc["meta"][k] for k in ("judge", "effort", "prompt_version")},
        "prompts_meta": bank.load_prompts()["meta"],
        "tokenizer": MODEL_ID,
    }
    OUT_ITEMS.write_text(
        json.dumps({"meta": meta, "items": items}, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rows = [
        {
            "label": it["id"],
            "family": "strongreject",
            "render": "chat",
            "user": it["jailbreak_prompt"],
            "look_for": "",
            "targets": [],
            "eval_positions": [s["pos"] for s in it["sites"]],
        }
        for it in items
    ]
    OUT_ROWS.write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT_ITEMS}: {len(items)} items, {n_sites} sites; {meta['outcomes']}")
    print(meta["families"])


if __name__ == "__main__":
    main()
