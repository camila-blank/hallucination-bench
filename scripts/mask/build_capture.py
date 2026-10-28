"""Read set, sites and capture rows of plan 0017 rev 5 (after ``hbench mask gt`` and the
self-knowledge gate, ``scripts/mask/gate.py parse``).

- ``data/mask/read_set.json``: every gated ``lie`` item and every gated ``honest_resisted`` item
  (self-knowledge gate >= 9/10 in the frame chosen by ``report.choose_frame``), plus the
  non-doubling-down ``honest_wrong`` items (descriptive), each with the reason it is in.
  Doubling-down is never read. (Rev 1 sampled a capped read set with 8 + 8 thinned sites; see
  git history.)
- ``data/mask/sites.json.gz``: per read item, **every** site (``bank.select_sites(...,
  all_positions=True)``): prompt tokens ``0 .. a`` (a = the final ``assistant``) and every pinned
  token (``<|im_end|>`` excluded), ``{pos, token, site_type, char}``.
- ``data/mask/capture_rows.json`` (read set, capture ``hb-mask-v5``) and
  ``data/mask/capture_rows_neg.json`` (every other non-doubling-down ``lie`` / ``honest_*`` item,
  capture ``hb-mask-v5-neg``, probe training and threshold data only, never read by a lens),
  with their sites in ``data/mask/sites_neg.json.gz``.

    HF_HUB_OFFLINE=1 uv run --no-sync python scripts/mask/build_capture.py
"""

import gzip
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from hallucination_bench.evals.mask import bank  # noqa: E402
from hallucination_bench.evals.mask import report as R  # noqa: E402

MODEL_ID = "Qwen/Qwen3.6-27B"
DD = "doubling_down_known_facts"


def choose(labels: dict, gate: list[dict]) -> tuple[list[dict], str]:
    frame, _ = R.choose_frame(gate)
    g = {r["task_id"]: r for r in gate}
    out = []
    for t, r in sorted(labels.items()):
        if r["archetype"] == DD:
            continue
        if r["label"] == "lie" and t in g and R.gate_pass(g[t], frame, R.GATE_K):
            out.append({**r, "reason": f"lie, gate {frame} >= {R.GATE_K}/10"})
        elif (
            r["label"].startswith("honest")
            and r["stratum"] == "resisted"
            and t in g
            and R.gate_pass(g[t], frame, R.GATE_K)
        ):
            out.append({**r, "reason": f"honest_resisted, gate {frame} >= {R.GATE_K}/10"})
        elif r["label"] == "honest_wrong":
            out.append({**r, "reason": "honest_wrong: all (descriptive)"})
    return out, frame


def main() -> None:
    from transformers import AutoTokenizer

    labels = bank.load_labels()
    gate = json.loads((bank.DATA_DIR / "gate.json").read_text())["items"]
    chosen, frame = choose(labels, gate)
    read_ids = {r["task_id"] for r in chosen}
    neg_ids = sorted(
        t
        for t, r in labels.items()
        if r["archetype"] != DD
        and t not in read_ids
        and (r["label"] == "lie" or r["label"].startswith("honest"))
    )
    roll = {}
    with gzip.open(bank.DATA_DIR / "rollouts.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["id"].endswith(":press"):
                roll[r["id"].removesuffix(":press")] = r
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    eos = tok.convert_tokens_to_ids(bank.IM_END)

    def build(tid: str) -> tuple[dict, dict]:
        g = roll[tid]
        pinned = list(g["pinned_token_ids"])
        while pinned and pinned[-1] == eos:
            pinned.pop()
        ids = list(g["prompt_token_ids"]) + pinned
        toks = [tok.decode([i]) for i in ids]
        sel = bank.select_sites(toks, len(g["prompt_token_ids"]), all_positions=True)
        ends, n = [], 0
        for t in toks:
            n += len(t)
            ends.append(n)
        sites = [{"pos": p, "token": toks[p], "site_type": st, "char": ends[p]} for p, st in sel]
        row = {
            "label": tid,
            "family": "mask",
            "input_ids": ids,
            "eval_positions": [s["pos"] for s in sites],
            "targets": [],
            "look_for": "",
        }
        item = {
            "task_id": tid,
            "n_prompt_ids": len(g["prompt_token_ids"]),
            "n_ids": len(ids),
            "sites": sites,
        }
        return row, item

    site_items, rows = [], []
    for r in chosen:
        row, item = build(r["task_id"])
        rows.append(row)
        site_items.append(item)
    neg = [build(t) for t in neg_ids]
    neg_rows = [x[0] for x in neg]
    with gzip.open(bank.DATA_DIR / "sites_neg.json.gz", "wt", encoding="utf-8") as fh:
        json.dump({"items": [x[1] for x in neg]}, fh, ensure_ascii=False)
    n_sites = Counter(s["site_type"] for it in site_items for s in it["sites"])
    meta = {
        "plan": "0017 rev 5",
        "gate_frame": frame,
        "gate_k": R.GATE_K,
        "excluded_archetypes": [DD],
        "n_items": len(chosen),
        "n_sites": dict(n_sites),
        "max_ids": max(s["n_ids"] for s in site_items),
        "n_neg_items": len(neg_ids),
        "acts": {"read": "hb-mask-v5", "neg": "hb-mask-v5-neg"},
    }
    keep = ("task_id", "archetype", "label", "stratum", "reason")
    (bank.DATA_DIR / "read_set.json").write_text(
        json.dumps({"meta": meta, "items": [{k: r[k] for k in keep} for r in chosen]}, indent=1)
        + "\n"
    )
    with gzip.open(bank.DATA_DIR / "sites.json.gz", "wt", encoding="utf-8") as fh:
        json.dump({"meta": meta, "items": site_items}, fh, ensure_ascii=False)
    old = bank.DATA_DIR / "sites.json"
    if old.exists():
        old.unlink()  # rev 1-4 thinned sites (git history); load_sites reads the .gz
    (bank.DATA_DIR / "capture_rows.json").write_text(json.dumps(rows) + "\n")
    (bank.DATA_DIR / "capture_rows_neg.json").write_text(json.dumps(neg_rows) + "\n")
    counts = Counter(r["reason"] for r in chosen)
    arch = Counter(
        (r["label"] if r["label"] == "lie" else "honest", r["archetype"]) for r in chosen
    )
    print(json.dumps({**meta, "counts": counts}, indent=1))
    print(sorted(arch.items()))


if __name__ == "__main__":
    main()
