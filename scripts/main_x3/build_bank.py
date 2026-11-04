"""Plan 0028 step 3: build the 3x main bank (``data/wsbench_chat_x3/``) from WSB + rollouts.

The 149 WSB items come first, rebuilt from ``data/wsbench_chat/capture_rows.json`` token ids;
their item fields must equal the WSB copy byte for byte and their ``wsb`` sites must equal the
WSB sites (the 8-thin), else the build fails. New items (``rollouts_*.json`` from
``gen_modal.py``) follow in label order; items with no read site are dropped (as WSB did).
Sites = ``thin(response_sites(...), 16)`` (``expand.item_sites``).

Writes items.json, capture_rows.json (every site) and capture_new_<tag>.json (only the cells to
read: every site of a new item, the non-``wsb`` sites of a WSB item; tag a = WSB items + new
dailydialog, tag b = new lmsys; an existing tag file must be reproduced exactly).

    uv run --no-sync python scripts/main_x3/build_bank.py --rollouts a.json,b.json --out DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from hallucination_bench.evals.wsbench_chat.expand import (  # noqa: E402
    MAX_SITES_WSB,
    MAX_SITES_X3,
    item_sites,
    seed_key,
)

WSB = REPO / "data" / "wsbench_chat"
MAX_POSITIONS = 2048  # reads.sh capture --max-positions (modal_pipeline.MAX_POSITIONS)
ITEM_KEYS = ("id", "stratum", "source", "prompt", "response", "response_start")


def user_turn(decoded_prompt: str) -> str:
    import re

    m = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", decoded_prompt, re.S)
    if m is None:
        raise ValueError(f"no user turn in {decoded_prompt[:120]!r}")
    return m.group(1).strip()


def build(rollout_files: list[Path], tok) -> dict:
    raw = tok.backend_tokenizer  # tokenizers.Tokenizer, as chat_bank.py (no clean-up pass)

    def dec(ids):
        return raw.decode([int(t) for t in ids], skip_special_tokens=False)

    special = set(tok.all_special_ids) | {
        i for t, i in tok.get_vocab().items() if t.startswith("<|") and t.endswith("|>")
    }
    wsb_items = json.loads((WSB / "items.json").read_text())["items"]
    wsb_rows = {r["id"]: r for r in json.loads((WSB / "capture_rows.json").read_text())}
    items, rows = [], []

    def add(label, source, source_id, ids, pl, subset, wsb_sites=None):
        sites16 = item_sites(dec, ids, pl, MAX_SITES_X3)
        if not sites16:
            return False
        sites8 = item_sites(dec, ids, pl, MAX_SITES_WSB)
        keep8 = {s["pos"] for s in sites8}
        if not keep8 <= {s["pos"] for s in sites16}:
            raise AssertionError(f"{label}: 8-thin not a subset of the 16-thin")
        for s in sites16:
            if ids[s["pos"]] in special:
                raise AssertionError(f"{label}: special token at site {s}")
        if max(s["pos"] for s in sites16) >= MAX_POSITIONS:
            raise AssertionError(f"{label}: a site beyond the capture's max_positions")
        if wsb_sites is not None and sites8 != wsb_sites:
            raise AssertionError(f"{label}: WSB sites not reproduced")
        prompt = user_turn(dec(ids[:pl]))
        items.append(
            {
                "id": label,
                "stratum": "chat",
                "source": source,
                "prompt": prompt,
                "response": dec(ids[pl:]),
                "response_start": pl,
                "subset": subset,
                "seed_key": seed_key("chat", source, prompt),
                "sites": [{**s, "wsb": s["pos"] in keep8} for s in sites16],
            }
        )
        rows.append(
            {
                "label": label,
                "id": label,
                "family": "hallucination_chat",
                "source": source,
                "source_id": source_id,
                "subset": subset,
                "user": prompt,
                "input_ids": [int(t) for t in ids],
                "prompt_len": pl,
                "eval_positions": [s["pos"] for s in sites16],
                "targets": [],
                "look_for": "",
                "render": "plain",
            }
        )
        return True

    for it in wsb_items:
        r = wsb_rows[it["id"]]
        ok = add(
            it["id"],
            it["source"],
            r["source_id"],
            r["input_ids"],
            r["prompt_len"],
            "wsb149",
            wsb_sites=it["sites"],
        )
        assert ok
        got = items[-1]
        for k in ITEM_KEYS:
            if got[k] != it[k]:
                raise AssertionError(f"{it['id']}: {k} differs from the WSB copy")

    dropped = []
    new = [r for f in rollout_files for r in json.loads(f.read_text())["rows"]]
    for r in sorted(new, key=lambda r: r["label"]):
        if not add(
            r["label"], r["source"], r["source_id"], r["input_ids"], r["prompt_len"], "x3new"
        ):
            dropped.append(r["label"])
        elif items[-1]["seed_key"] != r["seed_key"]:
            raise AssertionError(f"{r['label']}: decoded prompt's seed_key != the rollout's")

    ids_seen = Counter(i["id"] for i in items)
    keys_seen = Counter(i["seed_key"] for i in items)
    assert max(ids_seen.values()) == 1 and max(keys_seen.values()) == 1, "duplicate id / seed"
    return {"items": items, "rows": rows, "dropped_no_sites": dropped}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--rollouts", required=True, help="comma-separated gen_modal.py outputs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.6-27B")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    files = [Path(p) for p in args.rollouts.split(",") if p]
    b = build(files, tok)
    items, rows = b["items"], b["rows"]
    gen = [json.loads(f.read_text()) for f in files]
    by_subset = Counter(i["subset"] for i in items)
    by_source = Counter((i["subset"], i["source"]) for i in items)
    meta = {
        "family": "hallucination_chat",
        "plan": "plans/0028-main-bank-expansion.md",
        "model": "Qwen/Qwen3.6-27B",
        "subsets": {
            "wsb149": "the WSB bank (wsbench-pub @ 92d763e), item fields byte-identical; "
            "sites with wsb=true are exactly its 1,123 sites",
            "x3new": "plan 0028 draw (scripts/main_x3/build_seeds.py), on-policy rollouts "
            "(scripts/main_x3/gen_modal.py)",
        },
        "read_sites": "expand.response_sites over per-token decodes (no eos site), thinned to "
        "16 per item (chat_bank.thin); response and char from whole-span decodes",
        "generation": [g.get("settings") for g in gen],
        "generation_versions": [g.get("versions") for g in gen],
        "n_items": len(items),
        "n_sites": sum(len(i["sites"]) for i in items),
        "n_items_by_subset": dict(by_subset),
        "n_items_by_subset_source": {f"{a}/{s}": n for (a, s), n in sorted(by_source.items())},
        "n_sites_wsb": sum(s["wsb"] for i in items for s in i["sites"]),
        "dropped_no_sites": b["dropped_no_sites"],
        "empty_rollouts": [k for g in gen for k in g.get("empty", [])],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "capture_rows.json").write_text(json.dumps(rows))
    wsb_pos = {
        i["id"]: {s["pos"] for s in i["sites"] if s["wsb"]}
        for i in items
        if i["subset"] == "wsb149"
    }
    new_rows = []
    for r in rows:
        keep = [p for p in r["eval_positions"] if p not in wsb_pos.get(r["label"], set())]
        if keep:
            new_rows.append({**r, "eval_positions": keep})
    # read batches (plan 0028): tag a = the WSB items' new sites + new dailydialog items,
    # tag b = new lmsys items. A batch file that already exists must be reproduced exactly
    # (its cells may already be read); a missing one is written.
    batches = {
        "a": [r for r in new_rows if r["subset"] == "wsb149" or r["source"] == "dailydialog"],
        "b": [r for r in new_rows if r["subset"] == "x3new" and r["source"] == "lmsys"],
    }
    assert sum(len(v) for v in batches.values()) == len(new_rows)
    for tag, brows in batches.items():
        f = args.out / f"capture_new_{tag}.json"
        if f.exists():
            if json.loads(f.read_text()) != brows:
                raise SystemExit(f"{f} differs from this build: its cells may be read already")
        elif brows:
            f.write_text(json.dumps(brows))
        meta[f"n_cells_new_per_layer_{tag}"] = sum(len(r["eval_positions"]) for r in brows)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "items.json").write_text(
        json.dumps({"meta": meta, "items": items}, ensure_ascii=False, indent=1)
    )
    keys = [k for k in meta if k.startswith("n_") or k == "dropped_no_sites"]
    print(json.dumps({k: meta[k] for k in keys}, indent=1))


if __name__ == "__main__":
    main()
