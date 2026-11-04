"""Plan 0028 step 1: screened seeds for the 3x main bank (CPU; needs HF access).

Same sources, order, screen and pool rule as the WorkspaceBench chat bank (global-workspace
``scripts/oracle_lens_evals/jlens_pr/build_seeds.py``, seed 7, n_per_source 75):

- lmsys: first user turn of every ``lmsys/lmsys-chat-1m`` row whose conversation starts with a
  user turn, stripped, parquet shards in sorted filename order (``lmsys_stream.py``);
- dailydialog: ``ConvLab/dailydialog`` train, streaming, first turn if its speaker is ``user``;
- screen (``expand.screen_chat_seed``): 20..2000 chars stripped, no ``` / http(s):// / NAME_\\d+,
  >= 98% ASCII characters (the English proxy); dedup by ``seed_key``;
- pool = the first ``4 * n_per_source`` unique screened seeds of a source (scan cap 20000).

Always re-derives the WSB draw (n 75, ``random.Random(7)`` shared, lmsys shuffled
first) and checks it picks exactly the bank's seeds. The expansion pool is the first
``4 * 3 * 75 = 900`` unique screened seeds (the WSB pool of 300 is its prefix); the WSB seeds are
removed by key, the rest is shuffled with ``random.Random("hb0028:<source>")`` and the first
``--n-new`` (150) are kept, in that seeded order.

    python scripts/main_x3/build_seeds.py --sources dailydialog,lmsys --out <dir>/seeds.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import unicodedata
from collections.abc import Iterator
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from hallucination_bench.evals.wsbench_chat.expand import screen_chat_seed, seed_key  # noqa: E402

WSB_SEED = 7
WSB_N = 75
SCAN_CAP = 20000
SOURCES = ("lmsys", "dailydialog")  # build_seeds.py's order (the shared rng depends on it)


def iter_lmsys() -> Iterator[tuple[str, str]]:
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.dataset_info("lmsys/lmsys-chat-1m")
    names = sorted(s.rfilename for s in info.siblings if s.rfilename.endswith(".parquet"))
    i = 0
    for n in names:
        path = api.hf_hub_download(repo_id="lmsys/lmsys-chat-1m", filename=n, repo_type="dataset")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=["conversation"]):
            for conv in batch.column("conversation").to_pylist():
                if conv and conv[0].get("role") == "user":
                    t = (conv[0].get("content") or "").strip()
                    if t:
                        yield str(i), t
                i += 1


def iter_dailydialog() -> Iterator[tuple[str, str]]:
    from datasets import load_dataset

    for ex in load_dataset("ConvLab/dailydialog", split="train", streaming=True):
        turns = ex["turns"]
        if turns and turns[0]["speaker"] == "user":
            yield str(ex["dialogue_id"]), str(turns[0]["utterance"]).strip()


ITERS = {"lmsys": iter_lmsys, "dailydialog": iter_dailydialog}


def pool(source: str, size: int) -> tuple[list[dict], dict]:
    """The first ``size`` unique screened seeds, in source order (build_seeds.py's loop)."""
    c = {"scanned": 0, "screened_out": 0, "kept_in_pool": 0}
    out: list[dict] = []
    seen: set[str] = set()
    for sid, text in ITERS[source]():
        c["scanned"] += 1
        if c["scanned"] > SCAN_CAP:
            break
        if not screen_chat_seed(text):
            c["screened_out"] += 1
            continue
        key = seed_key("chat", source, text)
        if key in seen:
            continue
        seen.add(key)
        out.append({"key": key, "cell": "chat", "source": source, "source_id": sid, "text": text})
        if len(out) >= size:
            break
    c["kept_in_pool"] = len(out)
    return out, c


def wsb_keys(items_path: Path) -> dict[str, set[str]]:
    """The WSB bank's seed keys per source (prompt = the stripped first user turn)."""
    keys: dict[str, set[str]] = {s: set() for s in SOURCES}
    for it in json.loads(items_path.read_text())["items"]:
        keys[it["source"]].add(seed_key("chat", it["source"], it["prompt"]))
    return keys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--sources", default="lmsys,dailydialog")
    ap.add_argument("--n-new", type=int, default=150, help="new seeds per source")
    ap.add_argument("--multiple", type=int, default=3, help="bank size multiple (pool 4*m*75)")
    ap.add_argument("--wsb-items", type=Path, default=REPO / "data/wsbench_chat/items.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    sources = [s for s in SOURCES if s in args.sources.split(",")]
    wsb = wsb_keys(args.wsb_items)
    doc = {"seeds": [], "counts": {}, "verify_wsb": {}, "rule": __doc__.split("\n\n")[1]}
    if args.out.exists():
        doc = json.loads(args.out.read_text())  # add sources one at a time (e.g. lmsys later)
    for source in sources:
        big, c = pool(source, 4 * args.multiple * WSB_N)
        # WSB reproduction: the first 300 of the pool, rng Random(7) shared, lmsys first
        rng = random.Random(WSB_SEED)
        if source == "dailydialog":
            rng.shuffle(list(range(4 * WSB_N)))  # the lmsys pool's shuffle (it had 300 seeds)
        small = list(big[: 4 * WSB_N])
        rng.shuffle(small)
        picked = {s["key"] for s in small[:WSB_N]}
        # the bank holds 74 lmsys items (chat-lmsys-0013 had no sites): picked must cover them
        ok = wsb[source] <= picked and len(picked - wsb[source]) == WSB_WSB_DROPPED[source]
        doc["verify_wsb"][source] = {
            "reproduced": ok,
            "n_bank": len(wsb[source]),
            "n_picked": len(picked),
            "picked_not_in_bank": sorted(picked - wsb[source]),
            "bank_not_picked": len(wsb[source] - picked),
        }
        if not ok:
            raise SystemExit(f"{source}: WSB draw not reproduced: {doc['verify_wsb'][source]}")
        rest = [s for s in big if s["key"] not in picked]
        # the Qwen tokenizer NFC-normalises, so a non-NFC seed's decoded prompt (and its key)
        # would not round-trip: new seeds must be NFC (the WSB draw above is untouched)
        n_non_nfc = sum(unicodedata.normalize("NFC", s["text"]) != s["text"] for s in rest)
        rest = [s for s in rest if unicodedata.normalize("NFC", s["text"]) == s["text"]]
        random.Random(f"hb0028:{source}").shuffle(rest)
        new = rest[: args.n_new]
        if len(new) < args.n_new:
            raise SystemExit(f"{source}: only {len(new)} new seeds")
        doc["seeds"] = [s for s in doc["seeds"] if s["source"] != source] + new
        doc["counts"][source] = {
            **c,
            "wsb_excluded": len(picked),
            "non_nfc_excluded": n_non_nfc,
            "new": len(new),
        }
        print(f"[seeds] {source}: {doc['counts'][source]} verify {doc['verify_wsb'][source]}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    print(f"[seeds] wrote {len(doc['seeds'])} -> {args.out}")


# WSB seeds rolled out but absent from the bank (no read sites): chat-lmsys-0013
WSB_WSB_DROPPED = {"lmsys": 1, "dailydialog": 0}

if __name__ == "__main__":
    main()
