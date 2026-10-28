"""Plan 0018 rev 5: extra real-user texts for the belief gate (WildChat-1M first user turns).

SAD's deploy batch is only 199 sampled WildChat texts and Qwen calls ~7% of them deployment, so the
deploy-belief group needs a larger pool. This samples first user turns from two pinned
WildChat-1M parquet shards (seeded), keeps non-toxic, non-redacted turns of 20-3,000 characters
(SAD's length range), drops duplicates and any text equal to a SAD text, and writes

* ``data/sad/raw/wildchat.jsonl`` (gitignored, like the SAD texts: raw user conversations are not
  committed) -- ``{id, sad_id: null, provenance: "deploy", source: "wildchat_1m", text,
  conversation_hash, shard, language}``;
* ``data/sad/manifest_wildchat.json`` (committed) -- ids, conversation hash, sha256, length only.

LMSYS-Chat-1M is gated for this HF account (403), so it is not used.

    UV_CACHE_DIR=/root/.cache/uv-hbsad uv run --no-sync --with pyarrow \\
        python scripts/sad/fetch_wildchat.py
"""

import argparse
import json
import random
from collections import Counter

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from hallucination_bench.evals.sad import bank

REPO = "allenai/WildChat-1M"
REVISION = "7d6490e462285cf85d91eabea0f9a954fbddcd1f"
SHARDS = ("data/train-00000-of-00014.parquet", "data/train-00007-of-00014.parquet")
PER_SHARD = 1500
SEED = 0
MIN_CHARS, MAX_CHARS = 20, 3000


def first_user_turn(conv: list[dict]) -> dict | None:
    for t in conv or []:
        if t.get("role") == "user":
            return t
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-shard", type=int, default=PER_SHARD)
    args = ap.parse_args()
    sad_hashes = {it["sha256"] for it in bank.load_sad_manifest()["items"]}
    rng = random.Random(SEED)
    rows: list[dict] = []
    seen: set[str] = set()
    for shard in SHARDS:
        path = hf_hub_download(REPO, shard, repo_type="dataset", revision=REVISION)
        pf = pq.ParquetFile(path)
        cand = []
        for rg in range(pf.num_row_groups):  # stream row groups (32 GB RAM cap shared)
            t = pf.read_row_group(
                rg, columns=["conversation_hash", "conversation", "toxic", "redacted"]
            )
            for r in t.to_pylist():
                if r["toxic"] or r["redacted"]:
                    continue
                u = first_user_turn(r["conversation"])
                if u is None or u.get("toxic") or u.get("redacted"):
                    continue
                text = u.get("content") or ""
                if not MIN_CHARS <= len(text.strip()) <= MAX_CHARS:
                    continue
                cand.append((r["conversation_hash"], text, u.get("language")))
        cand.sort()
        rng.shuffle(cand)
        n = 0
        for h, text, lang in cand:
            sha = bank.sha256_text(text)
            if sha in seen or sha in sad_hashes:
                continue
            seen.add(sha)
            rows.append(
                {
                    "id": f"wc-{h[:12]}",
                    "sad_id": None,
                    "provenance": "deploy",
                    "source": "wildchat_1m",
                    "text": text,
                    "conversation_hash": h,
                    "shard": shard,
                    "language": lang,
                }
            )
            n += 1
            if n >= args.per_shard:
                break
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == len(ids), "wildchat short ids collide"
    out = bank.raw_dir() / "wildchat.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    man = {
        "meta": {
            "repo": REPO,
            "revision": REVISION,
            "shards": list(SHARDS),
            "per_shard": args.per_shard,
            "seed": SEED,
            "filters": "first user turn; not toxic / redacted; 20-3000 chars; dedup; != SAD texts",
            "n_items": len(rows),
            "by_language": dict(Counter(r["language"] for r in rows).most_common(15)),
            "note": "texts are NOT in git; data/sad/raw/wildchat.jsonl",
        },
        "items": [
            {
                "id": r["id"],
                "conversation_hash": r["conversation_hash"],
                "provenance": "deploy",
                "source": "wildchat_1m",
                "sha256": bank.sha256_text(r["text"]),
                "n_chars": len(r["text"]),
            }
            for r in rows
        ],
    }
    bank.MANIFEST_WILDCHAT.write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(rows)} texts, gitignored) and {bank.MANIFEST_WILDCHAT}")
    print(man["meta"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
