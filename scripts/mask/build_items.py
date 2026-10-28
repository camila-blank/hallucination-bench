"""Build ``data/mask/items.json`` and the generation requests (plan 0017, stage G inputs).

Downloads the five binary MASK archetypes from HF ``cais/MASK`` at the pinned revision (parquet;
cached under ``$HBENCH_MASK_SRC``, default ``~/.cache/hb-mask/<revision>``, sha256 recorded),
draws a seeded sample of ``--per-archetype`` items per archetype (``random.Random(17)`` over
rows sorted by ``task_id``) and writes:

- ``data/mask/items.json``: ``{meta, items: [{task_id, archetype, system_prompt, user_prompt,
  dd_turn_1_prompt?, dd_turn_1_response?, proposition, option_a, option_b, ground_truth,
  formatted_ground_truth, belief_elicit_1..3?}]}``;
- ``outputs/mask/gen/requests.jsonl``: one request per (item, kind) for ``gen_modal.py``:
  ``{id, messages, n, temperature, top_p, max_tokens}`` with ids ``{task_id}:press`` (n = 3)
  and ``{task_id}:be{k}`` (be1 n = 3, be2 / be3 n = 1; none for ``provided_facts``);
  ``requests_x.jsonl``: ``{task_id}:press_x`` (n = 7), the other 7 of the 10 pressure samples.

    uv run --no-sync --with pyarrow python scripts/mask/build_items.py [--per-archetype 90] [--full]

``--full`` (plan rev 5) keeps doubling-down's seeded 90 and takes every row of the other four
archetypes; requests are still written for every item (the generator is seeded per request id,
and ``gen_modal.py`` callers filter to the ids not yet generated).
"""

import argparse
import hashlib
import json
import os
import random
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from hallucination_bench.evals.mask import source as S  # noqa: E402

OUT = REPO / "data" / "mask" / "items.json"
REQ = REPO / "outputs" / "mask" / "gen" / "requests.jsonl"
REQ_X = REQ.with_name("requests_x.jsonl")  # plan rev 2: 7 more pressure samples per item
SEED = 17
PRESS_N = 3
PRESS_X_N = 7
BELIEF_K = {1: 3, 2: 1, 3: 1}  # MASK's belief_k for the binary archetypes
SAMPLING = {"temperature": 1.0, "top_p": 1.0, "max_tokens": 1000}  # MASK defaults


def src_dir() -> Path:
    d = os.environ.get("HBENCH_MASK_SRC") or str(Path.home() / ".cache" / "hb-mask")
    return Path(d) / S.HF_REVISION


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fetch(archetype: str) -> Path:
    p = src_dir() / f"{archetype}.parquet"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        url = (
            f"https://huggingface.co/datasets/{S.HF_DATASET}/resolve/{S.HF_REVISION}/"
            f"{archetype}/test-00000-of-00001.parquet"
        )
        tmp = p.with_suffix(".tmp")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(p)
    return p


def item_of(archetype: str, row: dict) -> dict:
    opts = S.parse_proposition(row["proposition"])
    if opts is None:
        raise SystemExit(f"{row['task_id']}: unparseable binary proposition")
    if row["formatted_ground_truth"] not in ("A", "B"):
        raise SystemExit(f"{row['task_id']}: ground truth {row['formatted_ground_truth']!r}")
    it = {
        "task_id": row["task_id"],
        "archetype": archetype,
        "system_prompt": row["system_prompt"],
        "user_prompt": row["user_prompt"],
        "proposition": row["proposition"],
        "option_a": opts[0],
        "option_b": opts[1],
        "ground_truth": row["ground_truth"],
        "formatted_ground_truth": row["formatted_ground_truth"],
    }
    if archetype == "doubling_down_known_facts":
        it["dd_turn_1_prompt"] = row["dd_turn_1_prompt"]
        it["dd_turn_1_response"] = row["dd_turn_1_response"]
    for k in BELIEF_K:
        v = row.get(f"belief_elicit_{k}")
        if v:
            it[f"belief_elicit_{k}"] = v
    return it


def pressure_messages(it: dict) -> list[dict]:
    msgs = [{"role": "system", "content": it["system_prompt"]}]
    if it["archetype"] == "doubling_down_known_facts":
        msgs += [
            {"role": "user", "content": it["dd_turn_1_prompt"]},
            {"role": "assistant", "content": it["dd_turn_1_response"]},
        ]
    return [*msgs, {"role": "user", "content": it["user_prompt"]}]


def requests_of(it: dict) -> list[dict]:
    out = [{"id": f"{it['task_id']}:press", "messages": pressure_messages(it), "n": PRESS_N}]
    for k, n in BELIEF_K.items():
        if f"belief_elicit_{k}" in it:
            out.append(
                {
                    "id": f"{it['task_id']}:be{k}",
                    "messages": [
                        {"role": "system", "content": S.BELIEF_SYSTEM},
                        {"role": "user", "content": it[f"belief_elicit_{k}"]},
                    ],
                    "n": n,
                }
            )
    return [{**r, **SAMPLING} for r in out]


def main() -> None:
    import pyarrow.parquet as pq

    ap = argparse.ArgumentParser()
    ap.add_argument("--per-archetype", type=int, default=90)
    ap.add_argument(
        "--full",
        action="store_true",
        help="rev 5: every row of the four read archetypes (doubling-down keeps its sample)",
    )
    args = ap.parse_args()
    items: list[dict] = []
    files: dict[str, str] = {}
    counts: dict[str, int] = {}
    for arch in S.BINARY_ARCHETYPES:
        p = fetch(arch)
        files[arch] = sha256(p)
        rows = sorted(pq.read_table(p).to_pylist(), key=lambda r: r["task_id"])
        if len({r["task_id"] for r in rows}) != len(rows):
            raise SystemExit(f"{arch}: duplicate task_id")
        counts[arch] = len(rows)
        rng = random.Random(f"{SEED}:{arch}")
        pick = sorted(rng.sample(range(len(rows)), min(args.per_archetype, len(rows))))
        if args.full and arch != "doubling_down_known_facts":
            pick = list(range(len(rows)))
        items += [item_of(arch, rows[i]) for i in pick]
    ids = [it["task_id"] for it in items]
    if len(set(ids)) != len(ids):
        raise SystemExit("task_id collides across archetypes")
    meta = {
        "plan": "0017",
        "hf_dataset": S.HF_DATASET,
        "hf_revision": S.HF_REVISION,
        "mask_repo": S.MASK_REPO,
        "mask_commit": S.MASK_COMMIT,
        "parquet_sha256": files,
        "archetype_rows": counts,
        "per_archetype": args.per_archetype,
        "full": args.full,
        "seed": f"random.Random('{SEED}:<archetype>') over rows sorted by task_id",
        "sampling": SAMPLING,
        "press_n": PRESS_N,
        "belief_k": BELIEF_K,
        "n_items": len(items),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"meta": meta, "items": items}, indent=1, ensure_ascii=False) + "\n")
    REQ.parent.mkdir(parents=True, exist_ok=True)
    with REQ.open("w") as fh:
        for it in items:
            for r in requests_of(it):
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with REQ_X.open("w") as fh:
        for it in items:
            r = requests_of(it)[0]
            fh.write(
                json.dumps({**r, "id": r["id"] + "_x", "n": PRESS_X_N}, ensure_ascii=False) + "\n"
            )
    print(f"{len(items)} items -> {OUT}; requests -> {REQ}")


if __name__ == "__main__":
    main()
