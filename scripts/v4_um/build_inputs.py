"""V4 UM ladder inputs (local, no API / GPU).

items      Set A items of the 3 domains -> outputs/v4/um/modal_in/items_in.json (Modal ``items``)
           + V4 window check of every item
src        V4 src root outputs/v4/um/src/: items.json with the V4 greedy responses (from
           outputs/v4/um/modal_out/responses.jsonl), probe_texts.jsonl = plan 0003 texts + the
           Qwen UM generated texts (ids prefixed ``q``), readouts split per domain from
           outputs/v4/um/modal_out/readouts/<arm>/
probe-job  V4-window-checked probe texts of every domain -> modal_in/probe_<tag>.json
           (``--only-new`` = texts not captured yet, per the pack metas in modal_out/pack/)

  .venv/bin/python scripts/v4_um/build_inputs.py items
"""

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
V4UM = HB / "outputs" / "v4" / "um"
MIN = V4UM / "modal_in"
MOUT = V4UM / "modal_out"
SRC = V4UM / "src"
DOMAINS = ("country", "capital_city", "holiday")
DATA = HB / "data" / "sibling_probe"
ARMS = ("v4-s3don-sft", "v4-s3don-rl600", "v4-jlens-std-n1000")
METHOD_OF = {
    "v4-s3don-sft": "v4-s3don-sft",
    "v4-s3don-rl600": "v4-s3don-rl600",
    "v4-jlens-std-n1000": "jlens",
}


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def set_a() -> list[dict]:
    bank = json.loads((DATA / "items.json").read_text())
    return [it for it in bank["items"] if it["category"] in DOMAINS]


def cmd_items(_a: argparse.Namespace) -> None:
    import v4tok

    v4 = v4tok.get()
    its = set_a()
    bad = [it["id"] for it in its if not v4.window_ok(it["messages"], it["request"])]
    n_win = Counter(len(v4.window(it["messages"])[0]) for it in its)
    MIN.mkdir(parents=True, exist_ok=True)
    rows = [
        {"id": it["id"], "family": f"sibling_{it['category']}", "messages": it["messages"]}
        for it in its
    ]
    (MIN / "items_in.json").write_text(json.dumps(rows, ensure_ascii=False))
    print(
        f"{len(rows)} items {Counter(it['category'] for it in its)}; window lengths {dict(n_win)};"
        f" V4 window mismatches {bad}"
    )


def cmd_pool(_a: argparse.Namespace) -> None:
    """src/data/sibling_probe/probe_texts.jsonl: plan 0003 + Qwen UM texts passing the V4 window."""
    bank = json.loads((DATA / "items.json").read_text())
    its = [it for it in bank["items"] if it["category"] in DOMAINS]
    d = SRC / "data" / "sibling_probe"
    d.mkdir(parents=True, exist_ok=True)
    # probe-text pool: plan 0003 texts + the Qwen UM generated texts (re-checked by probe-texts)
    old = read_jsonl(DATA / "probe_texts.jsonl")
    ids = {t["id"] for t in old}
    extra = []
    for dom in DOMAINS:
        for t in read_jsonl(DATA / f"um_{dom}_probe_texts.jsonl"):
            if t.get("source") == "umgen":
                t = {k: v for k, v in t.items() if k != "source"}
                t["id"] = "q" + t["id"]
                assert t["id"] not in ids
                ids.add(t["id"])
                extra.append(t)
    # V4 window check (plan 0003's check was on the Qwen template); other categories are dropped
    import v4tok

    v4 = v4tok.get()
    req = {(it["category"], int(it["request_idx"])): it["request"] for it in its}
    keep, drop = [], Counter()
    for t in [*old, *extra]:
        if t["category"] not in DOMAINS:
            continue
        if v4.window_ok(t["messages"], req[(t["category"], int(t["request_idx"]))]):
            keep.append(t)
        else:
            drop[t["category"]] += 1
    with (d / "probe_texts.jsonl").open("w") as fh:
        for t in keep:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(
        f"probe-text pool {len(old)} plan 0003 + {len(extra)} Qwen UM -> "
        f"{len(keep)} in the 3 domains passing the V4 window check (dropped {dict(drop)})"
    )


def cmd_src(_a: argparse.Namespace) -> None:
    resp = {r["id"]: r["response"] for r in read_jsonl(MOUT / "responses.jsonl")}
    bank = json.loads((DATA / "items.json").read_text())
    its = []
    for it in bank["items"]:
        if it["category"] not in DOMAINS:
            continue
        if it["id"] not in resp:
            sys.exit(f"no V4 response for {it['id']}")
        its.append({**it, "response": resp[it["id"]], "response_qwen": it.get("response")})
    d = SRC / "data" / "sibling_probe"
    d.mkdir(parents=True, exist_ok=True)
    (d / "items.json").write_text(
        json.dumps(
            {
                **{k: v for k, v in bank.items() if k != "items"},
                "items": its,
                "v4_note": (
                    "responses = DeepSeek-V4-Flash greedy (chat, no thinking, 256 new tokens)"
                ),
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    cmd_pool(_a)
    # readouts per domain
    cat = {it["id"]: it["category"] for it in its}
    for arm in ARMS:
        files = sorted(glob.glob(str(MOUT / "readouts" / arm / "*.jsonl")))
        if not files:
            print(f"  {arm}: no readouts yet")
            continue
        by = defaultdict(list)
        for f in files:
            for r in read_jsonl(Path(f)):
                by[cat[r["id"]]].append(r)
        od = SRC / "outputs" / "sibling" / "readouts" / arm
        od.mkdir(parents=True, exist_ok=True)
        for dom, rows in by.items():
            with (od / f"sibling_{dom}.jsonl").open("w") as fh:
                for r in sorted(rows, key=lambda r: (r["id"], r["layer"], r["pos"])):
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(
            f"  {arm}: {sum(len(v) for v in by.values())} cells "
            f"{ {k: len(v) for k, v in by.items()} } from {len(files)} files"
        )


def cmd_probe_job(a: argparse.Namespace) -> None:
    import v4tok

    v4tok.get()  # load the V4 tokenizer up front (fail fast)
    done: set[str] = set()
    if a.from_pool:
        pool = read_jsonl(SRC / "data" / "sibling_probe" / "probe_texts.jsonl")
    if a.only_new:
        for f in glob.glob(str(MOUT / "pack" / "*" / "*.json")):
            done |= set(json.loads(Path(f).read_text())["labels"])
    job, rep = {}, {}
    for dom in a.domains.split(","):
        sub = "" if dom == "country" else dom
        if a.from_pool:
            texts = [t for t in pool if t["category"] == dom]
        else:
            texts = read_jsonl(V4UM / sub / "data" / f"um_{dom}_probe_texts.jsonl")
        todo = [t for t in texts if t["id"] not in done]
        n = max(1, -(-len(todo) // a.per_shard)) if todo else 0
        job[dom] = [
            [{"id": t["id"], "messages": t["messages"]} for t in todo[s::n]] for s in range(n)
        ]
        rep[dom] = {"texts": len(texts), "todo": len(todo), "shards": n}
    MIN.mkdir(parents=True, exist_ok=True)
    out = MIN / f"probe_{a.tag}.json"
    out.write_text(json.dumps(job, ensure_ascii=False))
    print(f"{out}: {rep}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("items")
    sub.add_parser("src")
    sub.add_parser("pool")
    p = sub.add_parser("probe-job")
    p.add_argument("--domains", default=",".join(DOMAINS))
    p.add_argument("--tag", default="s")
    p.add_argument("--per-shard", type=int, default=1200)
    p.add_argument("--only-new", action="store_true")
    p.add_argument("--from-pool", action="store_true")
    a = ap.parse_args()
    {"items": cmd_items, "src": cmd_src, "pool": cmd_pool, "probe-job": cmd_probe_job}[a.cmd](a)


if __name__ == "__main__":
    main()
