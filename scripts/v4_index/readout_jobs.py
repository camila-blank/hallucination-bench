"""V4 chat + story readouts at the AM layers 21/27/33 (+ the 3 regenerated chat items at every
layer) through the V4 producers' jobs mode, merged into HB readouts roots (plan 0014).

The bench producers (``wsbench_v4_modal.py`` / ``wsbench_v4_jlens_modal.py``) read the
hallucination / hallucination_story families as ``captured`` renders (capture_rows input_ids,
read_positions) at the bench grid's V4 layers 15/24/30/36/39 only. Their jobs mode
(``fanout_jobs``) runs the same recipe (B=1 adapter-disabled block-output capture; O-lens: AO
carrier + slot injection, 4 samples, T 1, top-p 0.95, top-k 64, 256 new; J-lens: stream-mean,
cosine top-10) on explicit (ids, positions, layers) jobs. Rows are converted back to the bench
readouts contract (O-lens: the 4 samples joined by ``join_samples``; J-lens: tokens + scores).

    jobs   -> runs/hb_v4_fix/jobs_{a,b}.json on the v4-flash volume (all arms read the same
              files; part a = everything but the regenerated chat items, launchable before
              they exist; part b = the regenerated items, every layer)
    merge  -> fetch each arm's shards, check the grid, write <dest>/{chat,story}_readouts/<arm>/
              hallucination.jsonl = existing readouts (minus the regenerated chat items) + new rows

A few unchanged chat items are also read at layer 15 as ``VAL:<id>`` jobs: the J-lens is
deterministic, so ``merge`` checks those rows match the bench producer's (same tokens, scores to
1e-3), i.e. the jobs path and the contract conversion reproduce the existing readouts; they are
not merged.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

JOBS = "runs/hb_v4_fix/jobs_{part}.json"  # a: all but the regenerated items, b: those
CHAT_BENCH = Path("/workspace/root/wsbench-pub-v4-opus")
STORY_BENCH = Path("/workspace/root/wsbench-pub-v4-story")
OLD_CHAT = CHAT_BENCH / "outputs" / "readouts"
OLD_STORY = Path("/workspace/hb-v4/outputs/v4/story_readouts")
NEW_LAYERS = [21, 27, 33]
GRID = [15, 24, 30, 36, 39]
REGEN = ("chat-lmsys-0000", "chat-lmsys-0001", "chat-lmsys-0002")
VAL = ("chat-dailydialog-0000", "chat-lmsys-0010", "chat-dailydialog-0040")
JLENS = "v4-jlens-std-n1000"
ARMS = ("v4-s3don-sft", "v4-s3don-rl600", JLENS)


def capture_rows(bench: Path, family: str) -> list[dict[str, Any]]:
    p = bench / "evals" / family / "capture_rows.json"
    rows: list[dict[str, Any]] = json.loads(p.read_text(encoding="utf-8"))
    return rows


def want_grid() -> dict[str, set[tuple[str, int, int]]]:
    """Family -> the (id, layer, pos) cells the merged readouts must hold exactly."""
    out: dict[str, set[tuple[str, int, int]]] = {}
    for fam, bench, f in (
        ("chat", CHAT_BENCH, "hallucination"),
        ("story", STORY_BENCH, "hallucination_story"),
    ):
        out[fam] = {
            (r["id"], li, p)
            for r in capture_rows(bench, f)
            for li in sorted(GRID + NEW_LAYERS)
            for p in r["read_positions"]
        }
    return out


def jobs(a: argparse.Namespace) -> None:
    js = []
    chat = capture_rows(CHAT_BENCH, "hallucination")
    for r in chat:
        if (r["id"] in REGEN) != (a.part == "b"):
            continue
        lays = sorted(GRID + NEW_LAYERS) if r["id"] in REGEN else NEW_LAYERS
        js.append({"id": r["id"], "ids": r["input_ids"], "layers": lays,
                   "positions": [["site", p] for p in r["read_positions"]]})  # fmt: skip
    for r in capture_rows(STORY_BENCH, "hallucination_story") if a.part == "a" else []:
        js.append({"id": r["id"], "ids": r["input_ids"], "layers": NEW_LAYERS,
                   "positions": [["site", p] for p in r["read_positions"]]})  # fmt: skip
    for r in chat if a.part == "a" else []:
        if r["id"] in VAL:
            js.append({"id": f"VAL:{r['id']}", "ids": r["input_ids"], "layers": [15],
                       "positions": [["site", p] for p in r["read_positions"]]})  # fmt: skip
    assert len({j["id"] for j in js}) == len(js)
    n = sum(len(j["positions"]) * len(j["layers"]) for j in js)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(js, fh)
    subprocess.run(
        ["modal", "volume", "put", "-f", "v4-flash", fh.name, JOBS.format(part=a.part)], check=True
    )
    print(f"{len(js)} jobs, {n} cells per arm -> v4-flash:{JOBS.format(part=a.part)}")


def join_samples(outs: list[str]) -> str:  # wsbench_v4_modal.join_samples, verbatim rule
    if len(outs) == 1:
        return outs[0]
    if not any(outs):
        return ""
    return "\n".join(f"- {t}" if t else "-" for t in outs)


def contract(row: dict[str, Any]) -> dict[str, Any]:
    out = {k: row[k] for k in ("id", "layer", "pos", "token")}
    if "samples" in row:
        out["samples"] = [join_samples(row["samples"])]
    else:
        out |= {"tokens": row["tokens"], "scores": row["scores"]}
    return out


def dumps(r: dict[str, Any]) -> str:
    # HB splits lines with str.splitlines(), which also breaks at raw U+0085/2028/2029
    s = json.dumps(r, ensure_ascii=False)
    return s.replace("\u0085", "\\u0085").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def fetch(arm: str, tmp: Path) -> list[dict[str, Any]]:
    prefix = f"newfam_out/{arm}"
    ls = subprocess.run(
        ["modal", "volume", "ls", "v4-flash", prefix], capture_output=True, text=True, check=True
    ).stdout.split()
    rows = []
    names = []
    for part in ("a", "b"):
        nm = sorted({Path(x).name for x in ls if Path(x).name.startswith(f"jobs_{part}.")})
        ns = {x.split(".")[1].split("of")[1] for x in nm}
        assert len(ns) == 1, f"{arm}: part {part} shards of several n_shards (or none): {nm}"
        assert len(nm) == int(ns.pop()), f"{arm}: part {part}: {nm}"
        names += nm
    for nm in names:
        dst = tmp / f"{arm}.{nm}"
        subprocess.run(
            ["modal", "volume", "get", "--force", "v4-flash", f"{prefix}/{nm}", str(dst)],
            check=True,
            capture_output=True,
        )
        rows += [json.loads(x) for x in dst.read_text(encoding="utf-8").splitlines() if x]
    return rows


def merge(a: argparse.Namespace) -> None:
    want = want_grid()
    dest = Path(a.dest)
    with tempfile.TemporaryDirectory() as td:
        for arm in a.arms:
            rows = fetch(arm, Path(td))
            val = [r for r in rows if r["id"].startswith("VAL:")]
            new = [contract(r) for r in rows if not r["id"].startswith("VAL:")]
            if arm == JLENS:  # deterministic: the jobs path must reproduce the bench rows
                old = {
                    (r["id"], r["layer"], r["pos"]): r
                    for r in map(json.loads, open(OLD_CHAT / arm / "hallucination.jsonl"))
                }
                # same tokens; scores to 1e-3 (the score matmul's batch differs: last-digit
                # rounding flips, observed max 1e-4)
                bad = [
                    r for r in map(contract, val)
                    if (o := old[(r["id"][4:], r["layer"], r["pos"])])["tokens"] != r["tokens"]
                    or o["token"] != r["token"]
                    or max(abs(x - y) for x, y in zip(o["scores"], r["scores"], strict=True)) > 1e-3
                ]  # fmt: skip
                print(f"{arm}: VAL {len(val)} rows, {len(bad)} differ from the bench producer")
                assert val and not bad, bad[:2]
            for fam, old_p in (
                ("chat", OLD_CHAT / arm / "hallucination.jsonl"),
                ("story", OLD_STORY / arm / "hallucination.jsonl"),
            ):
                ids = {k[0] for k in want[fam]}
                keep = [
                    json.loads(x)
                    for x in old_p.read_text(encoding="utf-8").splitlines()
                    if x and not (fam == "chat" and json.loads(x)["id"] in REGEN)
                ]
                merged = keep + [r for r in new if r["id"] in ids]
                keys = [(r["id"], r["layer"], r["pos"]) for r in merged]
                have = set(keys)
                miss, off = want[fam] - have, have - want[fam]
                n_empty = sum(
                    not (r.get("samples") or [""])[0] and not r.get("tokens") for r in merged
                )
                print(f"{arm}/{fam}: {len(keep)} kept + {len(merged) - len(keep)} new; missing "
                      f"{len(miss)}, off-grid {len(off)}, dup {len(keys) - len(have)}, "
                      f"empty {n_empty}")  # fmt: skip
                assert not (miss or off) and len(keys) == len(have), (list(miss)[:3], list(off)[:3])
                out = dest / f"{fam}_readouts" / arm / "hallucination.jsonl"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text("".join(dumps(r) + "\n" for r in merged), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=["jobs", "merge"])
    ap.add_argument("--dest", default="/workspace/hb-v4/outputs/v4r")
    ap.add_argument("--arms", type=lambda s: s.split(","), default=list(ARMS))
    ap.add_argument("--part", choices=["a", "b"], default="a")
    a = ap.parse_args()
    {"jobs": jobs, "merge": merge}[a.mode](a)
