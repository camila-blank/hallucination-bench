"""Build ``data/story/`` (plan 0007) from WorkspaceBench's frozen ``hallucination_creative`` bank.

Reads ``items.json`` and ``capture_rows.json`` from the global-workspace creative bank, thins
each item to 8 read sites with the old judge's rule (``judge.py``: ``step = len / 8``,
``sites[int(i * step)]``), and writes:

  data/story/items.json          {meta, items: [{id, stratum, source, prompt, response,
                                 response_start, sites: [{pos, token, kind, char}]}]}
  data/story/capture_rows.json   the source capture rows, ``eval_positions`` = the thinned sites

Deterministic; re-run to rebuild. Fails loudly on any check in plan 0007 §Bank.

    uv run --no-sync python scripts/build_story_bank.py [--src DIR]
"""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "story"
SRC_REL = "evals/workspace-bench/hillclimbing_evals/hallucination/creative"
# a global-workspace checkout (same variable as hallucination_bench.paths.GLOBAL_WORKSPACE)
DEFAULT_SRC = (
    Path(os.environ.get("HBENCH_GLOBAL_WORKSPACE") or "/workspace/global-workspace") / SRC_REL
)
MAX_SITES = 8
N_ITEMS = 20
SITE_KEYS = ("pos", "token", "kind", "char")


def thin(sites: list[dict[str, Any]], max_sites: int = MAX_SITES) -> list[dict[str, Any]]:
    """WorkspaceBench ``judge.py``'s site thinning (also ``chat_bank.thin``)."""
    if max_sites and len(sites) > max_sites:
        step = len(sites) / max_sites
        return [sites[int(i * step)] for i in range(max_sites)]
    return sites


def token_text(tok: str) -> str:
    return tok.replace("Ġ", " ").replace("Ċ", "\n")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_head(d: Path) -> str:
    out = subprocess.run(
        ["git", "-C", str(d), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def git_dirty(d: Path) -> str:
    """``git status --porcelain`` of the source bank dir (empty = files match HEAD)."""
    out = subprocess.run(
        ["git", "-C", str(d), "status", "--porcelain", "--", "."],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def build(src: Path, *, allow_dirty: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dirty = git_dirty(src)
    if dirty and not allow_dirty:
        raise SystemExit(f"source bank has uncommitted changes (source_commit would lie):\n{dirty}")
    bank = json.loads((src / "items.json").read_text(encoding="utf-8"))
    rows = json.loads((src / "capture_rows.json").read_text(encoding="utf-8"))
    rows_by_label = {r["label"]: r for r in rows}
    if len(bank["items"]) != N_ITEMS or len(rows) != N_ITEMS:
        raise SystemExit(f"expected {N_ITEMS} items/rows, got {len(bank['items'])}/{len(rows)}")

    items: list[dict[str, Any]] = []
    out_rows: list[dict[str, Any]] = []
    for it in bank["items"]:
        row = rows_by_label[it["id"]]
        sites = [{k: s[k] for k in SITE_KEYS} for s in thin(it["sites"])]
        if len(sites) != MAX_SITES:
            raise SystemExit(f"{it['id']}: {len(sites)} sites after thinning, want {MAX_SITES}")
        src_pos = set(row["eval_positions"])
        for s in sites:
            if s["pos"] not in src_pos:
                raise SystemExit(f"{it['id']}: site pos {s['pos']} not in eval_positions")
            if s["kind"] == "eos":
                raise SystemExit(f"{it['id']}: kept an eos site at pos {s['pos']}")
            if not it["response"][: s["char"]].endswith(token_text(s["token"])):
                raise SystemExit(f"{it['id']}: char {s['char']} does not end with {s['token']!r}")
        items.append(
            {
                "id": it["id"],
                "stratum": "story",
                "source": "hallucination_creative",
                "prompt": it["prompt"],
                "response": it["response"],
                "response_start": it["response_start"],
                "sites": sites,
            }
        )
        out_rows.append({**row, "eval_positions": sorted(s["pos"] for s in sites)})

    meta = {
        "eval": "story",
        "plan": "plans/0007-story-eval.md",
        "source_repo": "camilablank/global-workspace",
        "source_commit": git_head(src),
        "source_path": SRC_REL,
        "source_sha256": {
            "items.json": sha256(src / "items.json"),
            "capture_rows.json": sha256(src / "capture_rows.json"),
        },
        "model": bank["model"],
        "thinning": f"judge.py rule: step = len(sites) / {MAX_SITES}, sites[int(i * step)]",
        "n_items": len(items),
        "n_sites": sum(len(i["sites"]) for i in items),
        "note": "responses were capped at ~512 tokens; most stories stop mid-plot",
    }
    return {"meta": meta, "items": items}, out_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--allow-dirty", action="store_true", help="build from uncommitted source")
    args = ap.parse_args()
    bank, rows = build(args.src, allow_dirty=args.allow_dirty)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "items.json").write_text(
        json.dumps(bank, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    (OUT / "capture_rows.json").write_text(json.dumps(rows) + "\n", encoding="utf-8")
    print(
        f"wrote {OUT}/items.json ({bank['meta']['n_items']} items, "
        f"{bank['meta']['n_sites']} sites) and capture_rows.json"
    )


if __name__ == "__main__":
    main()
