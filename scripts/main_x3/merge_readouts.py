"""Plan 0028: grid checks and the merge of old + new main readouts onto the 3x bank.

check-new  --rows capture_new_<TAG>.json --readouts <new>/<arm>/hallucination.jsonl --kind
           The converted new rows are exactly rows x 11 layers (no missing / off-grid / dup).
merge      --bank <x3>/items.json --tags a[,b]
           For each official main arm, writes $HBENCH_X3_READOUTS/<arm>/hallucination.jsonl =
           the old file's rows (the WSB cells, unchanged and in their order) + every tag's new
           rows, then checks the FULL x3 grid at the arm's layers. The one s3d read (reader
           ``s3d``) feeds both s3d arms: ``s3d-rl600`` at its old file's layers (20/36/44/48/60)
           and ``s3d-rl600-all11`` at all 11; methods.json uses ``s3d-rl600`` at its layers and
           ``-all11`` elsewhere, so a new readout enters the index once.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from hallucination_bench import paths  # noqa: E402

LAYERS = tuple(range(20, 61, 4))
# arm -> (reader dir under the new tag root, old file root)
ARMS = {
    "s3d-rl600": "s3d",
    "s3d-rl600-all11": "s3d",
    "mlnla-rl600-all11": "mlnla-rl600-all11",
    "jlens": "jlens",
}


def _rows(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(x) for x in fh if x.strip()]


def check(rows: list[dict], want: set[tuple[str, int, int]], label: str, kind: str) -> dict:
    keys = [(r["id"], int(r["layer"]), int(r["pos"])) for r in rows]
    have = set(keys)
    out = {
        "rows": len(rows),
        "on_grid": len(have & want),
        "expected": len(want),
        "missing": len(want - have),
        "off_grid": len(have - want),
        "dup": len(keys) - len(have),
    }
    if kind == "prose":
        out["samples_per_row"] = dict(
            collections.Counter(len(r.get("samples") or []) for r in rows)
        )
    else:
        out["tokens_per_row"] = dict(collections.Counter(len(r.get("tokens") or []) for r in rows))
    print(f"{label}: {out}")
    if out["missing"] or out["off_grid"] or out["dup"] or not rows:
        raise SystemExit(f"{label}: grid check failed")
    return out


def cmd_check_new(a: argparse.Namespace) -> None:
    rows = json.loads(Path(a.rows).read_text())
    want = {(r["label"], la, int(p)) for r in rows for la in LAYERS for p in r["eval_positions"]}
    check(_rows(Path(a.readouts)), want, str(a.readouts), a.kind)


IGNORE_CFG = {"layers", "acts_dir", "acts_subdir", "n_prompts", "scales", "positions"}


def _norm_cfg(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if k in IGNORE_CFG:
            continue
        out[k] = _norm_cfg(v) if isinstance(v, dict) else v
    return out


def cmd_check_config(a: argparse.Namespace) -> None:
    """The new read's run_config.json equals the old main read's (reader instrument drift)."""
    new = _norm_cfg(json.loads(Path(a.new).read_text()))
    ref = _norm_cfg(json.loads(Path(a.ref).read_text()))
    diff = {k: (ref.get(k), new.get(k)) for k in set(new) | set(ref) if new.get(k) != ref.get(k)}
    print(f"{a.new} vs {a.ref}: {'IDENTICAL' if not diff else diff}")
    if diff:
        raise SystemExit("reader config drift")


def cmd_merge(a: argparse.Namespace) -> None:
    items = json.loads(Path(a.bank).read_text())["items"]
    new_root = Path(a.new_root)
    report = {}
    for arm, reader in ARMS.items():
        if a.arms and arm not in a.arms.split(","):
            continue
        old = _rows(paths.EXTRA_READOUTS / arm / "hallucination.jsonl")
        layers = sorted({int(r["layer"]) for r in old})
        wsb = {(i["id"], int(s["pos"])) for i in items for s in i["sites"] if s["wsb"]}
        bad = [r for r in old if (r["id"], int(r["pos"])) not in wsb]
        if bad:
            raise SystemExit(f"{arm}: {len(bad)} old rows off the WSB sites")
        new = []
        for tag in a.tags.split(","):
            new += [
                r
                for r in _rows(new_root / tag / reader / "hallucination.jsonl")
                if int(r["layer"]) in layers
            ]
        kind = "tokens" if arm == "jlens" else "prose"
        # expected = the WSB cells + every tag's capture rows; complete once it is the whole bank
        cells = set(wsb)
        for tag in a.tags.split(","):
            for r in json.loads((Path(a.bank).parent / f"capture_new_{tag}.json").read_text()):
                cells |= {(r["label"], int(p)) for p in r["eval_positions"]}
        bank_cells = {(i["id"], int(s["pos"])) for i in items for s in i["sites"]}
        if not cells <= bank_cells:
            raise SystemExit(f"{arm}: capture rows off the bank's sites")
        want = {(i, la, p) for (i, p) in cells for la in layers}
        merged = old + new
        stats = check(merged, want, f"{arm} merged", kind)
        dst = paths.X3_READOUTS / arm / "hallucination.jsonl"
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("w") as fh:
            for r in merged:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        report[arm] = {
            **stats,
            "layers": layers,
            "complete": cells == bank_cells,
            "n_bank_sites_unread": len(bank_cells - cells),
            "n_old": len(old),
            "n_new": len(new),
            "new_reader": reader,
            "sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
            "path": str(dst),
        }
    out = paths.X3_READOUTS / "merge_report.json"
    prev = json.loads(out.read_text()) if out.exists() else {}
    out.write_text(json.dumps({**prev, **report, "tags": a.tags}, indent=1))
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check-new")
    c.add_argument("--rows", required=True)
    c.add_argument("--readouts", required=True)
    c.add_argument("--kind", choices=["prose", "tokens"], required=True)
    m = sub.add_parser("merge")
    m.add_argument("--bank", required=True)
    m.add_argument("--tags", required=True)
    m.add_argument("--arms", default="")
    m.add_argument("--new-root", default=str(paths.X3_NEW_READOUTS))
    k = sub.add_parser("check-config")
    k.add_argument("--new", required=True)
    k.add_argument("--ref", required=True)
    a = ap.parse_args()
    {"check-new": cmd_check_new, "merge": cmd_merge, "check-config": cmd_check_config}[a.cmd](a)


if __name__ == "__main__":
    main()
