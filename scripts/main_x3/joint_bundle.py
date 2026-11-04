"""Plan 0028: the joint all-11-layer readout bundle per (item, site), one JSONL per index method.

For each method (``s3d-rl600``, ``mlnla-rl600``, ``jlens``) the arm per layer follows
``data/index/methods.json`` (``main`` list, first arm holding the layer wins; a method without
an entry is its own arm). A record: ``{id, subset, source, pos, kind, char, token, wsb,
layers: {"20": {arm, samples} | {arm, tokens, scores, summary}, ...}}``. Prose readouts keep the
first ``HAL_K = 3`` non-empty samples (the claim map's rule); J-lens keeps the decoded top-10 and
the interp-v1 summary (k = 1; ``null`` until summarised). No judging.

    uv run --no-sync python scripts/main_x3/joint_bundle.py --bank <x3>/items.json \\
        --summaries <out>/wsbench_chat/claude-opus-5-5/jlens/cache.jsonl --dest <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from hallucination_bench import DATA, paths  # noqa: E402
from hallucination_bench.readouts import decode_token  # noqa: E402

HAL_K = 3
LAYERS = tuple(range(20, 61, 4))
METHODS = ("s3d-rl600", "mlnla-rl600", "jlens")


def method_arms(method: str) -> list[str]:
    m = json.loads((DATA / "index" / "methods.json").read_text())["methods"]
    return list(m.get(method, {}).get("main", [method]))


def load(arm: str) -> dict[tuple[str, int, int], dict]:
    out = {}
    with (paths.X3_READOUTS / arm / "hallucination.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            out[(r["id"], int(r["layer"]), int(r["pos"]))] = r
    return out


def summaries(path: Path | None) -> dict[str, str]:
    """Last valid interp-v1 ``summ:`` answer per cell key."""
    out: dict[str, str] = {}
    if path is None or not path.exists():
        return out
    with path.open() as fh:
        for line in fh:
            if '"summ:' not in line[:40]:
                continue
            r = json.loads(line)
            res = r.get("result")
            if r["key"].startswith("summ:") and isinstance(res, dict) and res.get("interpretation"):
                out[r["key"][5:]] = res["interpretation"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--summaries", type=Path, default=None)
    ap.add_argument("--dest", type=Path, required=True)
    a = ap.parse_args()
    items = json.loads(a.bank.read_text())["items"]
    summ = summaries(a.summaries)
    a.dest.mkdir(parents=True, exist_ok=True)
    index = {
        "bank": str(a.bank),
        "bank_sha256": hashlib.sha256(a.bank.read_bytes()).hexdigest(),
        "layers": list(LAYERS),
        "k_prose": HAL_K,
        "methods": {},
    }
    for method in METHODS:
        arms = method_arms(method)
        data = {arm: load(arm) for arm in arms}
        have_layers = {arm: {k[1] for k in d} for arm, d in data.items()}
        arm_of = {}
        for la in LAYERS:
            arm_of[la] = next((arm for arm in arms if la in have_layers[arm]), None)
            if arm_of[la] is None:
                raise SystemExit(f"{method}: no arm holds layer {la}")
        n = n_missing = n_unsummarised = 0
        dst = a.dest / f"{method}.jsonl"
        with dst.open("w") as fh:
            for it in items:
                for s in it["sites"]:
                    layers = {}
                    for la in LAYERS:
                        arm = arm_of[la]
                        r = data[arm].get((it["id"], la, int(s["pos"])))
                        if r is None:
                            n_missing += 1
                            layers[str(la)] = None
                            continue
                        if "tokens" in r:
                            key = f"{it['id']}__L{la:03d}__p{int(s['pos'])}"
                            sm = summ.get(key)
                            n_unsummarised += sm is None
                            layers[str(la)] = {
                                "arm": arm,
                                "tokens": [decode_token(t) for t in r["tokens"]],
                                "scores": r.get("scores"),
                                "summary": sm,
                            }
                        else:
                            smp = [x for x in r.get("samples") or [] if x.strip()][:HAL_K]
                            layers[str(la)] = {"arm": arm, "samples": smp}
                    rec = {k: it[k] for k in ("id", "subset", "source")}
                    rec |= {k: s[k] for k in ("pos", "kind", "char", "token", "wsb")}
                    rec["layers"] = layers
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
        index["methods"][method] = {
            "path": str(dst),
            "sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
            "arm_per_layer": {str(k): v for k, v in arm_of.items()},
            "n_sites": n,
            "n_missing_cells": n_missing,
            "n_unsummarised_cells": n_unsummarised if method == "jlens" else None,
            "readouts": {
                arm: hashlib.sha256(
                    (paths.X3_READOUTS / arm / "hallucination.jsonl").read_bytes()
                ).hexdigest()
                for arm in arms
            },
        }
        print(method, index["methods"][method]["n_sites"], n_missing, n_unsummarised)
    (a.dest / "index.json").write_text(json.dumps(index, indent=1))
    print(f"wrote {a.dest / 'index.json'}")


if __name__ == "__main__":
    main()
