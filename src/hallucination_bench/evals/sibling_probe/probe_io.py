"""Inputs of the Modal probe job and loaders for its outputs (plan 0003 §Phase E).

``hbench sibling probes --prepare`` writes ``outputs/sibling/probes/inputs.json``: the concept
order per category (the column order of every probe output), the item / probe-text metadata and
the Qwen token ids of every concept's token aliases (unembedding check). The Modal job
(``scripts/sibling_probe/probes_modal.py``) writes per-category pieces to ``<dir>/pieces/<recipe>/
<category>/``; :func:`merge_pieces` merges them into ``<dir>/<recipe>/`` (recipes ``v2`` =
shrinkage-whitened DiffMean, primary; ``v0`` = raw DiffMean, plan 0003 rev 11). A directory with
``validity.jsonl`` directly in it is a legacy single-recipe layout (rev 10, ``probes_v2_rawdm``).
"""

import gzip
import json
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from hallucination_bench import REPO_ROOT
from hallucination_bench.evals.sibling_probe import bank as B
from hallucination_bench.evals.sibling_probe.concepts import load_inventory

PROBES_DIR = REPO_ROOT / "outputs" / "sibling" / "probes"
TOKENIZER = "Qwen/Qwen3.6-27B"


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def token_ids(tokenizer: Any) -> dict[str, dict[str, list[int]]]:
    out: dict[str, dict[str, list[int]]] = {}
    for cat in load_inventory().values():
        for c in cat.concepts:
            ids = []
            for t in c.token_aliases:
                enc = tokenizer.encode(t, add_special_tokens=False)
                if len(enc) != 1:
                    raise ValueError(f"{cat.name}/{c.id}: token alias {t!r} is {len(enc)} tokens")
                ids.append(int(enc[0]))
            out.setdefault(cat.name, {})[c.id] = ids
    return out


def prepare_inputs(path: Path = PROBES_DIR / "inputs.json", tokenizer: Any = None) -> dict:
    inv = load_inventory()
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    doc = {
        "concepts": {name: cat.ids for name, cat in inv.items()},
        "items": [
            {k: it[k] for k in ("id", "category", "gold", "request_idx")} for it in B.load_items()
        ],
        "probe_texts": [
            {k: t[k] for k in ("id", "category", "concept", "kind", "request_idx")}
            for t in B.load_probe_texts()
        ],
        "token_ids": token_ids(tokenizer),
        "git_sha": git_sha(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return doc


def iter_eval_scores(path: Path = PROBES_DIR / "eval_scores.jsonl.gz") -> Iterator[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def load_validity(path: Path = PROBES_DIR / "validity.jsonl") -> dict[tuple[str, str, int], dict]:
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[(r["category"], r["concept"], int(r["layer"]))] = r
    return out


RECIPES: tuple[str, ...] = ("v2", "v0")  # primary first
RECIPE_FILES: tuple[str, ...] = (
    "thresholds.json",
    "validity.jsonl",
    "unembed.jsonl",
    "eval_scores.jsonl.gz",
)


def recipe_dirs(probes_dir: Path = PROBES_DIR) -> dict[str, Path]:
    """recipe -> directory holding its merged outputs (legacy layout: ``{"v0": probes_dir}``)."""
    found = {r: probes_dir / r for r in RECIPES if (probes_dir / r / "validity.jsonl").exists()}
    if not found and (probes_dir / "validity.jsonl").exists():
        return {"v0": probes_dir}
    return found


def merge_pieces(probes_dir: Path = PROBES_DIR, pieces: Path | None = None) -> list[Path]:
    """Merge ``pieces/<recipe>/<category>/*`` into ``<probes_dir>/<recipe>/``: validity / unembed
    jsonl and eval_scores (gzip members) concatenated, thresholds.json merged by category,
    ``directions_<cat>.safetensors`` / ``oof_<cat>.npz`` copied."""
    pieces = pieces or probes_dir / "pieces"
    written: list[Path] = []
    for rdir in sorted(p for p in pieces.iterdir() if p.is_dir()):
        out = probes_dir / rdir.name
        out.mkdir(parents=True, exist_ok=True)
        cats = sorted(p for p in rdir.iterdir() if p.is_dir())
        thr: dict = {"concepts": {}, "thresholds": {}}
        for name in ("validity.jsonl", "unembed.jsonl", "eval_scores.jsonl.gz"):
            with open(out / name, "wb") as fh:
                for c in cats:
                    fh.write((c / name).read_bytes())  # gzip members concatenate
            written.append(out / name)
        for c in cats:
            d = json.loads((c / "thresholds.json").read_text())
            thr["concepts"] |= d["concepts"]
            thr["thresholds"] |= d["thresholds"]
            shutil.copy(c / "directions.safetensors", out / f"directions_{c.name}.safetensors")
            shutil.copy(c / "oof.npz", out / f"oof_{c.name}.npz")
        (out / "thresholds.json").write_text(json.dumps(thr))
        written.append(out / "thresholds.json")
    return written


SWEEP_DIR = REPO_ROOT / "outputs" / "sibling" / "probe_sweep"


def compare_with_sweep(
    probes_dir: Path = PROBES_DIR, sweep_dir: Path = SWEEP_DIR, recipe: str = "v2"
) -> dict[str, Any]:
    """Max abs difference between the pipeline's ``recipe`` validity / transfer AUROCs and the
    recipe sweep's (``V2`` / ``V0`` per-concept ``auroc`` / ``transfer``), over every
    (category, concept, layer) of the sweep (plan 0003 rev 11: the port must match, < 1e-3)."""
    from hallucination_bench.evals.sibling_probe.score import index_scores, transfer_auroc

    d = recipe_dirs(probes_dir)[recipe]
    val = load_validity(d / "validity.jsonl")
    scores = index_scores(iter_eval_scores(d / "eval_scores.jsonl.gz"))
    items = {it["id"]: it for it in B.load_items()}
    cats = {
        p.stem: json.loads(p.read_text()) for p in sweep_dir.glob("*.json") if p.stem != "config"
    }
    cats = {k: v for k, v in cats.items() if "results" in v}
    tr = transfer_auroc(scores, items, {c: v["concepts"] for c, v in cats.items()})
    key = recipe.upper()
    dv, dt, n = [], [], 0
    for cat, v in cats.items():
        for L, res in v["results"].items():
            for c, m in res[key]["per_concept"].items():
                k = (cat, c, int(L))
                if k not in val:
                    continue
                n += 1
                dv.append(abs(m["auroc"] - val[k]["auroc"]))
                if np.isfinite(m["transfer"]) or np.isfinite(tr.get(k, float("nan"))):
                    dt.append(abs(m["transfer"] - tr[k]))
    return {
        "recipe": recipe,
        "n": n,
        "max_abs_diff_validity": max(dv, default=float("nan")),
        "max_abs_diff_transfer": max(dt, default=float("nan")),
    }
