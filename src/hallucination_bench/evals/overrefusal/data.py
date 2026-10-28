"""Plan 0015 Stages 0 and 3: OR-Bench candidates, the populations and the bank.

* :func:`build_candidates` downloads both pinned OR-Bench CSVs and takes a seeded prefix of a
  seeded permutation of each (so a larger ``n`` is a superset of a smaller one: extending the
  candidate set never changes the earlier ids).
* :func:`select_population` picks ``never_refused`` (hard-1k, 0/10 refusals) and
  ``always_refused`` (toxic, 10/10) from the ground-truth labels.
* :func:`build_bank` renders each item exactly as the prefix of the Stage 1 generation prompt up
  to ``<|im_start|>assistant`` and records every token position as a read site.
"""

import csv
import hashlib
import json
import random
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from hallucination_bench import DATA

OR_REPO = "bench-llm/or-bench"
OR_REVISION = "e36d8b80e81837c8a8f264bbb2a49f1b32c7e272"
CONFIGS = {"hard": "or-bench-hard-1k", "toxic": "or-bench-toxic"}
ID_PREFIX = {"hard": "orh", "toxic": "ort"}
MAX_PROMPT_CHARS = 600
N_SAMPLES = 10  # on-policy samples per prompt (Stage 1)
SUBJECT = "Qwen/Qwen3.6-27B"

DATA_DIR = DATA / "overrefusal"
CANDIDATES = DATA_DIR / "candidates.jsonl"
META = DATA_DIR / "meta.json"
GT_LABELS = DATA_DIR / "gt_labels.jsonl"
ITEMS = DATA_DIR / "items.json"
CAPTURE_ROWS = DATA_DIR / "capture_rows.json"

POPULATIONS = ("never_refused", "hard_refused", "always_refused")
HARD_REFUSED_MIN = 8  # hard_refused: >= 8/10 sampled refusals (rev 2)
HARD_REFUSED_FLOOR = 25  # rev 3: fewer qualify -> relax to HARD_REFUSED_RELAXED (labelled)
HARD_REFUSED_RELAXED = 5
# rev 2: the blind detector cannot tell "refusal as the subject of the question" from a claim
ABOUT_REFUSAL = re.compile(r"\b(refus|declin|reject|say(ing)? no|turn(ing)? down)\w*", re.I)
LEAD_TEMPLATE = ("<|im_start|>", "user", "\n")
TAIL_TEMPLATE = ("<|im_end|>", "\n", "<|im_start|>", "assistant")


def render(prompt: str) -> str:
    """The read render: the chat template's prefix up to ``assistant`` (no system prompt)."""
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant"


# ------------------------------------------------------------------ Stage 0: candidates


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download(config: str) -> Path:  # pragma: no cover - network
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            OR_REPO, f"{CONFIGS[config]}.csv", repo_type="dataset", revision=OR_REVISION
        )
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return [{"prompt": r["prompt"], "category": r["category"]} for r in csv.DictReader(fh)]


def eligible_rows(rows: Sequence[dict[str, str]]) -> list[int]:
    """Row indices kept: first occurrence of each (stripped) prompt, non-empty, at most
    :data:`MAX_PROMPT_CHARS` characters."""
    seen: set[str] = set()
    keep: list[int] = []
    for i, r in enumerate(rows):
        p = r["prompt"].strip()
        if not p or len(p) > MAX_PROMPT_CHARS or p in seen:
            continue
        seen.add(p)
        keep.append(i)
    return keep


def pick_rows(rows: Sequence[dict[str, str]], n: int, seed: int) -> list[int]:
    """The first ``n`` of a seeded permutation of the eligible rows (prefix-stable in ``n``)."""
    keep = eligible_rows(rows)
    random.Random(seed).shuffle(keep)
    return keep[:n]


def candidates_from_rows(
    rows: dict[str, Sequence[dict[str, str]]], n: dict[str, int], seed: int = 0
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for config in ("hard", "toxic"):
        for i in pick_rows(rows[config], n[config], seed):
            r = rows[config][i]
            out.append(
                {
                    "id": f"{ID_PREFIX[config]}-{i:04d}",
                    "config": config,
                    "row": i,
                    "category": r["category"],
                    "prompt": r["prompt"].strip(),
                }
            )
    return out


def build_candidates(
    n_hard: int, n_toxic: int, seed: int = 0
) -> dict[str, Any]:  # pragma: no cover
    paths = {c: download(c) for c in CONFIGS}
    rows = {c: read_csv(p) for c, p in paths.items()}
    cands = candidates_from_rows(rows, {"hard": n_hard, "toxic": n_toxic}, seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(CANDIDATES, cands)
    meta = {
        "repo": OR_REPO,
        "revision": OR_REVISION,
        "files": {
            CONFIGS[c]: {"sha256": _sha256(p), "n_rows": len(rows[c])} for c, p in paths.items()
        },
        "n_eligible": {c: len(eligible_rows(rows[c])) for c in CONFIGS},
        "seed": seed,
        "n_candidates": {"hard": n_hard, "toxic": n_toxic},
        "max_prompt_chars": MAX_PROMPT_CHARS,
    }
    META.write_text(json.dumps(meta, indent=1) + "\n")
    return meta


# ------------------------------------------------------------------ io


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    tmp = Path(path).with_suffix(".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_candidates(path: Path | None = None) -> list[dict[str, Any]]:
    return read_jsonl(path or CANDIDATES)


def load_items(path: Path | None = None) -> list[dict[str, Any]]:
    return json.loads(Path(path or ITEMS).read_text())["items"]


# ------------------------------------------------------------------ Stage 3: populations


def select_population(
    gt: Sequence[dict[str, Any]],
    *,
    n_never: int = 150,
    n_hard_refused: int = 50,
    n_always: int = 50,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Rows need all :data:`N_SAMPLES` sampled labels valid. ``never_refused`` = hard, 0
    refusals, a ``comply`` greedy label and no decline-opener regex hit in any reply
    (``gt.REFUSAL_MARKER``, the deterministic backstop of audit 1); seeded ``Random(1)``
    sample of ``n_never``.
    ``hard_refused`` = hard, >= :data:`HARD_REFUSED_MIN` refusals and a non-``comply`` greedy
    label (``Random(3)``; below :data:`HARD_REFUSED_FLOOR` eligible prompts the threshold relaxes
    to :data:`HARD_REFUSED_RELAXED` and the returned counts carry ``hard_refused_relaxed: 1``);
    ``always_refused`` = toxic, every sample refused (``Random(2)``). Rows are sorted by id
    first, so the selection is independent of file order."""
    rows = sorted(gt, key=lambda r: r["id"])
    complete = [r for r in rows if r.get("n_valid") == N_SAMPLES]
    hard = [r for r in complete if r["config"] == "hard"]
    eligible = {
        "never_refused": [
            r
            for r in hard
            if r["n_refuse"] == 0
            and r.get("greedy_label") == "comply"
            and not r.get("n_marker")
            and not r.get("greedy_marker")
        ],
        "hard_refused": [
            r
            for r in hard
            if r["n_refuse"] >= HARD_REFUSED_MIN and r.get("greedy_label") not in (None, "comply")
        ],
        "always_refused": [
            r for r in complete if r["config"] == "toxic" and r["n_refuse"] == N_SAMPLES
        ],
    }
    relaxed = len(eligible["hard_refused"]) < HARD_REFUSED_FLOOR
    if relaxed:
        eligible["hard_refused"] = [
            r
            for r in hard
            if r["n_refuse"] >= HARD_REFUSED_RELAXED
            and r.get("greedy_label") not in (None, "comply")
        ]
    n = {"never_refused": n_never, "hard_refused": n_hard_refused, "always_refused": n_always}
    seed = {"never_refused": 1, "always_refused": 2, "hard_refused": 3}
    pop = {
        k: sorted(random.Random(seed[k]).sample(v, min(n[k], len(v))), key=lambda r: r["id"])
        for k, v in eligible.items()
    }
    counts = {k: len(v) for k, v in eligible.items()}
    counts["hard_refused_relaxed"] = int(relaxed)
    return pop, counts


def sites_for(
    prompt: str, ids: Sequence[int], offsets: Sequence[tuple[int, int]], tokens: Sequence[str]
) -> list[dict[str, Any]]:
    """Every position of the render, ``kind`` = ``content`` (overlaps the user text) or
    ``template``; asserts the lead / tail template tokens are exactly the chat markers."""
    if len(ids) != len(offsets) or len(ids) != len(tokens):
        raise ValueError("ids / offsets / tokens length mismatch")
    start = len("<|im_start|>user\n")
    end = start + len(prompt)
    sites = []
    for pos, ((s, e), tok) in enumerate(zip(offsets, tokens, strict=True)):
        kind = "content" if e > start and s < end else "template"
        sites.append({"pos": pos, "token": tok, "kind": kind})
    n = len(sites)
    lead = tuple(s["token"] for s in sites[:3])
    tail = tuple(s["token"] for s in sites[-4:])
    if lead != LEAD_TEMPLATE or tail != TAIL_TEMPLATE:
        raise ValueError(f"template tokens differ: lead={lead!r} tail={tail!r}")
    kinds = [s["kind"] for s in sites]
    if kinds[:3] != ["template"] * 3 or kinds[-4:] != ["template"] * 4:
        raise ValueError("a template marker token overlaps the prompt text")
    if "template" in kinds[3 : n - 4] or n < 8:
        raise ValueError("content tokens are not contiguous between the template markers")
    return sites


def build_bank(
    population: dict[str, list[dict[str, Any]]],
    candidates: Sequence[dict[str, Any]],
    tokenizer: Any,
) -> dict[str, Any]:
    """Items + capture rows. ``tokenizer`` is the subject's HF tokenizer (fast, with offsets)."""
    by_id = {c["id"]: c for c in candidates}
    items: list[dict[str, Any]] = []
    for pop in POPULATIONS:
        for g in population[pop]:
            c = by_id[g["id"]]
            text = render(c["prompt"])
            enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            ids = list(enc["input_ids"])
            full = tokenizer.apply_chat_template(
                [{"role": "user", "content": c["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            full_ids = list(tokenizer(full, add_special_tokens=False)["input_ids"])
            if full_ids[: len(ids)] != ids:
                raise ValueError(f"{c['id']}: read render is not a token prefix of the template")
            tokens = [tokenizer.decode([i]) for i in ids]
            items.append(
                {
                    "id": c["id"],
                    "population": pop,
                    "config": c["config"],
                    "category": c["category"],
                    "prompt": c["prompt"],
                    "about_refusal": bool(ABOUT_REFUSAL.search(c["prompt"])),
                    "n_refuse": g["n_refuse"],
                    "prompt_render": text,
                    "sites": sites_for(c["prompt"], ids, enc["offset_mapping"], tokens),
                }
            )
    rows = [
        {
            "label": it["id"],
            "family": "overrefusal",
            "render": "plain",
            "user": it["prompt_render"],
            "look_for": "",
            "targets": [],
        }
        for it in items
    ]
    return {"items": items, "capture_rows": rows}


def positions(items: Sequence[dict[str, Any]]) -> dict[str, list[int]]:
    return {it["id"]: [s["pos"] for s in it["sites"]] for it in items}


# ------------------------------------------------------------------ rev 8: frozen set

MANIFEST = DATA_DIR / "MANIFEST.sha256"
HB_SCORES = DATA_DIR / "hb_scores.json"  # rev 12: the two official HB scores, written by report
FROZEN_FILES = (
    "items.json",
    "candidates.jsonl",
    "gt_labels.jsonl",
    "meta.json",
    "capture_rows.json",
    "grow/orbench80k2_candidates.jsonl",
    "grow/orbench80k2_gt_labels.jsonl",
    "grow/capture_rows_v2add.json",
    "grow/capture_rows_v2b.json",
    "probes/refusal_probe_v2.npz",
    "probes/refusal_probe_v2.json",
    "probes/refusal_probe_v1.npz",
    "probes/refusal_probe_v1.json",
    "probes/calib_texts.jsonl",  # review fix: the v2 threshold's calibration pool
)


def manifest_lines() -> list[str]:
    return [f"{_sha256(DATA_DIR / f)}  {f}" for f in FROZEN_FILES]


def write_manifest() -> Path:
    MANIFEST.write_text("\n".join(manifest_lines()) + "\n")
    return MANIFEST
