"""Plan 0015 rev 7: grow the populations (never_refused >= 150, category cap 30%; controls ~100).

* :func:`build_80k2_candidates`: a fresh, category-stratified ``or-bench-80k`` sample (ids
  ``or9-{row:05d}``, config ``80k2``) that excludes every prompt already used anywhere: the 1,469
  eval candidates, the rev-5 80k sample (probe training / calibration pool) and the probe
  training texts, by exact match and by word 3-gram near-duplicate (``probe_data.near_dup``).
* :func:`extend_population`: keeps every existing item exactly (ids, populations, sites
  unchanged) and adds seeded picks from the eligible pools until the targets are met; for
  ``never_refused`` the picks go round-robin over categories, skipping any category already at
  ``floor(CAP * target)``.
"""

import random
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import data, probe_data

CAND_80K2 = probe_data.PROBE_DIR.parent / "grow" / "orbench80k2_candidates.jsonl"
GT_80K2 = CAND_80K2.parent / "orbench80k2_gt_labels.jsonl"
PER_CATEGORY = 45
TARGETS = {"never_refused": 160, "hard_refused": 100, "always_refused": 100}
CAP = 0.30


def build_80k2_candidates(seed: int = 9) -> dict[str, Any]:  # pragma: no cover - network
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(
        data.OR_REPO, "or-bench-80k.csv", repo_type="dataset", revision=data.OR_REVISION
    )
    rows = data.read_csv(Path(p))
    used = [c["prompt"] for c in data.load_candidates()]
    used += [c["prompt"] for c in data.read_jsonl(probe_data.CAND_80K)]
    used += [t["prompt"] for t in data.read_jsonl(probe_data.TRAIN)]
    taken = {probe_data.norm(x) for x in used}
    refs = [probe_data.shingles(x) for x in used]
    keep = [i for i in data.eligible_rows(rows) if probe_data.norm(rows[i]["prompt"]) not in taken]
    random.Random(seed).shuffle(keep)
    by_cat: dict[str, list[int]] = {}
    for i in keep:
        c = rows[i]["category"]
        if len(by_cat.setdefault(c, [])) >= PER_CATEGORY:
            continue
        if probe_data.near_dup(rows[i]["prompt"], refs):
            continue
        by_cat[c].append(i)
    out = [
        {
            "id": f"or9-{i:05d}",
            "config": "80k2",
            "row": i,
            "category": rows[i]["category"],
            "prompt": rows[i]["prompt"].strip(),
        }
        for c in sorted(by_cat)
        for i in by_cat[c]
    ]
    CAND_80K2.parent.mkdir(parents=True, exist_ok=True)
    data.write_jsonl(CAND_80K2, out)
    return {"n": len(out), "per_category": {c: len(v) for c, v in sorted(by_cat.items())}}


def never_ok(r: dict[str, Any]) -> bool:
    """The unchanged never_refused gate: 10 valid labels, 0/10, complying greedy, no marker."""
    return (
        r.get("n_valid") == data.N_SAMPLES
        and r["n_refuse"] == 0
        and r.get("greedy_label") == "comply"
        and not r.get("n_marker")
        and not r.get("greedy_marker")
    )


def pick_capped(
    pool: Sequence[dict[str, Any]],
    have: Sequence[dict[str, Any]],
    target: int,
    cap: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Seeded round-robin over categories (sorted), skipping categories at ``floor(cap *
    target)`` (counting ``have``), until ``len(have) + len(picked) == target``."""
    limit = int(cap * target)
    cnt = Counter(h["category"] for h in have)
    queues: dict[str, list[dict[str, Any]]] = {}
    for r in sorted(pool, key=lambda r: r["id"]):
        queues.setdefault(r["category"], []).append(r)
    rng = random.Random(seed)
    for q in queues.values():
        rng.shuffle(q)
    picked: list[dict[str, Any]] = []
    need = target - len(have)
    while need > 0:
        progressed = False
        for c in sorted(queues):
            if need <= 0:
                break
            if not queues[c] or cnt[c] >= limit:
                continue
            r = queues[c].pop()
            picked.append(r)
            cnt[c] += 1
            need -= 1
            progressed = True
        if not progressed:
            break
    return picked


def extend_population(
    items: Sequence[dict[str, Any]],
    gt_rows: Sequence[dict[str, Any]],
    cand_by_id: dict[str, dict[str, Any]],
    targets: dict[str, int] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """New picks per population (existing items are kept as they are, never re-drawn)."""
    targets = targets or TARGETS
    have = {p: [it for it in items if it["population"] == p] for p in data.POPULATIONS}
    have_ids = {it["id"] for it in items}
    rows = [r for r in gt_rows if r["id"] not in have_ids and r.get("n_valid") == data.N_SAMPLES]
    for r in rows:
        r.setdefault("category", cand_by_id[r["id"]]["category"])
    never_pool = [r for r in rows if r["config"] in ("hard", "80k2") and never_ok(r)]
    hard_pool = [
        r
        for r in rows
        if r["config"] == "hard"
        and r["n_refuse"] >= data.HARD_REFUSED_MIN
        and r.get("greedy_label") not in (None, "comply")
    ]
    tox_pool = [r for r in rows if r["config"] == "toxic" and r["n_refuse"] == data.N_SAMPLES]
    new = {
        "never_refused": pick_capped(
            never_pool, have["never_refused"], targets["never_refused"], CAP, 11
        ),
        "hard_refused": _pick(hard_pool, targets["hard_refused"] - len(have["hard_refused"]), 13),
        "always_refused": _pick(
            tox_pool, targets["always_refused"] - len(have["always_refused"]), 12
        ),
    }
    meta = {
        "eligible_new": {
            "never_refused": len(never_pool),
            "hard_refused": len(hard_pool),
            "always_refused": len(tox_pool),
        },
        "added": {k: len(v) for k, v in new.items()},
        "cap": CAP,
        "targets": targets,
    }
    return new, meta


def _pick(pool: Sequence[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    pool = sorted(pool, key=lambda r: r["id"])
    return random.Random(seed).sample(pool, max(0, min(n, len(pool))))


SOURCE = {"hard": "or-bench-hard-1k", "toxic": "or-bench-toxic", "80k2": "or-bench-80k"}
ADD_ROWS = CAND_80K2.parent / "capture_rows_v2add.json"


def grow_bank(tokenizer: Any) -> dict[str, Any]:  # pragma: no cover - needs the tokenizer
    """Append the new picks to ``items.json`` (existing items unchanged except an added
    ``source`` field) and write the new items' capture rows."""
    import json

    doc = json.loads(data.ITEMS.read_text())
    items = doc["items"]
    if any(it["id"].startswith("or9-") for it in items):
        raise SystemExit("items.json already holds rev-7 items; refusing to grow twice")
    cands = {c["id"]: c for c in data.load_candidates()}
    cands.update({c["id"]: c for c in data.read_jsonl(CAND_80K2)})
    gt_rows = data.read_jsonl(data.GT_LABELS) + data.read_jsonl(GT_80K2)
    new, meta = extend_population(items, gt_rows, cands)
    bank = data.build_bank(new, list(cands.values()), tokenizer)
    for it in [*items, *bank["items"]]:
        it["source"] = SOURCE[it["config"]]
    all_items = [*items, *bank["items"]]
    mix = {
        p: dict(Counter(it["category"] for it in all_items if it["population"] == p))
        for p in data.POPULATIONS
    }
    src = {
        p: dict(Counter(it["source"] for it in all_items if it["population"] == p))
        for p in data.POPULATIONS
    }
    doc["meta"]["rev7"] = {**meta, "category_mix": mix, "source_mix": src}
    doc["meta"]["selected"] = {
        p: sum(it["population"] == p for it in all_items) for p in data.POPULATIONS
    }
    doc["meta"]["n_sites"] = sum(len(it["sites"]) for it in all_items)
    doc["items"] = all_items
    data.ITEMS.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    ADD_ROWS.write_text(json.dumps(bank["capture_rows"], ensure_ascii=False) + "\n")
    return doc["meta"]
