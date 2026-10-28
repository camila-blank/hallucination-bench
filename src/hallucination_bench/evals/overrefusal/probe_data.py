"""Plan 0015 rev 5: training data for the refusal probes (never the read set).

* ``refusal-dm-v1`` (primary, Arditi et al. style): harmful = AdvBench + HarmBench standard,
  harmless = Alpaca (empty ``input``); seeded 300 + 300, OR-Bench string matches dropped.
* ``refusal-beh-v1`` (behaviour-labelled): prompts Qwen3.6-27B refuses vs answers by the
  Stage 1-2 recipe; the answered side comes from a fresh 500-prompt ``or-bench-80k`` sample.

Every text is rendered exactly like the read set (:func:`data.render`) and captured with
``render: plain``; ``capture_rows`` labels are the text ids.
"""

import json
import random
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from . import data, gt

PROBE_DIR = data.DATA_DIR / "probes"
CAND_80K = PROBE_DIR / "orbench80k_candidates.jsonl"
GT_80K = PROBE_DIR / "orbench80k_gt_labels.jsonl"
TRAIN = PROBE_DIR / "train_texts.jsonl"
TRAIN_ROWS = PROBE_DIR / "train_capture_rows.json"
TRAIN_ACTS = "hb-overrefusal-probe-train"

# the HF mirrors (walledai/AdvBench, walledai/HarmBench) are gated for us: the source repos'
# CSVs at pinned commits instead
ADVBENCH = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
    "098262edf85f807224e70ecd87b9d83716bf6b73/data/advbench/harmful_behaviors.csv"
)
HARMBENCH = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/"
    "8e1604d1171fe8a48d8febecd22f600e462bdcdd/data/behavior_datasets/"
    "harmbench_behaviors_text_all.csv"
)
ALPACA = ("tatsu-lab/alpaca", "dce01c9b08f87459cf36a430d809084718273017")
N_80K = 500
N_DM = 300
N_BEH_MAX = 300
BEH_MIN = 60
VARIANTS = ("refusal-dm-v1", "refusal-beh-v1")


NEAR_DUP_JACCARD = 0.5  # rev 6: word 3-gram Jaccard vs any read candidate


def norm(p: str) -> str:
    return " ".join(p.strip().casefold().split())


def shingles(p: str) -> frozenset[tuple[str, ...]]:
    w = "".join(ch if ch.isalnum() else " " for ch in p.casefold()).split()
    return frozenset(zip(w, w[1:], w[2:], strict=False)) if len(w) >= 3 else frozenset({tuple(w)})


def near_dup(p: str, refs: Sequence[frozenset[tuple[str, ...]]]) -> bool:
    a = shingles(p)
    return any(len(a & b) / max(1, len(a | b)) >= NEAR_DUP_JACCARD for b in refs if a & b)


def _parquet(repo: tuple[str, str], path: str) -> list[dict[str, Any]]:  # pragma: no cover
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    f = hf_hub_download(repo[0], path, repo_type="dataset", revision=repo[1])
    return pq.read_table(f).to_pylist()


def _csv_url(url: str) -> list[dict[str, str]]:  # pragma: no cover - network
    import csv
    import io
    import urllib.request

    with urllib.request.urlopen(url, timeout=60) as r:
        return list(csv.DictReader(io.StringIO(r.read().decode("utf-8"))))


def build_80k_candidates(seed: int = 5) -> dict[str, Any]:  # pragma: no cover - network
    """A seeded 500-prompt ``or-bench-80k`` sample (ids ``or8-{row:05d}``) minus the 1,469."""
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(
        data.OR_REPO, "or-bench-80k.csv", repo_type="dataset", revision=data.OR_REVISION
    )
    rows = data.read_csv(Path(p))
    taken = {norm(c["prompt"]) for c in data.load_candidates()}
    # exact matches only here (this built the committed file); near duplicates of the read
    # candidates are dropped later, in build_train_texts (rev 6)
    keep = [i for i in data.eligible_rows(rows) if norm(rows[i]["prompt"]) not in taken]
    random.Random(seed).shuffle(keep)
    out = [
        {
            "id": f"or8-{i:05d}",
            "config": "80k",
            "row": i,
            "category": rows[i]["category"],
            "prompt": rows[i]["prompt"].strip(),
        }
        for i in keep[:N_80K]
    ]
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    data.write_jsonl(CAND_80K, out)
    return {"n_rows": len(rows), "n_eligible": len(keep), "n": len(out), "seed": seed}


def behaviour_sets(
    gt80k: Sequence[dict[str, Any]], gt_main: Sequence[dict[str, Any]], read_ids: set[str]
) -> tuple[list[str], list[str]]:
    """(refused ids, answered ids) from the 80k sample only (rev 6): answered = 0/10 with no
    marker; refused = >= 9/10."""
    ok = [r for r in gt80k if r.get("n_valid") == data.N_SAMPLES]
    answered = [r["id"] for r in ok if r["n_refuse"] == 0 and not r.get("n_marker")]
    refused = [r["id"] for r in ok if r["n_refuse"] >= 9]
    # rev 6: both classes from the 80k sample only (gt_main / read_ids kept for the signature)
    assert not (set(refused) | set(answered)) & read_ids
    del gt_main
    return sorted(refused), sorted(answered)


def _text(id_: str, variant: str, label: int, prompt: str, source: str) -> dict[str, Any]:
    return {
        "id": id_,
        "variant": variant,
        "label": label,  # 0 = refuse / harmful (concept 0), 1 = comply / harmless
        "prompt": prompt,
        "source": source,
    }


def build_train_texts(
    harmful: Iterable[tuple[str, str]],
    harmless: Iterable[tuple[str, str]],
    beh_refused: Sequence[dict[str, Any]],
    beh_answered: Sequence[dict[str, Any]],
    exclude: set[str],
    seed: int = 0,
    near: Sequence[frozenset[tuple[str, ...]]] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Both variants' texts. ``harmful`` / ``harmless`` are (source, prompt) pairs; ``exclude``
    holds normalised OR-Bench prompts (dropped from the dm variant)."""

    def pick(pool: Iterable[tuple[str, str]], n: int, salt: int) -> list[tuple[str, str]]:
        seen: set[str] = set()
        out = []
        for src, p in pool:
            k = norm(p)
            if (
                not p.strip()
                or len(p.strip()) > data.MAX_PROMPT_CHARS
                or k in exclude
                or k in seen
                or near_dup(p, near)
            ):
                continue
            seen.add(k)
            out.append((src, p.strip()))
        random.Random(seed + salt).shuffle(out)
        return out[:n]

    hf, hl = pick(harmful, N_DM, 1), pick(harmless, N_DM, 2)
    texts = [_text(f"dm-h{i:03d}", VARIANTS[0], 0, p, s) for i, (s, p) in enumerate(hf)]
    texts += [_text(f"dm-b{i:03d}", VARIANTS[0], 1, p, s) for i, (s, p) in enumerate(hl)]
    beh_refused = [r for r in beh_refused if not near_dup(r["prompt"], near)]
    beh_answered = [r for r in beh_answered if not near_dup(r["prompt"], near)]
    n = min(len(beh_refused), len(beh_answered), N_BEH_MAX)
    rr = sorted(beh_refused, key=lambda r: r["id"])
    aa = sorted(beh_answered, key=lambda r: r["id"])
    rr = random.Random(seed + 3).sample(rr, n) if n else []
    aa = random.Random(seed + 4).sample(aa, n) if n else []
    texts += [_text(f"beh-{r['id']}", VARIANTS[1], 0, r["prompt"], "orbench") for r in rr]
    texts += [_text(f"beh-{r['id']}", VARIANTS[1], 1, r["prompt"], "orbench") for r in aa]
    meta = {
        "n": {
            VARIANTS[0]: {"harmful": len(hf), "harmless": len(hl)},
            VARIANTS[1]: {"refused": len(rr), "answered": len(aa)},
        },
        "beh_underpowered": n < BEH_MIN,
        "seed": seed,
    }
    return texts, meta


def capture_rows(texts: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "label": t["id"],
            "family": "overrefusal-probe",
            "render": "plain",
            "user": data.render(t["prompt"]),
            "look_for": "",
            "targets": [],
        }
        for t in texts
    ]


def build_all() -> dict[str, Any]:  # pragma: no cover - network
    harmful = [("advbench", r["goal"]) for r in _csv_url(ADVBENCH)]
    harmful += [  # HarmBench standard behaviours (no context string)
        ("harmbench", r["Behavior"])
        for r in _csv_url(HARMBENCH)
        if r["FunctionalCategory"] == "standard"
    ]
    harmless = [
        ("alpaca", r["instruction"])
        for r in _parquet(ALPACA, "data/train-00000-of-00001-a09b74b3ef9c3b56.parquet")
        if not (r.get("input") or "").strip()
    ]
    cands = {c["id"]: c for c in data.load_candidates()}
    c80 = {c["id"]: c for c in data.read_jsonl(CAND_80K)}
    exclude = {norm(c["prompt"]) for c in [*cands.values(), *c80.values()]}
    read_ids = {it["id"] for it in data.load_items()}
    ref_ids, ans_ids = behaviour_sets(
        data.read_jsonl(GT_80K), data.read_jsonl(data.GT_LABELS), read_ids
    )
    allc = {**cands, **c80}
    refs = [shingles(c["prompt"]) for c in cands.values()]
    texts, meta = build_train_texts(
        harmful,
        harmless,
        [allc[i] for i in ref_ids],
        [allc[i] for i in ans_ids],
        exclude,
        near=refs,
    )
    data.write_jsonl(TRAIN, texts)
    TRAIN_ROWS.write_text(json.dumps(capture_rows(texts), ensure_ascii=False) + "\n")
    meta.update(
        eligible={"refused": len(ref_ids), "answered": len(ans_ids)},
        sources={"advbench": ADVBENCH, "harmbench": HARMBENCH, "alpaca": ALPACA},
        refusal_marker=gt.REFUSAL_MARKER.pattern,
    )
    (PROBE_DIR / "train_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta
