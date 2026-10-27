"""Write data/strongreject/prompts.json (plan 0016 §Data): 40 forbidden prompts stratified by
category (7/7/7/7/6/6 over the sorted categories, random.Random(0) within each) x the 7 families
of ``families.FAMILIES`` = 280 prompts. Deterministic; re-run, never hand-edit."""

import hashlib
import json
import random
from collections import defaultdict

from hallucination_bench import DATA
from hallucination_bench.evals.strongreject import families as F

PER_CATEGORY = (7, 7, 7, 7, 6, 6)
OUT = DATA / "strongreject" / "prompts.json"


def sha(p) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    rows = F.load_dataset()
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    cats = sorted(by_cat)
    assert len(cats) == len(PER_CATEGORY) == 6, cats
    rng = random.Random(0)
    chosen: list[dict] = []
    for cat, n in zip(cats, PER_CATEGORY, strict=True):
        chosen += sorted(rng.sample(by_cat[cat], n), key=lambda r: r["forbidden_id"])
    assert len(chosen) == 40
    prompts = []
    for r in chosen:
        for fam in F.FAMILY_NAMES:
            prompts.append(
                {
                    "id": f"sr-{r['forbidden_id']:03d}-{fam}",
                    "forbidden_id": r["forbidden_id"],
                    "category": r["category"],
                    "source": r["source"],
                    "family": fam,
                    "forbidden_prompt": r["forbidden_prompt"],
                    "jailbreak_prompt": F.wrap(r["forbidden_prompt"], fam),
                }
            )
    meta = {
        "plan": "0016",
        "n_forbidden": len(chosen),
        "n_prompts": len(prompts),
        "families": list(F.FAMILY_NAMES),
        "per_category": dict(zip(cats, PER_CATEGORY, strict=True)),
        "seed": 0,
        "dataset": {
            "repo": "alexandrasouly/strongreject",
            "commit": F.DATASET_COMMIT,
            "sha256": sha(F.DATASET_CSV),
        },
        "wrappers": {
            "repo": "dsbowen/strong_reject",
            "commit": F.PACKAGE_COMMIT,
            "sha256": sha(F.WRAPPERS_JSON),
        },
    }
    OUT.write_text(
        json.dumps({"meta": meta, "prompts": prompts}, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT}: {len(prompts)} prompts")


if __name__ == "__main__":
    main()
