"""Plan 0019 (mask): capture rows + site files for the 7 new layers (20/24/32/40/48/56/60),
PROMPT positions only (input_ids cut after the final ``assistant`` token; causal, so the prompt
activations equal those of the full sequence).

- ``data/mask/l11/capture_rows_hr.json``: the 155 gated honest_resisted items (lens reads +
  probes) -> capture ``hb-mask-L11-hr``;
- ``data/mask/l11/capture_rows_rest.json``: every other read-set item (probe data only; LOAO trains
  on every labelled item) -> ``hb-mask-L11-rest``;
- ``data/mask/l11/capture_rows_neg.json``: the unread ``hb-mask-v5-neg`` items ->
  ``hb-mask-L11-neg``;
- ``data/mask/l11/sites_l11.json.gz``: the honest_resisted prompt sites (the L11 read grid).

    uv run --no-sync python scripts/mask/l11_build.py
"""

import gzip
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from hallucination_bench.evals.mask import bank  # noqa: E402
from hallucination_bench.evals.mask import report as R  # noqa: E402

OUT = bank.DATA_DIR / "l11"


def _rows(capture_file: str) -> dict[str, dict]:
    return {r["label"]: r for r in json.loads((bank.DATA_DIR / capture_file).read_text())}


def _prompt_only(row: dict, sites: list[dict]) -> dict:
    ps = [s["pos"] for s in sites if s["site_type"] == "prompt"]
    a = max(ps)
    assert ps == list(range(a + 1)), row["label"]
    return {**row, "input_ids": row["input_ids"][: a + 1], "eval_positions": ps}


def main() -> None:
    gate = json.loads((bank.DATA_DIR / "gate.json").read_text())["items"]
    frame, _ = R.choose_frame(gate)
    hr, _li = R._gated(gate, frame)
    sites = bank.load_sites()
    with gzip.open(bank.DATA_DIR / "sites_neg.json.gz", "rt", encoding="utf-8") as fh:
        neg_sites = {r["task_id"]: r["sites"] for r in json.load(fh)["items"]}
    rows, neg_rows = _rows("capture_rows.json"), _rows("capture_rows_neg.json")
    OUT.mkdir(parents=True, exist_ok=True)
    hr_rows = [_prompt_only(rows[t], sites[t]) for t in sorted(hr)]
    rest = [_prompt_only(rows[t], sites[t]) for t in sorted(set(sites) - hr)]
    neg = [_prompt_only(neg_rows[t], neg_sites[t]) for t in sorted(neg_sites)]
    for name, rr in (("hr", hr_rows), ("rest", rest), ("neg", neg)):
        (OUT / f"capture_rows_{name}.json").write_text(json.dumps(rr) + "\n")
    l11 = [
        {"task_id": t, "sites": [s for s in sites[t] if s["site_type"] == "prompt"]}
        for t in sorted(hr)
    ]
    with gzip.open(OUT / "sites_l11.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(
            {"meta": {"plan": "0019", "population": "gated honest_resisted"}, "items": l11}, fh
        )
    n = sum(len(x["sites"]) for x in l11)
    print(f"hr {len(hr_rows)} items / {n} prompt sites; rest {len(rest)}; neg {len(neg)}")


if __name__ == "__main__":
    main()
