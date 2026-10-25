"""Export the V4 UM ladder's labelled units as a hallucination-index sibling root (plan 0014).

``hbench index`` keys sibling units by their ``method`` field, but the V4 ladder labels the V4
J-lens arm as ``jlens`` (the ladder code's fixed J-lens name), which collides with the Qwen
``jlens`` of the official sibling root. This copies

    <um>/label/units_labeled.jsonl                 (country)
    <um>/<domain>/label/units_labeled.jsonl        (capital_city, holiday)

to the same layout under ``<dest>`` with ``jlens`` renamed ``v4-jlens-std-n1000`` (in ``method``
and in the ``cross`` / ``cross_any_layer`` method lists); every other field and line order is
kept, so the index reads exactly the units ``um3_v4.py`` aggregates. Default paths follow the
index's own ``--out`` layout (``<out>/sibling/um``),
so the documented ``hbench index`` command (``--dest`` + root flags) finds it.

    .venv/bin/python scripts/v4_um/export_index_root.py [--um outputs/v4/um]
        [--dest outputs/v4/sibling/um]
"""

import argparse
import json
from pathlib import Path

from hallucination_bench import index

ROOT = Path(__file__).resolve().parents[2]
RENAME = {"jlens": "v4-jlens-std-n1000"}


def _rename(rec: dict) -> dict:
    rec["method"] = RENAME.get(rec["method"], rec["method"])
    for k in ("cross", "cross_any_layer"):
        if isinstance(rec.get(k), list):
            rec[k] = [RENAME.get(m, m) for m in rec[k]]
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--um", default=str(ROOT / "outputs" / "v4" / "um"))
    ap.add_argument("--dest", default=str(ROOT / "outputs" / "v4" / "sibling" / "um"))
    a = ap.parse_args()
    src, dest = index.um_paths(Path(a.um)), index.um_paths(Path(a.dest))
    for d in index.UM_DOMAINS:
        lines = src[d].read_text(encoding="utf-8").splitlines()
        out = [json.dumps(_rename(json.loads(x)), ensure_ascii=False) for x in lines]
        dest[d].parent.mkdir(parents=True, exist_ok=True)
        dest[d].write_text("\n".join(out) + "\n", encoding="utf-8")
        print(f"{d}: {len(out)} units -> {dest[d]}")


if __name__ == "__main__":
    main()
