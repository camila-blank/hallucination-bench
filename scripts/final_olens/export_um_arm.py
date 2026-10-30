"""Export ONE arm's labelled UM ladder units as a hallucination-index sibling root.

``um_ladder_arm.py`` labels a new arm in a COPY of the official um root, where the frozen arms'
units are relabelled too (cross-method support now includes the new arm). The index must read
the frozen arms from the official root only, and one method under two roots is an error, so this
keeps the units whose ``method`` is ``--arm`` (every field and line order unchanged) and writes
them in the index layout (country at ``label/``, others at ``<domain>/label/``).

    python scripts/final_olens/export_um_arm.py --arm final-olens-rl600-x4 --um <copy> --dest <root>
"""

import argparse
import json
from pathlib import Path

from hallucination_bench import index


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--um", required=True, help="the labelled copy of the um root")
    ap.add_argument("--dest", required=True)
    a = ap.parse_args()
    if Path(a.dest).resolve() == Path(a.um).resolve():
        raise SystemExit("--dest must not be the labelled copy itself")
    src, dest = index.um_paths(Path(a.um)), index.um_paths(Path(a.dest))
    keep = {}
    for d in index.UM_DOMAINS:  # read and check every domain before writing any
        keep[d] = [x for x in src[d].read_text(encoding="utf-8").splitlines() if x.strip()
                   and json.loads(x)["method"] == a.arm]  # fmt: skip
        if not keep[d]:
            print(f"WARNING {d}: no {a.arm} units in {src[d]} (the domain is left out)")
    if not any(keep.values()):
        raise SystemExit(f"no {a.arm} units in any domain of {a.um}")
    for d, rows in keep.items():
        if not rows:
            continue
        dest[d].parent.mkdir(parents=True, exist_ok=True)
        dest[d].write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(f"{d}: {len(rows)} {a.arm} units -> {dest[d]}")


if __name__ == "__main__":
    main()
