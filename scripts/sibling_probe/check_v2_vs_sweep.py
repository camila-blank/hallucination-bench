"""Offline check (plan 0003 rev 11): the pipeline's V2 (and V0) validity / transfer AUROCs equal the
recipe sweep's (max abs diff < 1e-3).

    uv run python scripts/sibling_probe/check_v2_vs_sweep.py [--probes-dir outputs/sibling/probes]
"""

import argparse
import json
import sys
from pathlib import Path

from hallucination_bench.evals.sibling_probe import probe_io as PIO

ap = argparse.ArgumentParser()
ap.add_argument("--probes-dir", default=str(PIO.PROBES_DIR))
ap.add_argument("--sweep-dir", default=str(PIO.SWEEP_DIR))
ap.add_argument("--tol", type=float, default=1e-3)
a = ap.parse_args()
bad = False
for recipe in PIO.recipe_dirs(Path(a.probes_dir)):
    r = PIO.compare_with_sweep(Path(a.probes_dir), Path(a.sweep_dir), recipe)
    ok = r["max_abs_diff_validity"] < a.tol and r["max_abs_diff_transfer"] < a.tol
    bad |= not ok
    print(json.dumps(r), "OK" if ok else "MISMATCH")
sys.exit(1 if bad else 0)
