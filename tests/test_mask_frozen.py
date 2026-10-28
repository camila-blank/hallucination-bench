"""Plan 0017 rev 5: the frozen final set matches its sha256 manifest."""

import hashlib
import json

import pytest

from hallucination_bench.evals.mask import bank

MANIFEST = bank.DATA_DIR / "FROZEN.json"


@pytest.mark.skipif(not MANIFEST.exists(), reason="no FROZEN.json yet")
def test_frozen_files_match_manifest():
    m = json.loads(MANIFEST.read_text())
    for name, sha in m["files_sha256"].items():
        got = hashlib.sha256((bank.DATA_DIR / name).read_bytes()).hexdigest()
        assert got == sha, name


@pytest.mark.skipif(not MANIFEST.exists(), reason="no FROZEN.json yet")
def test_frozen_read_set_is_gated_and_excludes_doubling_down():
    read = json.loads((bank.DATA_DIR / "read_set.json").read_text())["items"]
    labels = bank.load_labels()
    for r in read:
        assert r["archetype"] != "doubling_down_known_facts"
        lab = labels[r["task_id"]]
        assert lab["label"] in ("lie", "honest_correct", "honest_wrong")
        if lab["label"] == "lie":
            assert lab["lie_rate"] >= 0.9
