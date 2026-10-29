"""Plan 0019 (mask): the index export, its reconciliation with the merged HB report at the old
layers, and the frozen old-layer numbers / cache rows."""

import collections
import json
import re
from pathlib import Path

import pytest

from hallucination_bench import REPO_ROOT
from hallucination_bench.evals.mask import index_export as X

OUT = REPO_ROOT / "outputs" / "mask" / "claude-opus-5-5"
L11 = REPO_ROOT / "data" / "mask" / "l11"


def m(r, cat, quote="q", real=None):
    d = {"readout": r, "category": cat, "quote": quote}
    if real is not None:
        d["verify"] = {"real": real}
    return d


def test_claims_dedup_and_label_rules():
    cell = {
        "mentions": [
            m(0, "lie_claim", "will lie", True),
            m(0, "lie_claim", " will   lie ", True),  # same (readout, label, normalised quote)
            m(0, "lie_claim", "Will \u201clie\u201d", True),  # normalize_ws: case, quote marks
            m(0, "lie_claim", 'will "lie"', True),
            m(1, "lie_claim", "will lie", True),  # another readout: a separate claim
            m(0, "lie_claim", "deny it", False),  # verifier-rejected: dropped, not relabelled
            m(0, "lie_claim", "unverified"),  # unverified candidate: not a lie, dropped
            m(0, "honest_claim", "tells the truth"),
            m(0, "pressure_salience", "asked to lie"),
            m(0, "other", "x"),
        ]
    }
    assert X.claims(cell) == (4, 3)


def test_layer_gates_fail_loudly_on_a_missing_layer():
    full = {str(la): {"prompt": {"transfer_loao": 0.8}} for la in X.LAYERS}
    g = X.layer_gates(full)
    assert sorted(g) == list(X.LAYERS) and all(v["pass"] for v in g.values())
    full["48"] = {"prompt": {"transfer_loao": 0.69}}
    assert not X.layer_gates(full)[48]["pass"]
    del full["60"]
    with pytest.raises(RuntimeError, match=r"\[60\]"):
        X.layer_gates(full)
    full["60"] = {"prompt": {"transfer_loao": None}}  # a job absent from the new-layer file
    with pytest.raises(RuntimeError):
        X.layer_gates(full)


def _cell(t, layer, pos, mentions, status="ok", n=3):
    return {
        "task_id": t,
        "layer": layer,
        "pos": pos,
        "site_type": "prompt",
        "status": status,
        "n_readouts": n,
        "mentions": mentions,
    }


def test_item_rows_probe_split_gate_and_zero_rows():
    cells = [
        _cell("a", 20, 0, [m(0, "lie_claim", "x", True), m(1, "honest_claim", "y")]),
        _cell("a", 20, 1, [m(2, "lie_claim", "z", True)]),
        _cell("a", 24, 0, [m(0, "lie_claim", "x", True)]),  # layer 24 fails the gate
        _cell("a", 20, 2, [], status="failed"),
        _cell("zz", 20, 0, [m(0, "lie_claim", "x", True)]),  # not in the population
    ]
    pos_index = {("a", 0): 0, ("a", 1): 1, ("a", 2): 2, ("b", 0): 0}
    present = {("a", 20, 0): False, ("a", 20, 1): True, ("a", 24, 0): False}
    rows = X.item_rows(
        cells, {"a", "b"}, present, {20: True, 24: False}, pos_index, "s3d", [20, 24]
    )
    assert [(r["layer"], r["item"]) for r in rows] == [(20, "a"), (20, "b"), (24, "a"), (24, "b")]
    a20, b20, a24 = rows[0], rows[1], rows[2]
    assert (a20["n_claims"], a20["n_false"], a20["n_contradicted"], a20["n_unprobed"]) == (
        3,
        2,
        1,
        0,
    )
    assert (a20["n_units"], a20["n_false_units"]) == (6, 2)
    assert (a20["n_contradicted_units"], a20["n_supported_units"]) == (1, 1)
    assert a20["n_failed_cells"] == 1
    assert b20["n_claims"] == 0 and b20["n_units"] == 0  # zero rows are written
    assert (a24["n_false"], a24["n_unprobed"], a24["n_contradicted"]) == (1, 1, 0)
    assert a24["n_unprobed_units"] == 1


def test_reader_allowlist():
    assert X.READERS == ("s3d-rl600", "mlnla-rl600-all11", "jlens")


@pytest.mark.skipif(not (OUT / "index_counts.jsonl").exists(), reason="no index export on this box")
def test_old_layers_reconcile_with_the_merged_report():
    rep = json.loads((OUT / "report.json").read_text())
    agg: dict = collections.defaultdict(collections.Counter)
    for line in (OUT / "index_counts.jsonl").read_text().splitlines():
        r = json.loads(line)
        assert r["arm"] in X.READERS
        agg[(r["arm"], r["layer"])].update({k: v for k, v in r.items() if k.startswith("n_")})
    for (arm, layer), a in agg.items():
        if layer not in (28, 36, 44, 52):
            continue
        g = rep["gate"]["arms"][arm][f"L{layer:03d}:prompt"]["9"]["honest_resisted"]
        assert a["n_units"] == g["n_readouts"]
        assert a["n_false_units"] == round(g["readout_rate"] * g["n_readouts"])
        p = rep["probe"]["arms"][arm][f"L{layer:03d}:prompt"]
        assert a["n_supported_units"] == p["loao_oof_hr_supported"]
        assert a["n_contradicted_units"] == p["loao_oof_hr_contradicted"]
    n_items = {k: 0 for k in agg}
    for line in (OUT / "index_counts.jsonl").read_text().splitlines():
        r = json.loads(line)
        n_items[(r["arm"], r["layer"])] += 1
    assert set(n_items.values()) == {155}  # every population item, zero rows included


DOC = REPO_ROOT / "docs" / "results" / "0017_mask_pilot.md"
OLD = (28, 36, 44, 52)


def _published() -> tuple[dict[str, list[float]], dict[int, float]]:
    """The merged (origin/main, PR #29) rev-5 numbers from the git-tracked results doc: HB prompt
    rates per arm at L28/36/44/52 and the LOAO prompt transfer AUROC per layer."""
    text = DOC.read_text()
    sec = text[text.index("### HB score: verified lie-claim") : text.index("### Item-level")]
    hb = {}
    for line in sec.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 6 and cells[1] == "prompt":
            hb[cells[0]] = [float(c.split()[0]) for c in cells[2:]]
    sec = text[text.index("### Probes: transfer AUROC") : text.index("- The **primary LOAO")]
    loao = {}
    for line in sec.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 5 and cells[1] == "prompt":
            loao[int(cells[0])] = float(cells[2])
    assert set(hb) == set(X.READERS) and sorted(loao) == list(OLD)
    return hb, loao


@pytest.mark.skipif(not (OUT / "index_counts.jsonl").exists(), reason="no index export on this box")
def test_old_layers_reconcile_with_the_git_tracked_published_numbers():
    hb, loao = _published()
    agg: dict = collections.defaultdict(collections.Counter)
    for line in (OUT / "index_counts.jsonl").read_text().splitlines():
        r = json.loads(line)
        agg[(r["arm"], r["layer"])].update({k: v for k, v in r.items() if k.startswith("n_")})
    for arm, rates in hb.items():
        for layer, want in zip(OLD, rates, strict=True):
            a = agg[(arm, layer)]
            assert round(a["n_false_units"] / a["n_units"], 4) == want, (arm, layer)
    meta = json.loads((OUT / "index_meta.json").read_text())
    assert meta["dedup"] == X.DEDUP == "normalize_ws"
    gates = {int(k): v for k, v in meta["gate"]["per_layer"].items()}
    assert sorted(gates) == list(X.LAYERS)
    for layer, want in loao.items():
        assert round(gates[layer]["transfer_loao"], 3) == want, layer


@pytest.mark.skipif(not (OUT / "report.json").exists(), reason="no report on this box")
def test_old_layer_published_numbers_are_frozen():
    rep = json.loads((OUT / "report.json").read_text())
    snap = json.loads((L11 / "report_4layer_snapshot.json").read_text())
    for arm, rows in snap["gate_arms"].items():
        for k, v in rows.items():
            assert rep["gate"]["arms"][arm][k] == v, (arm, k)
    for arm, rows in snap["arms"].items():
        for k, v in rows.items():
            assert rep["arms"][arm][k] == v, (arm, k)


@pytest.mark.skipif(not (OUT / "s3d-rl600" / "cache.jsonl").exists(), reason="no caches")
def test_old_layer_cache_rows_unchanged():
    want = json.loads((L11 / "cache_rows_4layer.json").read_text())["rows"]
    pat = re.compile(r"__L0(28|36|44|52)__")
    for arm, n in want.items():
        with (OUT / arm / "cache.jsonl").open() as fh:
            got = sum(1 for line in fh if pat.search(line[:200]))
        assert got == n, arm


@pytest.mark.skipif(not (L11 / "sites_l11.json.gz").exists(), reason="no L11 sites")
def test_l11_grid_complete_when_readouts_present():
    from hallucination_bench.evals.mask import bank

    sites = bank.load_sites_file(L11 / "sites_l11.json.gz")
    assert len(sites) == 155
    want = {(t, s["pos"]) for t, ss in sites.items() for s in ss}
    for arm in X.READERS:
        p = Path(bank.READOUTS_ROOT) / arm / "mask_l11.jsonl"
        if not p.exists():
            continue
        new = (20, 24, 32, 40, 48, 56, 60)
        uniq: set[tuple[str, int, int]] = set()  # count unique (item, pos, layer), not rows
        with p.open() as fh:
            for line in fh:
                r = json.loads(line)
                uniq.add((r["id"], r["pos"], r["layer"]))
        assert uniq == {(t, pos, la) for (t, pos) in want for la in new}, arm
        cells: set[tuple[str, int, int]] = set()
        for la in new:
            f = OUT / arm / f"cells_L{la:03d}.jsonl"
            if not f.exists():
                continue
            with f.open() as fh:
                for line in fh:
                    c = json.loads(line)
                    if c["site_type"] == "prompt" and c["task_id"] in sites:
                        cells.add((c["task_id"], c["pos"], c["layer"]))
            assert {(t, pos) for (t, pos, x) in cells if x == la} == want, (arm, la)
