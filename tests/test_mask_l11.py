"""MASK new-layer archive (plan 0019, stopped 2026-10-29): the claim unit used by the strict
pilot, the frozen old-layer numbers (against the git-tracked merged results doc and a snapshot),
unchanged old-layer cache rows, and new-layer grid completeness."""

import json
import re
from pathlib import Path

import pytest

from hallucination_bench import REPO_ROOT
from hallucination_bench.evals.mask import claims as X

OUT = REPO_ROOT / "outputs" / "mask" / "claude-opus-5-5"
L11 = REPO_ROOT / "data" / "mask" / "l11"
READERS = ("s3d-rl600", "mlnla-rl600-all11", "jlens")


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


DOC = REPO_ROOT / "docs" / "results" / "0017_mask_pilot.md"
OLD = (28, 36, 44, 52)


def _published() -> tuple[dict[str, list[float]], dict[int, float]]:
    """The merged (origin/main, PR #29) rev-5 numbers from the git-tracked results doc: HB prompt
    rates per arm at L28/36/44/52 and the LOAO prompt transfer AUROC per layer."""
    text = DOC.read_text()
    sec = text[text.index("### Diagnostic: verified lie-claim") : text.index("### Item-level")]
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
    assert set(hb) == set(READERS) and sorted(loao) == list(OLD)
    return hb, loao


@pytest.mark.skipif(not (OUT / "report.json").exists(), reason="no report on this box")
def test_old_layers_match_the_git_tracked_published_numbers():
    hb, loao = _published()
    rep = json.loads((OUT / "report.json").read_text())
    for arm, rates in hb.items():
        for layer, want in zip(OLD, rates, strict=True):
            g = rep["gate"]["arms"][arm][f"L{layer:03d}:prompt"]["9"]["honest_resisted"]
            assert round(g["readout_rate"], 4) == want, (arm, layer)
    for layer, want in loao.items():
        assert round(rep["probe"]["layers"][str(layer)]["prompt"]["transfer_loao"], 3) == want


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
    for arm in READERS:
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


def _strict_pilot():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "strict_pilot", REPO_ROOT / "scripts" / "mask" / "strict_pilot.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_strict_parse_singleton_fallback():
    """2026-10-29: Opus / Sonnet echo the span text in ``mention`` instead of ``m0``; 8037 valid
    singleton verdicts were dropped until a lone entry for a lone candidate was accepted."""
    S = _strict_pilot()

    def e(mention, cls="persona_text"):
        return {"mention": mention, "class": cls, "reason": "r"}

    ok = {"class": "persona_text", "reason": "r"}
    # singleton: echoed span text is accepted
    assert S.parse({"mentions": [e("repeated blame-shifting framing")]}, 1) == [ok]
    assert S.parse({"mentions": [e("m0")]}, 1) == [ok]
    # singleton with two entries: only an explicit id counts, no guessing
    assert S.parse({"mentions": [e("echo"), e("m0")]}, 1) == [ok]
    assert S.parse({"mentions": [e("echo"), e("other")]}, 1) == [None]
    # batches keep the strict id rule (an echoed entry fails over to a singleton retry)
    assert S.parse({"mentions": [e("echo"), e("m1")]}, 2) == [None, ok]
    # the class must still be valid
    assert S.parse({"mentions": [e("echo", "bogus")]}, 1) == [None]
    assert S.parse(None, 1) == [None]
