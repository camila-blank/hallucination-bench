"""A non-default ``--out`` gets its own data / probes dirs (pilot isolation); default unchanged."""

import argparse
from pathlib import Path

from hallucination_bench.evals.sibling_gate import stages as S
from hallucination_bench.evals.sibling_gate import units as U


def test_configure_routes_pilot_out_and_keeps_default(tmp_path):
    d0, p0 = S.DATA_DIR, S.PROBES_DIR
    try:
        S._configure(argparse.Namespace(out=str(U.OUT)))
        assert (S.DATA_DIR, S.PROBES_DIR) == (d0, p0)
        S._configure(argparse.Namespace(out=str(tmp_path)))
        assert S.DATA_DIR == Path(tmp_path) / "data"
        assert S.PROBES_DIR == Path(tmp_path) / "probes"
    finally:
        S.DATA_DIR, S.PROBES_DIR = d0, p0


def test_min_class_pairs_flag_defaults_to_all():
    import argparse as ap

    sub = ap.ArgumentParser().add_subparsers()
    S.add_parser(sub)
    a = sub.choices["sibling-gate"].parse_args(["concepts"])
    assert a.min_class_pairs == 0 and a.probe_acts == "acts-sibgate-probe"


def test_fit_layer_error_branch_returns_slug():
    """Every row ``fit_layer`` returns carries its slug (the local entrypoint pops it)."""
    src = (U.REPO_ROOT / "scripts" / "sibling_gate" / "probes_modal.py").read_text()
    appends = [ln.strip() for ln in src.splitlines() if "results.append(" in ln]
    assert appends and all('"slug"' in ln for ln in appends)


def test_capture_manifest_truncates_template_after_reply():
    src = (U.REPO_ROOT / "src/hallucination_bench/evals/sibling_gate/stages.py").read_text()
    assert 'text = text[: reply_at + len(t["assistant"])]' in src


def test_score_eval_windows_remap_l42_acts_dir():
    src = (U.REPO_ROOT / "scripts" / "sibling_gate" / "probes_modal.py").read_text()
    assert "/acts-jlens-pr-L42/" in src and "acts-hallucination-chat-L42cap" in src
