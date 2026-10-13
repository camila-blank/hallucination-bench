"""``hbench run|calibrate|report|consistency-v2|pilot-dump|pilot-dump-v2|units-v2``."""

import argparse
import json
import sys
from pathlib import Path

from hallucination_bench.llm import DEFAULT_EFFORT, JudgeConfigError

DEFAULT_MODEL = "claude-opus-5-5"
READOUTS_FILE = "hallucination.jsonl"


def _layers(s: str | None) -> list[int] | None:
    return [int(x) for x in s.split(",") if x.strip()] if s else None


def _effort(s: str) -> str | None:
    return None if s in ("", "none", "default") else s


def resolve_arms(args: argparse.Namespace) -> list[tuple[str, Path]]:
    arms: list[tuple[str, Path]] = []
    for a in args.arm or []:
        if "=" not in a:
            raise SystemExit(f"--arm expects name=path, got {a!r}")
        name, path = a.split("=", 1)
        arms.append((name, Path(path)))
    if args.readouts_root:
        root = Path(args.readouts_root)
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            f = d / READOUTS_FILE
            if f.exists():
                arms.append((d.name, f))
    if args.arms:
        keep = set(args.arms.split(","))
        unknown = keep - {n for n, _ in arms}
        if unknown:
            raise SystemExit(f"--arms names not found: {sorted(unknown)}")
        arms = [(n, p) for n, p in arms if n in keep]
    names = [n for n, _ in arms]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate arm names: {names}")
    if not arms:
        raise SystemExit("no arms: pass --arm name=path and/or --readouts-root DIR")
    for n, p in arms:
        if not p.exists():
            raise SystemExit(f"[{n}] readouts file not found: {p}")
    return arms


def _models(args: argparse.Namespace):
    from hallucination_bench.evals.wsbench_chat.run import Models

    return Models(
        judge=args.judge_model,
        summarizer=args.summarizer_model,
        classifier=args.classifier_model,
        effort=_effort(args.effort),
        map_model=getattr(args, "map_model", None),
        map_anchor=getattr(args, "anchor", None) or "none",
        eval=getattr(args, "eval", None) or "wsbench_chat",
    )


def _stages(args: argparse.Namespace) -> list[str]:
    """``--stages`` (default: every stage); ``--skip-classify`` keeps its plan-0001 meaning
    (summary, span and verify only) when ``--stages`` is not given."""
    from hallucination_bench.evals.wsbench_chat.run import STAGES, expand_stages

    if args.stages:
        sel = [x.strip() for x in args.stages.split(",") if x.strip()]
    elif args.skip_classify:
        sel = ["summary", "span", "verify"]
    else:
        sel = list(STAGES)
    try:
        expand_stages(sel, anchor=getattr(args, "anchor", None) or "none")
    except ValueError as e:
        raise SystemExit(str(e)) from None
    return sel


def cmd_run(args: argparse.Namespace) -> int:
    from hallucination_bench.evals.wsbench_chat import run as wrun

    arms = resolve_arms(args)
    models = _models(args)
    out = Path(args.out)
    if args.rebuild:
        for n, p in arms:
            s = wrun.rebuild_arm(n, p, out, models)
            _print_summary(n, s)
        return 0
    summaries = wrun.run_arms(
        arms,
        out,
        models,
        layers=_layers(args.layers),
        limit=args.limit,
        rpm=args.rpm,
        concurrency=args.concurrency,
        allow_missing=args.allow_missing,
        classify=not args.skip_classify,
        show_prompts=args.show_prompts,
        stages=_stages(args),
    )
    for n, s in summaries.items():
        _print_summary(n, s)
    # Failed calls are cached as failures and re-queued by the next run; a non-zero exit lets
    # the driver re-run the stage (scripts/run_all_arms.sh) instead of moving on.
    n_failed = sum(
        v for s in summaries.values() for k, v in s["health"].items() if k.startswith("n_failed_")
    )
    if n_failed:
        print(f"{n_failed} failed calls remain; re-run to retry them", file=sys.stderr)
        return 4
    return 0


def _print_summary(name: str, s: dict) -> None:
    r = s["overall"]["rates"]
    h = s["health"]

    def f(k: str) -> str:
        v = r.get(k)
        return "—" if v is None else f"{v:.3f}"

    failed = {k: h[f"n_failed_{k}"] for k in ("summary", "span", "verify", "classify", "map")}
    print(
        f"[{name}] layers={s['config']['judged_layers']} judged={h['n_judged_cells']} "
        f"pending={h['n_pending_cells']} failed={failed} "
        f"| v5c={f('hallucination_rate')} concept={f('concept_rate')} detail={f('detail_rate')} "
        f"strict={f('strict_hallucination_rate')} unsupported={f('unsupported_rate')} "
        f"unclassified_spans={h['n_unclassified']} spend_total=${s['spend']['total_usd']:.2f}",
        flush=True,
    )


def cmd_calibrate(args: argparse.Namespace) -> int:
    from hallucination_bench import calibrate

    if args.taxonomy == "v2":
        return _calibrate_v2(args)
    model = args.classifier_model or args.judge_model
    res = calibrate.run(
        args.split,
        model=model,
        effort=_effort(args.effort),
        out_root=Path(args.out),
        repeats=args.repeats,
        rpm=args.rpm,
        concurrency=args.concurrency,
        cases_path=Path(args.cases) if args.cases else None,
    )
    print(calibrate.render_markdown(res))
    sp = res["spend_this_run"]
    print(f"\nspend this run: ${sp['usd']:.4f} ({sp['calls']} calls)")
    if args.split == "heldout":
        print(f"GATE {'PASSED' if res['gate_passed'] else 'FAILED'}")
    return 0


def _calibrate_v2(args: argparse.Namespace) -> int:
    from hallucination_bench import calibrate_v2

    res = calibrate_v2.run(
        args.split,
        model=args.map_model or args.judge_model,
        effort=_effort(args.effort),
        out_root=Path(args.out),
        repeats=args.repeats,
        rpm=args.rpm,
        concurrency=args.concurrency,
        cases_file=Path(args.cases) if args.cases else None,
        anchor=args.anchor,
    )
    print(calibrate_v2.render_markdown(res))
    sp = res["spend_this_run"]
    print(f"\ncoverage: {json.dumps(res['coverage'])}")
    print(f"spend this run: ${sp['usd']:.4f} ({sp['calls']} calls)")
    if args.split == "heldout":
        print(f"GATE {'PASSED' if res['gate_passed'] else 'FAILED'}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from hallucination_bench import report

    if args.view == "v2":
        from hallucination_bench import report_v2

        root = report_v2.report(
            Path(args.out),
            args.judge_model,
            rebuild=not args.no_rebuild,
            audit_n=args.audit_n if args.audit_n is not None else 10,
            seed=args.seed,
            verdict_n=args.audit_verdict_n,
            eval_name=args.eval,
        )
        print(
            f"wrote {root / 'report_v2.md'}, {root / 'report_v2.json'}, "
            f"{root / 'probe_candidates.jsonl'}, {root / 'audit_samples_v2.md'}"
        )
        return 0

    root = report.report(
        Path(args.out),
        args.judge_model,
        rebuild=not args.no_rebuild,
        audit_n=args.audit_n if args.audit_n is not None else 15,
        seed=args.seed,
        eval_name=args.eval,
    )
    print(f"wrote {root / 'report.md'}, {root / 'report.json'}, {root / 'audit_samples.md'}")
    return 0


def cmd_consistency_v2(args: argparse.Namespace) -> int:
    from hallucination_bench import consistency_v2

    if args.rescore:
        res = consistency_v2.rescore(Path(args.out), args.map_model or args.judge_model)
        print(consistency_v2.render_markdown(res))
        print("\nre-scored from the cached r0/r1 cells: no API calls")
        return 0
    if not args.arms:
        raise SystemExit("--arms is required unless --rescore")
    res = consistency_v2.run(
        args.arms,
        readouts_root=Path(args.readouts_root),
        out_root=Path(args.out),
        models=_models(args),
        n=args.n,
        seed=args.seed,
        rpm=args.rpm,
        concurrency=args.concurrency,
    )
    print(consistency_v2.render_markdown(res))
    sp = res["spend_this_run"]
    print(f"\nspend this run: ${sp['usd']:.4f} ({sp['calls']} calls)")
    return 0


def cmd_pilot_dump(args: argparse.Namespace) -> int:
    from hallucination_bench import report
    from hallucination_bench.calibrate import validate_case

    cases = report.pilot_dump(
        Path(args.out),
        args.judge_model,
        args.arms.split(","),
        per_stratum=args.per_stratum,
        seed=args.seed,
        layers=_layers(args.layers),
    )
    errs = [e for c in cases for e in validate_case(c, require_gold=False)]
    if errs:
        print("warning: " + "; ".join(errs[:10]), file=sys.stderr)
    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    n_sp = sum(len(c["spans"]) for c in cases)
    n_sampled = sum(s["sampled"] for c in cases for s in c["spans"])
    print(f"wrote {len(cases)} cases ({n_sampled} sampled spans, {n_sp} spans total) to {dest}")
    return 0


def cmd_pilot_dump_v2(args: argparse.Namespace) -> int:
    from hallucination_bench import report_v2
    from hallucination_bench.calibrate_v2 import validate_case

    cases = report_v2.pilot_dump_v2(
        Path(args.out),
        args.judge_model,
        args.arms.split(","),
        per_stratum=args.per_stratum,
        seed=args.seed,
        layers=_layers(args.layers),
    )
    errs = [e for c in cases for e in validate_case(c, require_gold=False)]
    if errs:
        print("warning: " + "; ".join(errs[:10]), file=sys.stderr)
    _write_jsonl(Path(args.dest), cases)
    n_units = sum(len(c["gold"]["units"]) for c in cases)
    print(f"wrote {len(cases)} cells ({n_units} units to label) to {args.dest}")
    return 0


def cmd_units_v2(args: argparse.Namespace) -> int:
    """Fill each case's ``gold.units`` with the production units (leaf null) for labelling."""
    from hallucination_bench.calibrate_v2 import skeleton

    src = Path(args.cases)
    cases = [json.loads(x) for x in src.read_text(encoding="utf-8").splitlines() if x.strip()]
    out = [skeleton(c) for c in cases]
    _write_jsonl(Path(args.dest), out)
    print(f"wrote {len(out)} cases ({sum(len(c['gold']['units']) for c in out)} units)")
    return 0


def _write_jsonl(dest: Path, rows: list[dict]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        for c in rows:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out", default="outputs", help="output root (default: outputs)")
    p.add_argument("--judge-model", default=DEFAULT_MODEL)
    p.add_argument(
        "--effort",
        default=DEFAULT_EFFORT,
        help="Anthropic output_config.effort (default low; 'none' omits)",
    )
    p.add_argument("--rpm", type=float, default=1500.0)
    p.add_argument("--concurrency", type=int, default=64)


def _eval(p: argparse.ArgumentParser) -> None:
    from hallucination_bench.evals import EVALS

    p.add_argument(
        "--eval",
        choices=sorted(EVALS),
        default="wsbench_chat",
        help="which bank / output root <out>/<eval>/ (story: plan 0007; default wsbench_chat)",
    )


def _anchor(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--anchor",
        choices=("none", "v5c"),
        default="none",
        help="claim map: none = single-call (default; no established spans, no verify); "
        "v5c = plan rev 12 (v5c wrong spans passed as ESTABLISHED units)",
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="hbench", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="judge arms (summarise -> span -> verify -> map -> classify)")
    _common(r)
    r.add_argument("--arm", action="append", help="name=path/to/hallucination.jsonl (repeatable)")
    r.add_argument("--readouts-root", help="DIR with <arm>/hallucination.jsonl subdirectories")
    r.add_argument("--arms", help="comma-separated arm names to keep")
    r.add_argument("--layers", help="comma-separated layers (each arm judges those it has)")
    r.add_argument(
        "--limit",
        type=int,
        default=None,
        help="judge only the first N non-empty cells per arm x layer (smoke)",
    )
    r.add_argument("--summarizer-model", default=None)
    r.add_argument("--classifier-model", default=None)
    r.add_argument("--map-model", default=None, help="claim-map (:M) judge (default: judge)")
    _anchor(r)
    _eval(r)
    r.add_argument(
        "--stages",
        default=None,
        help="comma-separated stages that may call the API: summary,span,verify,classify,map "
        "(default: all; dependencies are added: map runs summary first if missing, and span "
        "only under --anchor v5c; single-call map also calls span on a seeded 10%% item "
        "subsample for the v5c comparison)",
    )
    r.add_argument("--allow-missing", action="store_true")
    r.add_argument(
        "--rebuild",
        action="store_true",
        help="regenerate cells.jsonl/summary.json from the cache (no API calls)",
    )
    r.add_argument(
        "--skip-classify",
        action="store_true",
        help="summary + span + verify only (pilot); ignored when --stages is given",
    )
    r.add_argument(
        "--show-prompts",
        type=int,
        default=0,
        metavar="K",
        help="write the prompts of the first K cells per arm x layer to <arm>/prompts_sample.txt",
    )
    r.set_defaults(fn=cmd_run)

    c = sub.add_parser("calibrate", help="score the classifier on calibration cases")
    _common(c)
    c.add_argument("--split", choices=("dev", "heldout"), required=True)
    c.add_argument(
        "--taxonomy",
        choices=("v1", "v2"),
        default="v1",
        help="v1: the taxonomy-v1 classifier on {split}.jsonl; v2: the claim map on "
        "{split}_v2.jsonl",
    )
    c.add_argument("--classifier-model", default=None)
    c.add_argument("--map-model", default=None)
    _anchor(c)
    c.add_argument("--repeats", type=int, default=2)
    c.add_argument("--cases", default=None, help="override the cases file")
    c.set_defaults(fn=cmd_calibrate)

    rp = sub.add_parser("report", help="cross-arm report + audit samples")
    rp.add_argument("--out", default="outputs")
    rp.add_argument("--judge-model", default=DEFAULT_MODEL)
    rp.add_argument("--no-rebuild", action="store_true")
    rp.add_argument(
        "--view", choices=("v1", "v2"), default="v1", help="v2: the claim-map report (plan 0002)"
    )
    rp.add_argument("--audit-n", type=int, default=None, help="per leaf; default 15 (v1) / 10 (v2)")
    rp.add_argument("--audit-verdict-n", type=int, default=20, help="v2: audit samples per verdict")
    rp.add_argument("--seed", type=int, default=0)
    _eval(rp)
    rp.set_defaults(fn=cmd_report)

    cs = sub.add_parser(
        "consistency-v2", help="claim-map judge consistency: two fresh :M calls per sampled cell"
    )
    _common(cs)
    cs.add_argument("--arms", default=None, help="arm:layer,... e.g. s3d-rl600:44,jlens:44")
    cs.add_argument(
        "--rescore",
        action="store_true",
        help="re-score the cached r0/r1 cells of this prompt version (no API calls)",
    )
    cs.add_argument("--n", type=int, default=50, help="cells per arm:layer (default 50)")
    cs.add_argument("--seed", type=int, default=0)
    cs.add_argument(
        "--readouts-root",
        default="/workspace/wsbench-paper-data/readouts",
        help="DIR with <arm>/hallucination.jsonl",
    )
    cs.add_argument("--summarizer-model", default=None)
    cs.add_argument("--classifier-model", default=None)
    cs.add_argument("--map-model", default=None)
    cs.set_defaults(fn=cmd_consistency_v2)

    pd = sub.add_parser("pilot-dump", help="sample pilot spans as unlabelled calibration cases")
    pd.add_argument("--out", default="outputs")
    pd.add_argument("--judge-model", default=DEFAULT_MODEL)
    pd.add_argument("--arms", required=True)
    pd.add_argument("--layers", default=None)
    pd.add_argument("--per-stratum", type=int, default=5, help="spans per arm x grounding")
    pd.add_argument("--seed", type=int, default=0)
    pd.add_argument("--dest", required=True)
    pd.set_defaults(fn=cmd_pilot_dump)

    p2 = sub.add_parser(
        "pilot-dump-v2", help="dump real cells with production units for v2 hand-labelling"
    )
    p2.add_argument("--out", default="outputs")
    p2.add_argument("--judge-model", default=DEFAULT_MODEL)
    p2.add_argument("--arms", required=True)
    p2.add_argument("--layers", default=None)
    p2.add_argument("--per-stratum", type=int, default=3, help="cells per arm x v5c cell class")
    p2.add_argument("--seed", type=int, default=0)
    p2.add_argument("--dest", required=True)
    p2.set_defaults(fn=cmd_pilot_dump_v2)

    u2 = sub.add_parser(
        "units-v2", help="fill v2 cases' gold.units with the production unit split (leaf null)"
    )
    u2.add_argument("--cases", required=True)
    u2.add_argument("--dest", required=True)
    u2.set_defaults(fn=cmd_units_v2)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except JudgeConfigError as e:
        print(f"JUDGE CONFIG ERROR: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
