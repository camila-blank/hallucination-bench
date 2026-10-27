"""``hbench strongreject {grade,run,report}`` (plan 0016), registered through
``cli.EVAL_CLIS`` (plan 0012 D2). Exit codes as elsewhere: 0 ok, 3 judge config / auth error
(raised as ``JudgeConfigError`` and mapped by ``cli.main``), 4 failed calls remain cached."""

import argparse
import os
import sys
from pathlib import Path

from hallucination_bench.cli import _common, _effort, _layers

DEFAULT_ROLLOUTS = "outputs/strongreject/rollouts.jsonl"
DEFAULT_READOUTS = os.environ.get("HBENCH_SR_READOUTS") or "/workspace/hb-readouts-strongreject"
DEFAULT_ARMS = "s3d-rl600,mlnla-rl600,jlens"


def _judge(args: argparse.Namespace):
    from hallucination_bench.evals.strongreject.run import Judge

    return Judge(model=args.judge_model, effort=_effort(args.effort))


def cmd_grade(args: argparse.Namespace) -> int:
    import asyncio

    from hallucination_bench.evals.strongreject import run as R

    res = asyncio.run(
        R.grade_async(
            Path(args.out),
            _judge(args),
            Path(args.rollouts),
            rpm=args.rpm,
            concurrency=args.concurrency,
            limit=args.limit,
            call=not args.rebuild,
        )
    )
    m = res["meta"]
    print(
        f"[grade] prompts={m['n_prompts']} failed_grades={m['n_failed_grades']} "
        f"outcomes={m['outcomes']} spend_total=${m['spend']['total_usd']:.2f}"
    )
    for f, s in res["families"].items():
        print(
            f"  {f:22s} {s['outcomes']} mean_score={s['mean_score']:.3f} "
            f"complied={s['complied_share']:.3f} adopted={s['adoption_share']:.3f}"
        )
    if res["n_failed"] and not args.rebuild:
        print(f"{res['n_failed']} grades failed; re-run to retry", file=sys.stderr)
        return 4
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from hallucination_bench.evals.strongreject import run as R

    arms = R.resolve_arms(Path(args.readouts_root), [a for a in args.arms.split(",") if a])
    judge = _judge(args)
    if args.rebuild:
        summaries = {
            n: R.rebuild(n, p, Path(args.out), judge, allow_missing=args.allow_missing)
            for n, p in arms
        }
    else:
        stages = [s.strip() for s in args.stages.split(",") if s.strip()]
        summaries = R.run(
            arms,
            Path(args.out),
            judge,
            stages=stages,
            layers=_layers(args.layers),
            limit=args.limit,
            rpm=args.rpm,
            concurrency=args.concurrency,
            allow_missing=args.allow_missing,
        )
    n_failed = 0
    for n, s in summaries.items():
        h = s["health"]
        print(
            f"[{n}] layers={s['config']['layers']} status={h['status']} "
            f"spend_total=${s['spend']['total_usd']:.2f}"
        )
        n_failed += R.n_failed(s)
    if n_failed and not args.rebuild:
        print(f"{n_failed} cells with failed calls remain; re-run to retry", file=sys.stderr)
        return 4
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from hallucination_bench.evals.strongreject import report as RP

    root = RP.report(
        Path(args.out),
        _judge(args),
        readouts_root=Path(args.readouts_root),
        arms=[a for a in args.arms.split(",") if a],
        rebuild=not args.no_rebuild,
    )
    print(f"wrote {root / 'report.md'}, {root / 'report.json'}")
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("strongreject", help="plan 0016: lens claims on failed jailbreaks")
    ss = p.add_subparsers(dest="sr_cmd", required=True)

    g = ss.add_parser("grade", help="StrongREJECT rubric on the on-policy rollouts -> outcomes")
    _common(g)
    g.add_argument("--rollouts", default=DEFAULT_ROLLOUTS)
    g.add_argument("--limit", type=int, default=None, help="first N prompts only (smoke)")
    g.add_argument("--rebuild", action="store_true", help="no API calls: outcomes from cache")
    g.set_defaults(fn=cmd_grade)

    r = ss.add_parser("run", help="Stage D (detect) + Stage E (echo) on the readouts")
    _common(r)
    r.add_argument("--readouts-root", default=DEFAULT_READOUTS)
    r.add_argument("--arms", default=DEFAULT_ARMS)
    r.add_argument("--stages", default="detect,echo")
    r.add_argument("--layers", default=None)
    r.add_argument("--limit", type=int, default=None, help="first N items only (smoke)")
    r.add_argument("--allow-missing", action="store_true")
    r.add_argument("--rebuild", action="store_true", help="no API calls: summaries from cache")
    r.set_defaults(fn=cmd_run)

    rp = ss.add_parser("report", help="report.md / report.json over the arms")
    _common(rp)
    rp.add_argument("--readouts-root", default=DEFAULT_READOUTS)
    rp.add_argument("--arms", default=DEFAULT_ARMS)
    rp.add_argument("--no-rebuild", action="store_true", help="read the existing summary.json")
    rp.set_defaults(fn=cmd_report)
