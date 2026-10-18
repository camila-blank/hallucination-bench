"""``hbench sibling judge|probes|score|report`` (plan 0003 stage 2), registered by
``hallucination_bench.cli`` (moved out of it by plan 0012 D2)."""

import argparse
import json
from pathlib import Path

from hallucination_bench.cli import DEFAULT_MODEL, _effort
from hallucination_bench.llm import DEFAULT_EFFORT


def cmd_sibling(args: argparse.Namespace) -> int:
    """Plan 0003 stage 2: ``hbench sibling judge|probes|score|report``."""
    import asyncio
    import subprocess

    from hallucination_bench.evals.sibling_probe import probe_io

    if args.sib == "judge":
        from hallucination_bench.evals.sibling_probe import judge as J

        arms = [a for a in args.arms.split(",") if a]
        unknown = set(arms) - set(J.ARMS)
        if unknown:
            raise SystemExit(f"unknown arms {sorted(unknown)} (known: {', '.join(J.ARMS)})")
        runs = asyncio.run(
            J.judge_arms(
                arms,
                rebuild=args.rebuild,
                rpm=args.rpm,
                concurrency=args.concurrency,
                models=J.JudgeModels(judge=args.judge_model, effort=_effort(args.effort)),
                readouts_root=Path(args.readouts_root),
                out_root=Path(args.out),
                limit=args.limit,
                summarize_all=not args.no_summarize_all,
            )
        )
        bad = sum(
            n for r in runs.values() for k, n in r["counts"].items() if k.startswith("status:fail")
        )
        if bad:
            print(f"{bad} cells with cached failures (re-run to retry)")
            return 4
        return 0
    if args.sib == "probes":
        out = Path(args.out_dir)
        if args.prepare:
            doc = probe_io.prepare_inputs(out / "inputs.json")
            print(
                f"[probes] inputs -> {out / 'inputs.json'}: {len(doc['items'])} items, "
                f"{len(doc['probe_texts'])} probe texts"
            )
        if args.launch:
            cmd = [
                *"uvx --python 3.12 modal@latest run".split(),
                "scripts/sibling_probe/probes_modal.py",
                "--inputs",
                str(out / "inputs.json"),
                "--eval-acts",
                args.eval_acts,
                "--probe-acts",
                args.probe_acts,
                "--run",
                args.run,
                "--out-dir",
                str(out),
            ]
            print("$ " + " ".join(cmd), flush=True)
            rc = subprocess.run(cmd, cwd=probe_io.REPO_ROOT).returncode
            if rc:
                return rc
        if args.fetch:
            cmd = [
                *"uvx --python 3.12 modal@latest volume get --force sibling-probes".split(),
                f"/{args.run}",
                str(out / "pieces"),
            ]
            rc = subprocess.run(cmd).returncode
            if rc:
                return rc
        if args.launch or args.fetch or args.merge:
            for f in probe_io.merge_pieces(out):
                print(f"[probes] merged {f}")
        summ = out / "summary.json"
        if summ.exists():
            print(summ.read_text())
        return 0
    if args.sib == "score":
        from hallucination_bench.evals.sibling_probe import score as S

        summary = S.load_and_score(
            arms=[a for a in args.arms.split(",") if a],
            judge_root=Path(args.judge_root),
            probes_dir=Path(args.probes_dir),
            n_boot=args.n_boot,
            seed=args.seed,
        )
        path = S.write_summary(summary, Path(args.out))
        for rec, rs in summary["recipes"].items():
            for r in rs["verdict"]:
                print(f"  [{rec}] {r['arm']:12s} {r['subset']:22s} -> {r['verdict']}")
        print(f"  recipe sensitivity: {summary['recipe_sensitivity'].get('status')}")
        print(f"[score] -> {path}")
        return 0
    if args.sib == "report":
        from hallucination_bench.evals.sibling_probe import report as R

        res = R.write_report(Path(args.score_dir), Path(args.plots_dir))
        print(json.dumps(res, indent=1))
        return 0
    raise SystemExit(f"unknown sibling command {args.sib}")


def register(sub: argparse._SubParsersAction) -> None:
    sb = sub.add_parser("sibling", help="plan 0003 sibling-concept probes (stage 2)")
    ss = sb.add_subparsers(dest="sib", required=True)
    sj = ss.add_parser("judge", help="prefilter + claim map + scope/coverage calls per arm")
    sj.add_argument("--arms", default="s3d-rl600,nla-rl-L42,jlens")
    sj.add_argument("--readouts-root", default="outputs/sibling/readouts")
    sj.add_argument("--out", default="outputs/sibling/judge")
    sj.add_argument("--judge-model", default=DEFAULT_MODEL)
    sj.add_argument("--effort", default=DEFAULT_EFFORT or "")
    sj.add_argument("--rpm", type=float, default=1500.0)
    sj.add_argument("--concurrency", type=int, default=64)
    sj.add_argument("--limit", type=int, default=None, help="first N items per arm (smoke)")
    sj.add_argument("--rebuild", action="store_true", help="cache only, no API calls")
    sj.add_argument(
        "--no-summarize-all",
        action="store_true",
        help="J-lens: summarise only judged cells (unmentioned then needs a summary: fewer U)",
    )
    sj.set_defaults(fn=cmd_sibling)
    sp = ss.add_parser("probes", help="prepare / launch / fetch the Modal probe job")
    sp.add_argument("--out-dir", default="outputs/sibling/probes")
    sp.add_argument("--prepare", action="store_true", help="write <out-dir>/inputs.json")
    sp.add_argument("--launch", action="store_true", help="run probes_modal.py (blocking)")
    sp.add_argument("--fetch", action="store_true", help="modal volume get the outputs of --run")
    sp.add_argument(
        "--merge", action="store_true", help="merge <out-dir>/pieces into <out-dir>/{v2,v0}"
    )
    sp.add_argument("--run", default="sibling-v3")
    sp.add_argument("--eval-acts", default="acts-sibling-eval")
    sp.add_argument("--probe-acts", default="acts-sibling-probe")
    sp.set_defaults(fn=cmd_sibling)
    sc = ss.add_parser("score", help="M/U tests, hallucination rates, verdict")
    sc.add_argument("--arms", default="s3d-rl600,nla-rl-L42,jlens")
    sc.add_argument("--judge-root", default="outputs/sibling/judge")
    sc.add_argument("--probes-dir", default="outputs/sibling/probes")
    sc.add_argument("--out", default="outputs/sibling/score")
    sc.add_argument("--n-boot", type=int, default=2000)
    sc.add_argument("--seed", type=int, default=0)
    sc.set_defaults(fn=cmd_sibling)
    sr = ss.add_parser("report", help="report.md + plots/sibling_probe/")
    sr.add_argument("--score-dir", default="outputs/sibling/score")
    sr.add_argument("--plots-dir", default="plots/sibling_probe")
    sr.set_defaults(fn=cmd_sibling)
