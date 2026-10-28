"""``hbench mask gt|run|report`` (plan 0017), registered through ``cli.EVAL_CLIS``."""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_MODEL = "claude-opus-5-5"


def _effort(s: str) -> str | None:
    return None if s in ("", "none", "default") else s


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out", default="outputs", help="output root (default: outputs)")
    p.add_argument("--judge-model", default=DEFAULT_MODEL)
    p.add_argument("--effort", default="low")
    p.add_argument("--rpm", type=float, default=1500.0)
    p.add_argument("--concurrency", type=int, default=64)


def cmd_gt(args: argparse.Namespace) -> int:
    from . import bank, gt

    items = bank.load_items()
    rollouts = gt.load_rollouts([Path(p) for p in args.rollouts.split(",")])
    out = Path(args.out) / "mask" / args.judge_model
    cache_path = out / "gt_cache.jsonl"
    effort = _effort(args.effort)
    res = {"n_failed": 0}
    if not args.labels_only:
        res = gt.run_gt(
            items,
            rollouts,
            cache_path,
            model=args.judge_model,
            effort=effort,
            rpm=args.rpm,
            concurrency=args.concurrency,
        )
    lab = gt.build_labels(items, rollouts, cache_path, model=args.judge_model, effort=effort)
    meta = {
        "plan": "0017",
        "judge_model": args.judge_model,
        "effort": effort,
        "mask_version": gt.MASK_VERSION,
        "press_version": gt.PRESS_VERSION,
        "n_items": len(lab["items"]),
        "counts": lab["counts"],
    }
    dest = bank.DATA_DIR / "labels.json"
    dest.write_text(
        json.dumps({"meta": meta, "items": lab["items"]}, indent=1, ensure_ascii=False) + "\n"
    )
    gt.write_rollouts_gz(items, rollouts, bank.DATA_DIR / "rollouts.jsonl.gz")
    print(f"[gt] labels -> {dest}: {lab['counts']}", flush=True)
    if res["n_failed"]:
        print(f"{res['n_failed']} failed calls remain; re-run to retry them", file=sys.stderr)
        return 4
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from . import run as mrun

    arms = mrun.resolve_arms(args.readouts_root, args.arms)
    out = Path(args.out) / "mask" / args.judge_model
    n_failed = mrun.run_mask(
        arms,
        out,
        model=args.judge_model,
        effort=_effort(args.effort),
        rpm=args.rpm,
        concurrency=args.concurrency,
        stages=[s for s in args.stages.split(",") if s],
        layers=[int(x) for x in args.layers.split(",")] if args.layers else None,
        limit=args.limit,
    )
    if n_failed:
        print(f"{n_failed} failed calls remain; re-run to retry them", file=sys.stderr)
        return 4
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from . import report

    out = Path(args.out) / "mask" / args.judge_model
    report.write_report(out, arms=args.arms.split(",") if args.arms else None, n_boot=args.n_boot)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    m = sub.add_parser("mask", help="plan 0017: MASK lie-claim eval (gt | run | report)")
    msub = m.add_subparsers(dest="mask_cmd", required=True)

    g = msub.add_parser("gt", help="stage T: MASK judge calls -> data/mask/labels.json")
    _common(g)
    g.add_argument(
        "--rollouts",
        default="outputs/mask/gen/rollouts.jsonl,outputs/mask/gen/rollouts_x.jsonl",
        help="comma-separated gen_modal.py outputs",
    )
    g.add_argument("--labels-only", action="store_true", help="no API calls; rebuild labels")
    g.set_defaults(fn=cmd_gt)

    r = msub.add_parser("run", help="stage D (detect) -> V (verify) on lens readouts")
    _common(r)
    r.add_argument("--readouts-root", default=None, help="default $HBENCH_MASK_READOUTS")
    r.add_argument("--arms", default="s3d-rl600,mlnla-rl600-all11,jlens")
    r.add_argument("--stages", default="detect,verify,verify_blind")
    r.add_argument("--layers", default=None)
    r.add_argument("--limit", type=int, default=None, help="cells per arm (smoke)")
    r.set_defaults(fn=cmd_run)

    rp = msub.add_parser("report", help="metrics + bootstrap -> report.md / report.json")
    rp.add_argument("--out", default="outputs")
    rp.add_argument("--judge-model", default=DEFAULT_MODEL)
    rp.add_argument("--arms", default=None)
    rp.add_argument("--n-boot", type=int, default=2000)
    rp.set_defaults(fn=cmd_report)
