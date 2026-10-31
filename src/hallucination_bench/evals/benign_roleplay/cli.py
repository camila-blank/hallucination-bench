"""``hbench benign-roleplay import|run|report`` (plan 0023), registered through
``cli.EVAL_CLIS``. Exit codes: 0 ok, 2 pin / layout / refused input, else ``wsbench``'s code."""

import argparse
import json
import sys
from pathlib import Path

from hallucination_bench import paths

from . import bank, metrics, wsb

# the jailbreak family's registered judge pin (WSB evals/jailbreak_recognition/judge_smoke_v2.json,
# plan 0037 J8.5 / Q5): HB passes no judge_model=, so this only names the output dir and is
# checked against every input's config.judge_model
DEFAULT_JUDGE = "claude-sonnet-5"


def _layers(s: str | None) -> list[int] | None:
    return [int(x) for x in s.split(",") if x.strip()] if s else None


def _arms(s: str) -> list[str]:
    arms = [a for a in s.split(",") if a]
    bad = set(arms) - set(wsb.KNOWN_ARMS)
    if bad:
        raise wsb.WsbError(f"unknown arms {sorted(bad)}; known {list(wsb.KNOWN_ARMS)}")
    return arms


def cmd_import(args: argparse.Namespace) -> int:
    root = Path(args.wsb)
    wsb.check_pin(root, args.wsb_sha)
    bp = wsb.bank_path(root)
    rel = str(bp.resolve().relative_to(root.resolve()))
    build_p = root / bank.BUILD_REL
    items_doc, hb_doc, gt = bank.build(
        json.loads(bp.read_text(encoding="utf-8")),
        json.loads(build_p.read_text(encoding="utf-8")),
        wsb_sha=args.wsb_sha,
        bank_rel=rel,
        bank_sha256=bank.sha256_bytes(bp.read_bytes()),
        build_sha256=bank.sha256_bytes(build_p.read_bytes()),
        n_target=args.n_target,
    )
    bank.write(items_doc, hb_doc, gt, Path(args.dest))
    print(
        f"[import] {items_doc['n_items']} items ({items_doc['n_shared']} shared, "
        f"{items_doc['n_hb_only']} hb_only), kind mix {items_doc['kind_mix']} -> {args.dest}"
    )
    return 0


def _out(args: argparse.Namespace) -> Path:
    return Path(args.out).resolve() / "benign_roleplay" / args.judge


def cmd_run(args: argparse.Namespace) -> int:
    calls = wsb.plan_calls(
        campaign=Path(args.campaign),
        readouts=Path(args.readouts_root),
        out=_out(args),
        layers=_layers(args.layers),
        arms=_arms(args.arms),
    )
    return wsb.run_calls(Path(args.wsb), calls, dry_run=args.dry_run)


def cmd_report(args: argparse.Namespace) -> int:
    out = _out(args)
    floor_res = metrics.read_result(Path(args.floor)) if args.floor else None
    s = metrics.summarize(
        bank.load_items(),
        out=out,
        campaign=Path(args.campaign),
        arms=_arms(args.arms),
        layers=_layers(args.layers),
        floor_res=floor_res,
        accept_refused_as_flagged=args.accept_refused_as_flagged,
    )
    for arm, a in s["rates"].items():
        if a["instrument"].get("judge_model") != args.judge:
            raise wsb.WsbError(
                f"{arm}: inputs judged by {a['instrument'].get('judge_model')}, not {args.judge}"
            )
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(s, indent=1) + "\n", encoding="utf-8")
    (out / "summary.md").write_text(metrics.render_md(s), encoding="utf-8")
    print(metrics.render_md(s), end="")
    print(f"wrote {out / 'summary.json'} and {out / 'summary.md'}")
    return 0


def _wrap(fn):  # type: ignore[no-untyped-def]
    def inner(args: argparse.Namespace) -> int:
        try:
            return int(fn(args))
        except (wsb.WsbError, ValueError) as e:  # ValueError: bank.build's fail-closed gate
            print(f"benign-roleplay: {e}", file=sys.stderr)
            return 2

    return inner


def register(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser("benign-roleplay", help="plan 0023: false jailbreak recognition (HB eval)")
    bsub = p.add_subparsers(dest="br_cmd", required=True)

    def common(q: argparse.ArgumentParser) -> None:
        q.add_argument("--out", default="outputs", help="output root (default: outputs)")
        q.add_argument("--judge", default=DEFAULT_JUDGE, help="the family pin (output dir name)")
        q.add_argument("--campaign", default=str(paths.BR_CAMPAIGN), help="campaign judged dir")
        q.add_argument("--arms", default=",".join(wsb.ARMS))
        q.add_argument("--layers", default=None, help="csv subset of readplan.GRID (C3)")

    im = bsub.add_parser("import", help="copy the gated controls from a pinned WSB checkout")
    im.add_argument("--wsb", default=str(paths.WSBENCH_PUB))
    im.add_argument("--wsb-sha", default=wsb.WSB_SHA)
    im.add_argument("--n-target", type=int, default=bank.N_TARGET)
    im.add_argument("--dest", default=str(bank.DATA_DIR))
    im.set_defaults(fn=_wrap(cmd_import))

    rn = bsub.add_parser("run", help="judge via the pinned `wsbench run` (no GPU)")
    common(rn)
    rn.add_argument("--wsb", default=str(paths.WSBENCH_PUB))
    rn.add_argument("--readouts-root", default=str(paths.BR_READOUTS))
    rn.add_argument("--dry-run", action="store_true", help="print the wsbench calls only")
    rn.set_defaults(fn=_wrap(cmd_run))

    rp = bsub.add_parser("report", help="summary.{json,md} from the judged results")
    common(rp)
    rp.add_argument("--floor", default=None, help="po-opus-v1 jailbreak_recognition results.json")
    rp.add_argument(
        "--accept-refused-as-flagged",
        action="store_true",
        help="admit WSB results incomplete ONLY through judge-refused (unjudged) cells; those "
        "items count as flagged (conservative) and the count is shown beside the rate",
    )
    rp.set_defaults(fn=_wrap(cmd_report))
