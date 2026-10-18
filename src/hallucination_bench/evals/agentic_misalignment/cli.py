"""``hbench run|report --eval agentic_misalignment`` (plan 0008): the AM driver behind the shared
``run`` / ``report`` subcommands (moved out of ``hallucination_bench.cli`` by plan 0012 D2)."""

import argparse
import sys
from pathlib import Path

from hallucination_bench.cli import _items, _layers, _models


def run(args: argparse.Namespace) -> int:
    """``hbench run --eval agentic_misalignment`` (plan 0008): the AM driver, its own
    ``--stages`` (detect,summ,verify,map,intent) and readouts file; exit 4 when failed calls
    remain."""
    from hallucination_bench.evals.agentic_misalignment import run as amrun

    try:
        stages = amrun.parse_stages(args.stages)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    arms = amrun.resolve_arms(args.arm, args.readouts_root, args.arms)
    models = _models(args)
    out = Path(args.out)
    if args.rebuild:
        summaries = {n: amrun.rebuild(n, p, out, models) for n, p in arms}
        intent = amrun.intent_status(out, models)
    else:
        res = amrun.run_am(
            arms,
            out,
            models,
            layers=_layers(args.layers),
            items=_items(args.items),
            limit=args.limit,
            rpm=args.rpm,
            concurrency=args.concurrency,
            stages=stages,
            show_prompts=args.show_prompts,
        )
        summaries, intent = res["arms"], res["intent"]
    for n, s in summaries.items():
        print_summary(n, s)
    print(
        f"[intent] ok={intent['n_ok']} failed={intent['n_failed']} pending={intent['n_pending']}"
        + (f" failed_items={intent['failed']}" if intent["failed"] else ""),
        flush=True,
    )
    if args.rebuild:
        return 0
    # intent failures count toward exit 4 only when the intent stage ran (a refused intent item is
    # resolved by intent_overrides.json, and must not block the detector / map runs)
    n_failed = sum(amrun.n_failed(s) for s in summaries.values())
    if "intent" in stages:
        n_failed += intent["n_failed"]
    # Stage H failures likewise count only when the verify stage ran (plan 0008 rev 13)
    if "verify" in stages:
        n_failed += sum(amrun.n_failed_verify(s) for s in summaries.values())
    if n_failed:
        print(f"{n_failed} failed calls remain; re-run to retry them", file=sys.stderr)
        return 4
    return 0


def print_summary(name: str, s: dict) -> None:
    h = s["health"]
    det = s["overall"]["detection"]
    v = s["overall"]["headline"]["verdict"]["rows"]

    def f(x: float | None) -> str:
        return "—" if x is None else f"{x:.3f}"

    failed = {k.removeprefix("n_failed_"): h[k] for k in h if k.startswith("n_failed_")}
    print(
        f"[{name}] layers={s['config']['judged_layers']} detected={det['n_detected']}/"
        f"{det['n_detect_ok']} rate={f(det['detected_rate'])} mentions={det['n_mentions']} "
        f"pending_detect={h['n_pending_detect']} map_ok={h['n_map_ok']} "
        f"pending_map={h['n_pending_map']} failed={failed} | headline spans="
        f"{s['overall']['headline']['n_spans']} hallucination={f(v['hallucination']['pooled'])} "
        f"thinky={f(v['thinky']['pooled'])} spend_total=${s['spend']['total_usd']:.2f}",
        flush=True,
    )


def report(args: argparse.Namespace) -> int:
    """``hbench report --eval agentic_misalignment`` (plan 0008): rebuild every AM arm from its
    readouts (default root: the paper readouts) plus the cache, then report.md / report.json."""
    if args.view == "v2":
        raise SystemExit(
            "report --eval agentic_misalignment has no --view v2: its report is the claim map "
            "already (drop --view)"
        )
    from hallucination_bench.evals.agentic_misalignment import report as amreport

    root = amreport.report(
        Path(args.out),
        args.judge_model,
        readouts_root=Path(args.readouts_root) if args.readouts_root else None,
        rebuild=not args.no_rebuild,
        audit_n=args.audit_n if args.audit_n is not None else 10,
        seed=args.seed,
    )
    print(f"wrote {root / 'report.md'}, {root / 'report.json'}")
    return 0
