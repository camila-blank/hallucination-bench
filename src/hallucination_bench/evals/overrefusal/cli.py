"""``hbench overrefusal candidates|judge-gt|bank|run|report`` (plan 0015), registered through
``hallucination_bench.cli.EVAL_CLIS`` (plan 0012 D2)."""

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_MODEL = "claude-opus-5-5"


def _readouts_root(args: argparse.Namespace) -> Path:
    return Path(
        args.readouts_root
        or os.environ.get("HBENCH_OR_READOUTS")
        or "/workspace/hb-readouts-overrefusal-v2"  # rev 7: old + v2add readouts
    )


def cmd_candidates(args: argparse.Namespace) -> int:
    from . import data

    print(json.dumps(data.build_candidates(args.n_hard, args.n_toxic, args.seed), indent=1))
    return 0


def cmd_judge_gt(args: argparse.Namespace) -> int:
    from . import run

    out = Path(args.out)
    res = run.run_gt(
        out,
        run.Judge(model=args.judge_model),
        samples_path=Path(args.samples) if args.samples else out / "overrefusal" / "samples.jsonl",
        greedy_path=Path(args.greedy)
        if args.greedy
        else out / "overrefusal" / "samples_greedy.jsonl",
        candidates_path=Path(args.candidates) if args.candidates else None,
        labels_out=Path(args.labels_out) if args.labels_out else None,
        rpm=args.rpm,
        concurrency=args.concurrency,
        limit=args.limit,
        write=not args.limit,
    )
    print(json.dumps(res, indent=1))
    return 4 if res["n_failed"] else 0


def cmd_bank(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    from . import data

    gt = data.read_jsonl(data.GT_LABELS)
    pop, eligible = data.select_population(
        gt, n_never=args.n_never, n_hard_refused=args.n_hard_refused, n_always=args.n_always
    )
    tok = AutoTokenizer.from_pretrained(data.SUBJECT)
    bank = data.build_bank(pop, data.load_candidates(), tok)
    meta = {
        "eligible": eligible,
        "selected": {k: len(v) for k, v in pop.items()},
        "n_sites": sum(len(it["sites"]) for it in bank["items"]),
        "subject": data.SUBJECT,
    }
    data.ITEMS.write_text(
        json.dumps({"meta": meta, "items": bank["items"]}, ensure_ascii=False, indent=1) + "\n"
    )
    data.CAPTURE_ROWS.write_text(json.dumps(bank["capture_rows"], ensure_ascii=False) + "\n")
    print(json.dumps(meta, indent=1))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from hallucination_bench.llm import JudgeConfigError

    from . import run

    stages = [s for s in args.stages.split(",") if s]
    bad = set(stages) - set(run.STAGES)
    if bad:
        raise SystemExit(f"unknown stages {sorted(bad)}; known {run.STAGES}")
    arms = run.resolve_arms(_readouts_root(args), args.arms.split(",") if args.arms else None)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    try:
        res = run.run(
            arms,
            Path(args.out),
            run.Judge(model=args.judge_model),
            stages=stages,
            layers=layers,
            limit=args.limit,
            rpm=args.rpm,
            concurrency=args.concurrency,
        )
    except JudgeConfigError as e:
        print(f"JUDGE CONFIG ERROR: {e}", file=sys.stderr)
        return 3
    nf = 0
    for name, s in res.items():
        print(
            f"[{name}] cells={s['n_cells']} status={s['status']} "
            f"refusal_cand={s['n_refusal_candidates']} rejected={s['n_rejected']} "
            f"unverified={s['n_unverified']} api_refused={s['n_api_refused']} "
            f"spend=${s['spend']['usd']:.2f}",
            flush=True,
        )
        nf += run.n_failed(s) if not args.limit else 0
    return 4 if nf else 0


def cmd_report(args: argparse.Namespace) -> int:
    from . import report

    root = report.report(
        Path(args.out),
        args.judge_model,
        _readouts_root(args),
        args.arms.split(",") if args.arms else None,
    )
    print(f"wrote {root / 'report.md'}, {root / 'report.json'}, {root / 'examples.jsonl'}")
    hb = json.loads((root / "report.json").read_text()).get("hb_scores")
    if hb:
        print("HB score (never_refused, tail): probe_validated [headline] / raw, % readouts")
        for arm, by in hb["arms"].items():
            cells = [
                f"L{la} {100 * d['tail']['probe_validated']['mean']:.1f} / "
                f"{100 * d['tail']['raw']['mean']:.1f}"
                for la, d in by.items()
            ]
            print(f"  {arm}: " + "  ".join(cells))
    return 0


def cmd_probe_data(args: argparse.Namespace) -> int:
    from . import probe_data

    if args.step == "80k":
        res: object = probe_data.build_80k_candidates()
    elif args.step == "calib":
        res = {"n_calib_texts": probe_data.build_calib_texts()}
    else:
        res = probe_data.build_all()
    print(json.dumps(res, indent=1, default=str))
    return 0


def cmd_probe_report(args: argparse.Namespace) -> int:
    from . import probe_report

    arms = args.arms.split(",") if args.arms else ["s3d-rl600", "mlnla-rl600-all11", "jlens"]
    root = probe_report.report(Path(args.out), args.judge_model, arms)
    print(f"wrote {root / 'probe_report.md'}, {root / 'probe_report.json'}")
    return 0


def cmd_grow(args: argparse.Namespace) -> int:
    from . import grow

    if args.step == "candidates":
        res = grow.build_80k2_candidates()
    elif args.step == "pairs":
        from transformers import AutoTokenizer

        from . import data

        res = grow.grow_pair_controls(AutoTokenizer.from_pretrained(data.SUBJECT))
    else:
        from transformers import AutoTokenizer

        from . import data

        res = grow.grow_bank(AutoTokenizer.from_pretrained(data.SUBJECT))
    print(json.dumps(res, indent=1, default=str))
    return 0


def cmd_freeze(args: argparse.Namespace) -> int:
    from . import data

    doc = json.loads(data.ITEMS.read_text())
    doc["meta"]["frozen"] = args.date
    doc["meta"]["hb_score"] = (
        "verified refusal_intent readout rate on never_refused (tail sites headline, all sites "
        "reported); hard_refused / always_refused are diagnostics only"
    )
    data.ITEMS.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {data.write_manifest()}")
    return 0


def cmd_wsb_export(args: argparse.Namespace) -> int:
    from . import wsb_export

    res = wsb_export.export(Path(args.out), args.judge_model, Path(args.family_dir))
    print(json.dumps(res, indent=1))
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("overrefusal", help="plan 0015: OR-Bench over-refusal eval")
    osub = p.add_subparsers(dest="or_cmd", required=True)

    c = osub.add_parser(
        "candidates", help="Stage 0: pinned OR-Bench -> data/overrefusal/candidates.jsonl"
    )
    c.add_argument("--n-hard", type=int, default=400)
    c.add_argument("--n-toxic", type=int, default=150)
    c.add_argument("--seed", type=int, default=0)
    c.set_defaults(fn=cmd_candidates)

    def common(q: argparse.ArgumentParser) -> None:
        q.add_argument("--out", default="outputs")
        q.add_argument("--judge-model", default=DEFAULT_MODEL)
        q.add_argument("--rpm", type=float, default=1000.0)
        q.add_argument("--concurrency", type=int, default=96)
        q.add_argument("--limit", type=int, default=None, help="smoke: at most N calls / cells")

    g = osub.add_parser("judge-gt", help="Stage 2: refusal labels (orgt-v1) -> gt_labels.jsonl")
    common(g)
    g.add_argument("--candidates", default="", help="rev 5: another candidates file (80k sample)")
    g.add_argument("--samples", default="", help="rev 5: its samples file")
    g.add_argument("--labels-out", default="", help="rev 5: where its labels go")
    g.add_argument("--greedy", default="", help="rev 7: its greedy samples file")
    g.set_defaults(fn=cmd_judge_gt)

    b = osub.add_parser("bank", help="Stage 3: populations + items.json + capture_rows.json")
    b.add_argument("--n-never", type=int, default=150)
    b.add_argument("--n-hard-refused", type=int, default=50)
    b.add_argument("--n-always", type=int, default=50)
    b.set_defaults(fn=cmd_bank)

    r = osub.add_parser("run", help="Stages 5-6: detector + verifier")
    common(r)
    r.add_argument("--readouts-root", default=None, help="default $HBENCH_OR_READOUTS")
    r.add_argument("--arms", default="")
    r.add_argument("--layers", default="")
    r.add_argument("--stages", default="detect,verify")
    r.set_defaults(fn=cmd_run)

    gw = osub.add_parser("grow", help="rev 7: 80k2 candidates | append new items to the bank")
    gw.add_argument("step", choices=["candidates", "bank", "pairs"])
    gw.set_defaults(fn=cmd_grow)

    wx = osub.add_parser("wsb-export", help="rev 8: WorkspaceBench refusal-prediction family")
    wx.add_argument("--family-dir", required=True)
    wx.add_argument("--out", default="outputs")
    wx.add_argument("--judge-model", default="claude-sonnet-5-5")
    wx.set_defaults(fn=cmd_wsb_export)

    fz = osub.add_parser("freeze", help="rev 8: mark items.json frozen + MANIFEST.sha256")
    fz.add_argument("--date", default="2026-10-28")
    fz.set_defaults(fn=cmd_freeze)

    pd = osub.add_parser("probe-data", help="rev 5: probe training data (80k sample | texts)")
    pd.add_argument("step", choices=["80k", "texts", "calib"])
    pd.set_defaults(fn=cmd_probe_data)

    pr = osub.add_parser("probe-report", help="rev 5: probe validity, premise, tagging, misses")
    pr.add_argument("--out", default="outputs")
    pr.add_argument("--judge-model", default="claude-sonnet-5-5")
    pr.add_argument("--arms", default="")
    pr.set_defaults(fn=cmd_probe_report)

    rp = osub.add_parser("report", help="Stage 7: report.md / report.json / examples.jsonl")
    rp.add_argument("--out", default="outputs")
    rp.add_argument("--judge-model", default=DEFAULT_MODEL)
    rp.add_argument("--readouts-root", default=None)
    rp.add_argument("--arms", default="")
    rp.set_defaults(fn=cmd_report)
