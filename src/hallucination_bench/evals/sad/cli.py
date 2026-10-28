"""``hbench sad ...`` (plan 0018): registered through ``hallucination_bench.cli.EVAL_CLIS``.

hbench sad gate-requests      # raw texts -> outputs/sad/gate/requests.jsonl (Modal input)
hbench sad gate               # vLLM samples -> data/sad/gate.json (+ survival)
hbench sad select             # gate.json -> data/sad/selected.json + outputs/sad/capture_rows.json
hbench sad check-manifest M   # a capture manifest.json vs the local tokenisation
hbench sad run --arms a,b     # detector + verifier, then summary.json per arm
hbench sad report             # report.md / report.json over every arm dir
"""

import argparse
import json
import sys
from pathlib import Path

from hallucination_bench import paths

DEFAULT_JUDGE = "claude-opus-5-5"
DEFAULT_ARMS = "s3d-rl600,mlnla-rl600-all11,jlens"
TOKENIZER = "Qwen/Qwen3.6-27B"
GATE_DIR = Path("outputs/sad/gate")


def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER)


def cmd_gate_requests(args: argparse.Namespace) -> int:
    from . import bank, gate

    rows = list(bank.load_raw().values())
    req = gate.requests(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in req:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(req)} requests ({len(rows)} items x {gate.N_SAMPLES}) -> {out}")
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    from . import bank, gate

    req = {
        r["key"]: r
        for r in map(json.loads, Path(args.requests).read_text().splitlines())
        if r.get("key")
    }
    samples = []
    for line in Path(args.samples).read_text().splitlines():
        if line.strip():
            s = json.loads(line)
            samples.append({**s, "yes_is_a": req[s["key"]]["yes_is_a"]})
    prov = {it["id"]: it["provenance"] for it in bank.load_manifest()["items"]}
    g = gate.aggregate(samples, prov)
    missing = sorted(set(prov) - set(g))
    surv = gate.survival(g)
    doc = {
        "meta": {
            "question_variant": list(gate.QUESTION_VARIANT),
            "question_sha256": gate.QUESTION_SHA256,
            "system": gate.SYSTEM,
            "n_samples": gate.N_SAMPLES,
            "threshold": gate.THRESHOLD,
            "temperature": gate.TEMPERATURE,
            "top_p": gate.TOP_P,
            "order_balance": "even samples (A) yes/(B) no, odd samples (A) no/(B) yes",
            "model": "Qwen/Qwen3.6-27B (vLLM 0.25.1, enable_thinking=False)",
            "n_items_missing": len(missing),
            "survival": surv,
        },
        "items": g,
    }
    Path(args.dest).write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(surv, indent=1))
    if missing:
        print(f"WARNING: {len(missing)} items have no samples", file=sys.stderr)
    return 0


def cmd_select(args: argparse.Namespace) -> int:
    from . import bank, select

    g = json.loads(bank.GATE_JSON.read_text())["items"]
    raw = bank.load_raw()
    sel = select.build_selected(g, raw, _tokenizer(), TOKENIZER)
    bank.SELECTED_JSON.write_text(json.dumps(sel, indent=1) + "\n")
    rows = select.capture_rows(sel, raw)
    out = Path(args.capture_rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False) + "\n")
    print(json.dumps(sel["meta"], indent=1))
    print(f"wrote {bank.SELECTED_JSON} and {out} ({len(rows)} capture rows; gitignored)")
    return 0


def cmd_top_up(args: argparse.Namespace) -> int:
    """Rev 5: keep the rev-4 selection, fill each belief to --n-per-belief; capture rows for the
    NEW items only (they go to a second acts subdir)."""
    from . import bank, select

    g = json.loads(bank.GATE_JSON.read_text())["items"]
    prev = json.loads(bank.SELECTED_JSON.read_text())
    raw = bank.load_raw()
    sel, new = select.top_up(prev, g, raw, _tokenizer(), TOKENIZER, args.n_per_belief)
    bank.SELECTED_JSON.write_text(json.dumps(sel, indent=1) + "\n")
    sub = {"meta": sel["meta"], "items": [it for it in sel["items"] if it["id"] in set(new)]}
    rows = select.capture_rows(sub, raw)
    out = Path(args.capture_rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False) + "\n")
    print(json.dumps(sel["meta"], indent=1))
    print(f"wrote {bank.SELECTED_JSON} and {out} ({len(rows)} NEW capture rows; gitignored)")
    return 0


def cmd_probe_inputs(args: argparse.Namespace) -> int:
    """Rev 6: probe-job inputs (no text) + capture rows for the training / contrast texts."""
    from . import bank, probe

    g = json.loads(bank.GATE_JSON.read_text())["items"]
    sel = json.loads(bank.SELECTED_JSON.read_text())
    inp, rows, crow = probe.build_inputs(g, sel, bank.load_raw(), _tokenizer())
    d = Path(args.dest)
    d.mkdir(parents=True, exist_ok=True)
    (d / "probe_inputs.json").write_text(json.dumps(inp) + "\n")
    (d / "capture_rows_probe_train.json").write_text(json.dumps(rows, ensure_ascii=False))
    (d / "capture_rows_probe_contrast.json").write_text(json.dumps(crow, ensure_ascii=False))
    print(json.dumps(inp["meta"]))
    return 0


def cmd_probe_report(args: argparse.Namespace) -> int:
    from . import probe_report

    doc = probe_report.report(
        Path(args.out), args.judge_model, Path(args.probes), Path(args.inputs)
    )
    print(json.dumps(doc["aurocs"], indent=1))
    return 0


def cmd_check_manifest(args: argparse.Namespace) -> int:
    """The capture's token grid must equal the local tokenisation at every read site."""
    from . import bank, select

    man = json.loads(Path(args.manifest).read_text())
    sel = json.loads(bank.SELECTED_JSON.read_text())
    raw = bank.load_raw()
    tok = _tokenizer()
    by_label = {p["label"]: p for p in man["prompts"]}
    bad = 0
    want_items = [it for it in sel["items"] if not args.new_only or it.get("batch") == "rev5"]
    for it in want_items:
        p = by_label.get(it["id"])
        if p is None:
            print(f"{it['id']}: not in the manifest")
            bad += 1
            continue
        full = bank.plain_render(raw[it["id"]]["text"]) + select.GEN_SUFFIX
        ids = tok(full, add_special_tokens=False)["input_ids"]
        want = [t.replace("Ġ", " ") for t in tok.convert_ids_to_tokens(ids)]
        if p["n_pos"] != len(ids) or p["tokens"] != want:
            print(f"{it['id']}: token grid differs (n_pos {p['n_pos']} vs {len(ids)})")
            bad += 1
            continue
        evp = sorted(p.get("eval_positions") or [])
        if evp != sorted(s["pos"] for s in it["sites"]):
            print(f"{it['id']}: eval_positions differ from selected sites")
            bad += 1
    print(f"check-manifest: {len(want_items) - bad}/{len(want_items)} items ok")
    return 1 if bad else 0


def cmd_run(args: argparse.Namespace) -> int:
    from . import run

    stages = {s for s in args.stages.split(",") if s}
    unknown = stages - {"detect", "verify"}
    if unknown:
        raise SystemExit(f"unknown --stages {sorted(unknown)} (detect,verify)")
    root = Path(args.readouts_root)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    effort = None if args.effort in ("", "none") else args.effort
    n_failed = 0
    for arm in [a for a in args.arms.split(",") if a]:
        s = run.run_arm(
            arm,
            root / arm / run.READOUTS_FILE,
            Path(args.out),
            judge=args.judge_model,
            effort=effort,
            layers=layers,
            stages=stages,
            rpm=args.rpm,
            concurrency=args.concurrency,
            limit=args.limit,
        )
        h = s["health"]
        n_failed += h.get("n_failed_detect", 0) + h.get("n_unverified_in_sample", 0)
        print(
            f"[{arm}] layers={s['layers']} cells={s['n_cells']} health={h} "
            f"mentions={s['n_mentions']} verify_sample={s['n_verify_sample']} "
            f"spend=${s['spend']['total_usd']:.2f} run={s.get('run_stats')}",
            flush=True,
        )
    if n_failed:
        print(f"{n_failed} failed / unverified remain; re-run to retry", file=sys.stderr)
        return 4
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from . import report

    root = report.report(
        Path(args.out),
        args.judge_model,
        readouts_root=Path(args.readouts_root),
        n_examples=args.n_examples,
    )
    print(f"wrote {root / 'report.md'}, {root / 'report.json'}")
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("sad", help="plan 0018: SAD stages-oversight eval-vs-deployment eval")
    s = p.add_subparsers(dest="sad_cmd", required=True)

    q = s.add_parser("gate-requests", help="vLLM gate requests (texts; gitignored)")
    q.add_argument("--out", default=str(GATE_DIR / "requests.jsonl"))
    q.set_defaults(fn=cmd_gate_requests)

    q = s.add_parser("gate", help="parse gate samples -> data/sad/gate.json")
    q.add_argument("--requests", default=str(GATE_DIR / "requests.jsonl"))
    q.add_argument("--samples", default=str(GATE_DIR / "samples.jsonl"))
    q.add_argument("--dest", default="data/sad/gate.json")
    q.set_defaults(fn=cmd_gate)

    q = s.add_parser("select", help="gated items + read sites -> selected.json, capture rows")
    q.add_argument("--capture-rows", default="outputs/sad/capture_rows.json")
    q.set_defaults(fn=cmd_select)

    q = s.add_parser("top-up", help="rev 5: keep the selection, fill each belief group")
    q.add_argument("--n-per-belief", type=int, default=100)
    q.add_argument("--capture-rows", default="outputs/sad/capture_rows_rev5.json")
    q.set_defaults(fn=cmd_top_up)

    q = s.add_parser("probe-inputs", help="rev 6: probe-job inputs + capture rows (gitignored)")
    q.add_argument("--dest", default="outputs/sad")
    q.set_defaults(fn=cmd_probe_inputs)

    q = s.add_parser("probe-report", help="rev 6: premise check, probe-checked claims, misses")
    q.add_argument("--out", default="outputs")
    q.add_argument("--judge-model", default=DEFAULT_JUDGE)
    q.add_argument("--probes", default="outputs/sad/probes")
    q.add_argument("--inputs", default="outputs/sad/probe_inputs.json")
    q.set_defaults(fn=cmd_probe_report)

    q = s.add_parser("check-manifest", help="capture manifest vs local tokenisation")
    q.add_argument("manifest")
    q.add_argument("--new-only", action="store_true", help="only the rev-5 top-up items")
    q.set_defaults(fn=cmd_check_manifest)

    for name, fn in (("run", cmd_run), ("report", cmd_report)):
        q = s.add_parser(name)
        q.add_argument("--out", default="outputs")
        q.add_argument("--judge-model", default=DEFAULT_JUDGE)
        q.add_argument("--readouts-root", default=str(paths.SAD_READOUTS))
        if name == "run":
            q.add_argument("--arms", default=DEFAULT_ARMS)
            q.add_argument("--layers", default=None)
            q.add_argument("--stages", default="detect,verify")
            q.add_argument("--effort", default="low")
            q.add_argument("--rpm", type=float, default=1500.0)
            q.add_argument("--concurrency", type=int, default=64)
            q.add_argument("--limit", type=int, default=None, help="first N cells per arm (smoke)")
        else:
            q.add_argument("--n-examples", type=int, default=10)
        q.set_defaults(fn=fn)
