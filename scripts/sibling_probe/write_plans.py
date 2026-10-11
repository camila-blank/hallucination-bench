"""ReadSpec plan files, s3d specs and post-``prompts`` / post-capture checks (plan 0003 §Phase C).

  plans          items.json -> ``outputs/sibling/plan_eval/sibling_<cat>.jsonl``;
                 probe_texts.jsonl -> ``outputs/sibling/plan_probe/sibprobe_<cat>.jsonl`` (render
                 ``chat_context``, ``from_last_sentence_start``, layers 28,36,42,44,52);
                 ``outputs/sibling/s3d_specs.json`` (fanout_custom, layers 28,36,44,52);
                 ``outputs/sibling/jlens_jobs.json``. Every row's read window is checked first
                 (window identity, exit 2 on a failure). ``--limit-eval N`` / ``--ids a,b`` with
                 ``--tag smoke`` write a smoke plan of those eval items only (``plan_eval_smoke``).
  check-prompts  a ``wsbench_public_plan.py prompts`` out-dir: each row's ``input_ids`` equal the
                 local render, ``eval_positions`` equal ``from_last_sentence_start``, and the
                 window ids equal the canonical window of the row's request sentence.
  check-cells    merged readouts (s3d shards merged / converted J-lens / NLA) vs the render
                 meta: complete (id, layer, pos) grid, no duplicates, tokens equal (exit 2).
  check-manifest a capture ``manifest.json`` (``modal volume get wsbench-acts
                 /<subdir>/manifest.json``) against the same render dir: labels, ``n_pos``,
                 ``eval_positions`` and the display tokens at the eval positions.

    uv run python scripts/sibling_probe/write_plans.py plans
    uv run python scripts/sibling_probe/write_plans.py check-prompts \\
        --render-dir outputs/sibling/render_eval
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from hallucination_bench import REPO_ROOT
from hallucination_bench.evals.sibling_probe import bank as B
from hallucination_bench.evals.sibling_probe.concepts import load_inventory

OUT = REPO_ROOT / "outputs" / "sibling"
TOKENIZER = "Qwen/Qwen3.6-27B"


def _rows_and_requests() -> tuple[list[dict], list[dict], dict[str, str]]:
    items = B.load_items()
    probes = B.load_probe_texts() if B.PROBE_TEXTS_PATH.exists() else []
    inv = load_inventory()
    req = {it["id"]: it["request"] for it in items}
    for p in probes:
        req[p["id"]] = inv[p["category"]].requests[p["request_idx"]]
    return items, probes, req


def cmd_plans(args: argparse.Namespace) -> None:
    items, probes, req = _rows_and_requests()
    wc = B.WindowChecker(AutoTokenizer.from_pretrained(TOKENIZER))
    if args.ids:
        want = args.ids.split(",")
        items = [it for it in items if it["id"] in want]
        probes = []
    elif args.limit_eval:
        items = items[: args.limit_eval]
        probes = []
    bad = [r["id"] for r in [*items, *probes] if not wc.ok(r["messages"], req[r["id"]])]
    if bad:
        print(f"window identity FAILS on {len(bad)} rows: {bad[:10]}")
        sys.exit(2)
    sfx = f"_{args.tag}" if args.tag else ""
    n_eval = B.write_plan_files(items, OUT / f"plan_eval{sfx}", B.EVAL_FAMILY_PREFIX)
    print(f"eval plan -> {OUT / f'plan_eval{sfx}'}: {n_eval}")
    if not (args.limit_eval or args.ids):
        n_probe = B.write_plan_files(probes, OUT / "plan_probe", B.PROBE_FAMILY_PREFIX)
        print(f"probe plan -> {OUT / 'plan_probe'}: {n_probe}")
        n = B.write_s3d_specs(items, OUT / "s3d_specs.json")
        print(f"s3d specs -> {OUT / 's3d_specs.json'}: {n} rows, layers {list(B.S3D_LAYERS)}")
        jobs = [
            {
                "acts": "acts-sibling-eval",
                "out": "gen-sibling-eval-jlens",
                "layers": list(B.S3D_LAYERS),
            }
        ]
        (OUT / "jlens_jobs.json").write_text(json.dumps(jobs))
        print(f"jlens jobs -> {OUT / 'jlens_jobs.json'}")


def _render_rows(render_dir: Path) -> list[dict]:
    rows = []
    for p in sorted((render_dir / "prompts").glob("*.json")):
        rows += json.loads(p.read_text())
    return rows


def cmd_check_prompts(args: argparse.Namespace) -> None:
    items, probes, req = _rows_and_requests()
    msgs = {r["id"]: r["messages"] for r in [*items, *probes]}
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    wc = B.WindowChecker(tok)
    rows = _render_rows(args.render_dir)
    bad = []
    max_tokens = 0
    for r in rows:
        lab = r["label"]
        ids = B.render_ids(tok, msgs[lab])
        pos = B.from_last_sentence_start(B.display_tokens(tok, ids))
        win = [r["input_ids"][p] for p in r["eval_positions"]]
        max_tokens = max(max_tokens, len(r["input_ids"]))
        if r["input_ids"] != ids:
            bad.append((lab, "input_ids differ from the local render"))
        elif r["eval_positions"] != pos:
            bad.append((lab, f"eval_positions {r['eval_positions']} != {pos}"))
        elif win != wc.canonical(req[lab]):
            bad.append((lab, "window ids differ from the canonical window"))
    print(f"[check-prompts] {len(rows)} rows, max_tokens {max_tokens}, problems {len(bad)}")
    for b in bad[:20]:
        print("  ", b)
    if bad:
        sys.exit(2)


def cmd_check_manifest(args: argparse.Namespace) -> None:
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    rows = {r["label"]: r for r in _render_rows(args.render_dir)}
    man = json.loads(args.manifest.read_text())
    bad = []
    for e in man["prompts"]:
        r = rows.get(e["label"])
        if r is None:
            bad.append((e["label"], "not in the render dir"))
            continue
        if e["n_pos"] != len(r["input_ids"]):
            bad.append((e["label"], f"n_pos {e['n_pos']} != {len(r['input_ids'])}"))
        if list(e.get("eval_positions") or []) != r["eval_positions"]:
            bad.append((e["label"], "eval_positions differ"))
        toks = e.get("tokens") or []
        if toks:
            # the manifest stores display spellings (Ġ -> space, Ċ kept), as wsbench does
            disp = B.display_tokens(tok, r["input_ids"])
            want = [disp[p] for p in r["eval_positions"]]
            got = [toks[p] for p in r["eval_positions"]] if len(toks) == e["n_pos"] else toks
            if got != want:
                bad.append((e["label"], f"tokens differ: {got[:4]} vs {want[:4]}"))
    missing = set(rows) - {e["label"] for e in man["prompts"]}
    print(
        f"[check-manifest] {len(man['prompts'])} prompts, layers {man.get('layers')}, "
        f"missing {len(missing)}, problems {len(bad)}"
    )
    for b in bad[:20]:
        print("  ", b)
    if bad or missing:
        sys.exit(2)


def cmd_check_cells(args: argparse.Namespace) -> None:
    """Merged readouts vs the render meta: every (id, layer, pos) present exactly once, and the
    row's ``token`` equals the meta's decoded token at that position (s3d / converted rows)."""
    layers = [int(x) for x in args.layers.split(",")]
    meta: dict[str, dict] = {}
    for p in sorted(args.meta_dir.glob("*.json")):
        meta.update(json.loads(p.read_text()))
    want = {
        (iid, ly, int(pos))
        for iid, m in meta.items()
        for ly in layers
        for pos in m["positions_reported"]
    }
    got: Counter = Counter()
    bad_tok = []
    for path in args.readouts:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["id"], int(r["layer"]), int(r["pos"]))
            got[key] += 1
            m = meta.get(r["id"])
            if m is not None and "token" in r:
                exp = m["token"].get(str(int(r["pos"]) % m["n"]))
                if exp is not None and r["token"] != exp:
                    bad_tok.append((key, r["token"], exp))
    missing = want - set(got)
    extra = set(got) - want
    dup = [k for k, n in got.items() if n > 1]
    print(
        f"[check-cells] expected {len(want)}, got {len(got)}, missing {len(missing)}, "
        f"extra {len(extra)}, duplicated {len(dup)}, token mismatches {len(bad_tok)}"
    )
    for x in [*sorted(missing)[:5], *bad_tok[:5]]:
        print("  ", x)
    if missing or dup or bad_tok:
        sys.exit(2)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plans")
    p.add_argument("--limit-eval", type=int, default=0)
    p.add_argument("--ids", default="", help="csv of eval ids (a smoke plan; no probe plan)")
    p.add_argument("--tag", default="")
    p.set_defaults(fn=cmd_plans)
    c = sub.add_parser("check-prompts")
    c.add_argument("--render-dir", type=Path, required=True)
    c.set_defaults(fn=cmd_check_prompts)
    m = sub.add_parser("check-manifest")
    m.add_argument("--render-dir", type=Path, required=True)
    m.add_argument("--manifest", type=Path, required=True)
    m.set_defaults(fn=cmd_check_manifest)
    k = sub.add_parser("check-cells")
    k.add_argument("--meta-dir", type=Path, required=True)
    k.add_argument("--readouts", type=Path, nargs="+", required=True)
    k.add_argument("--layers", required=True)
    k.set_defaults(fn=cmd_check_cells)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
