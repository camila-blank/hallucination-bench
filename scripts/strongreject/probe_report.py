"""Probe-check report (plan 0016 rev 4): premise, probe-checked hallucination rates, misses.
Reads outputs/strongreject/probe_sites.jsonl (probe_modal.py) and rebuilds each arm's slim cell
records from its cache (no API calls); writes outputs/strongreject/<judge>/probe_report.{md,json}.

    uv run --no-sync python scripts/strongreject/probe_report.py [--arms a,b,c]
"""

import argparse
import json
from pathlib import Path

from hallucination_bench import REPO_ROOT
from hallucination_bench.evals.strongreject import probe_check as PC
from hallucination_bench.evals.strongreject import run as R

PRIMARY = "refusal-dm-v1"  # refusal-beh-v1 is underpowered (34 + 34 < 60, plan 0015 rule)


def f(x, ci=True):
    if not x or x.get("value") is None:
        return "—"
    s = f"{x['value'] * 100:.1f}"
    return s + (f" [{x['ci'][0] * 100:.1f}, {x['ci'][1] * 100:.1f}]" if ci else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="s3d-rl600,mlnla-rl600,jlens")
    ap.add_argument("--readouts-root", default="/workspace/hb-readouts-strongreject")
    ap.add_argument("--sites", default=str(REPO_ROOT / "outputs/strongreject/probe_sites.jsonl"))
    a = ap.parse_args()
    judge = R.Judge()
    sites = [json.loads(x) for x in open(a.sites, encoding="utf-8")]
    out: dict = {"premise": None, "arms": {}}
    arms = [x for x in a.arms.split(",") if x]
    items = None
    for arm in arms:
        A = R.open_arm(
            arm,
            Path(a.readouts_root) / arm / R.READOUTS_FILE,
            Path("outputs"),
            judge,
            allow_missing=True,
            readonly=True,
        )
        items = A.items
        recs = []
        for c in A.cells:
            r = R.cell_record(A, c, judge)
            r.pop("mentions")
            if r["status"] not in ("ok", "pending", "failed"):
                r["status"] = "failed" if r["status"].startswith("failed") else "pending"
            recs.append(r)
        out["arms"][arm] = {}
        for thr in PC.THRESHOLDS:
            probe = {(s["id"], s["layer"], s["pos"]): s for s in sites if s["variant"] == PRIMARY}
            out["arms"][arm][thr] = PC.probe_checked(recs, probe, items, PRIMARY, thr)
    out["premise"] = PC.premise(sites, items)
    root = R.model_root(Path("outputs"), judge)
    (root / "probe_report.json").write_text(json.dumps(out, indent=1) + "\n")
    md = ["# strongreject probe check (plan 0016 rev 4; plan 0015's refusal probe)", ""]
    md += ["## Premise: share of items with refusal present (unsuccessful | soft_fail)", ""]
    md += [
        "| variant | threshold | layer | " + " | ".join(PC.TAIL_CLASSES) + " | content |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for v, T in out["premise"].items():
        for thr, Ls in T.items():
            for la, C in Ls.items():
                cells = [
                    f"{f(C[c]['unsuccessful'], False)} \\| {f(C[c]['soft_fail'], False)}"
                    if c in C
                    else "—"
                    for c in (*PC.TAIL_CLASSES, "content")
                ]
                md.append(f"| {v} | {thr} | {la} | " + " | ".join(cells) + " |")
    for thr in PC.THRESHOLDS:
        md += [
            "",
            f"## Probe-checked compliance rates, unsuccessful items ({PRIMARY}, {thr})",
            "",
            "| arm | layer | hc tail original | hc tail probe-contradicted | hc all original | "
            "hc all probe-contradicted | hc claims (all) probe-consistent share | pa all "
            "original | pa all probe-contradicted | miss share (tail) | miss share (all) |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for arm in arms:
            for la, L in out["arms"][arm][thr].items():
                sc = L["hc_all"]["share_probe_consistent"]
                md.append(
                    f"| {arm} | {la} | {f(L['hc_tail']['original'])} | "
                    f"{f(L['hc_tail']['probe_contradicted'])} | {f(L['hc_all']['original'])} | "
                    f"{f(L['hc_all']['probe_contradicted'])} | "
                    f"{'—' if sc is None else f'{sc * 100:.1f}'} "
                    f"(n={L['hc_all']['n_claims']}) | {f(L['pa_all']['original'], False)} | "
                    f"{f(L['pa_all']['probe_contradicted'], False)} | "
                    f"{f({'value': L['miss_tail']['miss_share'], 'ci': [None]}, False)} | "
                    f"{f({'value': L['miss_all']['miss_share'], 'ci': [None]}, False)} |"
                )
    (root / "probe_report.md").write_text("\n".join(md) + "\n")
    print(f"wrote {root / 'probe_report.md'}")


if __name__ == "__main__":
    main()
