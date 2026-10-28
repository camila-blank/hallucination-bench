"""Probe-check report (plan 0016 rev 5): premise, the standard probe-checked rate per region,
misses. Reads outputs/strongreject/probe_sites.jsonl (probe_modal.py, scored with plan 0015's
refusal_probe_v2) and rebuilds each arm's slim cell records from its cache (no API calls);
writes outputs/strongreject/<judge>/probe_report.{md,json}.

    uv run --no-sync python scripts/strongreject/probe_report.py --npz <refusal_probe_v2.npz>
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from hallucination_bench import REPO_ROOT
from hallucination_bench.evals.strongreject import probe_check as PC
from hallucination_bench.evals.strongreject import run as R

VALIDITY_MIN = 0.9  # plan 0015's pre-registered validity floor
OFFSET_CLS = {0: "tail0", 1: "tail1", 2: "tail2", 3: "tail3"}


def f(x, ci=True):
    if not x or x.get("value") is None:
        return "—"
    s = f"{x['value'] * 100:.1f}"
    return s + (f" [{x['ci'][0] * 100:.1f}, {x['ci'][1] * 100:.1f}]" if ci else "")


def pct(x):
    return "—" if x is None else f"{x * 100:.2f}"


def validity(npz: Path, variant: str) -> dict[tuple[int, str], bool]:
    """(layer, position class) -> valid: tail offsets by their OOF AUROC; content sites use the
    `assistant` offset's (plan 0015's secondary content rule)."""
    arr = np.load(npz)
    out = {}
    for k in arr.files:
        v, lay, name = k.split("/")
        if v != variant or name != "validity":
            continue
        la = int(lay[1:])
        val = arr[k]
        for o, cls in OFFSET_CLS.items():
            out[(la, cls)] = bool(val[o] >= VALIDITY_MIN)
        out[(la, "content")] = bool(val[0] >= VALIDITY_MIN)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="the probe artifact the sites were scored with")
    ap.add_argument("--variant", default="refusal-dm-v1")
    ap.add_argument("--arms", default="s3d-rl600,mlnla-rl600,jlens")
    ap.add_argument("--readouts-root", default="/workspace/hb-readouts-strongreject")
    ap.add_argument("--sites", default=str(REPO_ROOT / "outputs/strongreject/probe_sites.jsonl"))
    a = ap.parse_args()
    judge = R.Judge()
    valid = validity(Path(a.npz), a.variant)
    sites = [json.loads(x) for x in open(a.sites, encoding="utf-8")]
    probe = {(s["id"], s["layer"], s["pos"]): s for s in sites if s["variant"] == a.variant}
    arms = [x for x in a.arms.split(",") if x]
    out: dict = {
        "artifact": str(a.npz),
        "artifact_sha256": hashlib.sha256(Path(a.npz).read_bytes()).hexdigest(),
        "variant": a.variant,
        "valid": {f"L{la}/{c}": v for (la, c), v in sorted(valid.items())},
        "arms": {},
    }
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
        out["arms"][arm] = {
            "standard": PC.standard(recs, probe, items, "primary", valid),
            "secondary": PC.probe_checked(recs, probe, items, a.variant, "primary"),
        }
    out["premise"] = PC.premise([s for s in sites if s["variant"] == a.variant], items)
    root = R.model_root(Path("outputs"), judge)
    (root / "probe_report.json").write_text(json.dumps(out, indent=1) + "\n")
    md = [
        f"# strongreject probe check (plan 0016 rev 5): {a.variant} from `{a.npz}`",
        "",
        f"sha256 `{out['artifact_sha256'][:16]}`; invalid (layer, class) cells (validity < 0.9) "
        f"count as undetermined: {[k for k, v in out['valid'].items() if not v] or 'none'}.",
        "",
        "## Premise: share of items with refusal present (unsuccessful | soft_fail)",
        "",
        "| layer | `<|im_end|>` | `\\n` | `<|im_start|>` | `assistant` | content |",
        "|---|---|---|---|---|---|",
    ]
    for la, C in out["premise"][a.variant]["primary"].items():
        cells = [
            f"{f(C[c]['unsuccessful'], False)} \\| {f(C[c]['soft_fail'], False)}" if c in C else "—"
            for c in (*PC.TAIL_CLASSES, "content")
        ]
        md.append(f"| {la} | " + " | ".join(cells) + " |")
    for key, name in (("hc", "harmful_compliance"), ("pa", "persona_adoption")):
        md += [
            "",
            f"## Probe-checked rate, `{name}`, unsuccessful items (standard definition)",
            "",
            "rate = probe-contradicted claim readouts / readouts (pooled, %); claim rate = all "
            "counted claims / readouts; counts = claim readouts supported / contradicted / "
            "undetermined.",
            "",
            "| arm | layer | region | readouts | claim rate | probe-checked rate | supported | "
            "contradicted | undetermined |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for arm in arms:
            for la, L in out["arms"][arm]["standard"].items():
                for scope in PC.REGION_SCOPES:
                    x = L[f"{key}_{scope}"]
                    md.append(
                        f"| {arm} | {la} | {scope} | {x['n_readouts']} | {pct(x['claim_rate'])} | "
                        f"{pct(x['probe_checked_rate'])} | {x['n_supported']} | "
                        f"{x['n_contradicted']} | {x['n_undetermined']} |"
                    )
    md += [
        "",
        "## Misses (probe-present sites with no lens refusal claim, unsuccessful)",
        "",
        "| arm | layer | turn sites (4 template-tail tokens) | all sites |",
        "|---|---|---|---|",
    ]
    for arm in arms:
        for la, L in out["arms"][arm]["secondary"].items():
            md.append(
                f"| {arm} | {la} | {pct(L['miss_turn']['miss_share'])} | "
                f"{pct(L['miss_all']['miss_share'])} |"
            )
    (root / "probe_report.md").write_text("\n".join(md) + "\n")
    print(f"wrote {root / 'probe_report.md'}")


if __name__ == "__main__":
    main()
