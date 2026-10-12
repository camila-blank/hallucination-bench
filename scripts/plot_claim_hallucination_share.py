"""% of claims that are hallucinations, per method (and per layer for O-lens / ML-NLA step 600).

    uv run python scripts/plot_claim_hallucination_share.py [--out plots/full_run_v2.7]

A *claim* is any judged segment that is not `junk` (non-claim / off-topic) and not failed or
unknown. Hallucination share = (detail + false_claim segments) / claim segments, counted per
segment and pooled within an item; the 95% CI resamples items (2000 draws, seed 0). Streams
cells.jsonl, so it is memory-safe on the large 11-layer arms. Writes
claims_hallucination_share.png and claims_hallucination_share_by_layer.png.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / "outputs" / "wsbench_chat" / "claude-opus-5-5"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
NOT_CLAIM = {"junk", "non_claim", "off_topic", "failed", "unknown", "unknown_incorrect"}
METHODS = [
    ("s3d-rl600-all11", "O-lens RL (11 layers)", False),
    ("s3d-rl600", "O-lens RL (5 layers)", False),
    ("s3d-sft251", "O-lens SFT (5 layers)", False),
    ("mlnla-rl600-all11", "ML-NLA step 600 (11 layers)", False),
    ("mlnla-rl2-800-all11", "ML-NLA step 800 (11 layers)", False),
    ("nla-rl-L42", "NLA-RL (L42)", False),
    ("nla-sft-L42", "NLA-SFT (L42)", False),
    ("patchscopes", "Patchscopes (5 layers)", False),
    ("jlens", "J-lens (5 layers)", True),
    ("rlens", "R-lens (5 layers)", True),
    ("tuned", "Tuned lens (5 layers)", True),
    ("logit", "Logit lens (5 layers)", True),
    ("template-v3", "Template lens (5 layers)", True),
    ("sae-l42", "SAE (L42)", True),
]
LAYER_ARMS = [("s3d-rl600-all11", "O-lens RL"), ("mlnla-rl600-all11", "ML-NLA step 600")]


def tally(arm: str) -> dict:
    """{layer|'all': {item: [n_claims, n_false_claim, n_detail]}}."""
    out: dict = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    with (CELLS / arm / "cells.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            segs = (r.get("map") or {}).get("segments") or []
            for s in segs:
                leaf = s.get("leaf")
                if leaf in NOT_CLAIM:
                    continue
                for key in ("all", r["layer"]):
                    acc = out[key][r["id"]]
                    acc[0] += 1
                    acc[1] += leaf == "false_claim"
                    acc[2] += leaf == "detail"
    return out


def ratio_ci(per_item: dict, col: int, n_boot: int = 2000) -> tuple[float, float, float]:
    m = np.array(list(per_item.values()), dtype=float)
    point = m[:, col].sum() / m[:, 0].sum()
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(m), size=(n_boot, len(m)))
    num, den = m[:, col][idx].sum(1), m[:, 0][idx].sum(1)
    boots = num / den
    return 100 * point, 100 * np.percentile(boots, 2.5), 100 * np.percentile(boots, 97.5)


def with_hallu(per_item: dict) -> dict:
    return {k: [v[0], v[1], v[2], v[1] + v[2]] for k, v in per_item.items()}


def bar_plot(rows, title, out, height) -> None:
    rows = sorted(rows, key=lambda r: r[1][0])
    fig, ax = plt.subplots(figsize=(11.5, height), dpi=160)
    y = np.arange(len(rows))[::-1]
    fc = np.array([r[2] for r in rows])
    de = np.array([r[3] for r in rows])
    tot = np.array([r[1][0] for r in rows])
    lo = tot - np.array([r[1][1] for r in rows])
    hi = np.array([r[1][2] for r in rows]) - tot
    ax.barh(
        y, fc, height=0.66, color="#2a78d6", edgecolor="white", linewidth=1.5, label="False claim"
    )
    ax.barh(
        y,
        de,
        left=fc,
        height=0.66,
        color="#eda100",
        edgecolor="white",
        linewidth=1.5,
        label="Detail",
    )
    ax.errorbar(tot, y, xerr=[lo, hi], fmt="none", ecolor=TEXT, elinewidth=1, capsize=2)
    for yi, vi, hii, n in zip(y, tot, hi, [r[4] for r in rows], strict=True):
        ax.text(vi + hii + 0.8, yi, f"{vi:.1f}%  (n={n:,})", va="center", fontsize=10, color=TEXT)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=12)
    ax.set_xlim(0, max(10.0, float((tot + hi).max()) * 1.25))
    ax.set_xlabel(
        "Hallucinated claims as % of all claims  — lower is better", fontsize=13, color=TEXT
    )
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=12, colors=MUTED, length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.set_title(title, fontsize=16, color=TEXT, loc="left", pad=34)
    ax.legend(fontsize=11, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    fig.text(
        0.01,
        0.005,
        "Claim = any judged segment except junk / non-claim (failed excluded); n = claim "
        "segments. Bars stack false claim + detail; whiskers: item-bootstrap 95% CI of the total.\n"
        "† token lens, judged via its summary. Claim-map judge v2.7 (plan 0002 rev 18), pre-gate.",
        fontsize=9,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "plots" / "full_run_v2.7"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    tallies = {arm: tally(arm) for arm, _, _ in METHODS}

    def row(label: str, per_item: dict):
        m = with_hallu(per_item)
        n = int(sum(v[0] for v in m.values()))
        return (label, ratio_ci(m, 3), ratio_ci(m, 1)[0], ratio_ci(m, 2)[0], n)

    rows = [row(f"{lab}{' †' if tok else ''}", tallies[arm]["all"]) for arm, lab, tok in METHODS]
    bar_plot(
        rows,
        "% of claims that are hallucinations, per method",
        out / "claims_hallucination_share.png",
        7.8,
    )
    layer_rows = [
        r for r, (arm, _, _) in zip(rows, METHODS, strict=True) if arm not in ("s3d-rl600-all11",)
    ]
    for arm, name in LAYER_ARMS:
        for L in range(20, 61, 4):
            layer_rows.append(row(f"{name} L{L}", tallies[arm][L]))
    bar_plot(
        layer_rows,
        "% of claims that are hallucinations, per method and layer",
        out / "claims_hallucination_share_by_layer.png",
        13.5,
    )
    for r in sorted(rows, key=lambda r: r[1][0]):
        print(
            f"{r[0]:32s} {r[1][0]:5.1f} [{r[1][1]:.1f}, {r[1][2]:.1f}]  FC {r[2]:.1f}  "
            f"D {r[3]:.1f}  n={r[4]}"
        )


if __name__ == "__main__":
    main()
