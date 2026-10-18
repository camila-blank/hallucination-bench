"""Story eval results under the claim-map judge (plan 0007, claimmap-v2.7, scoring rev 20).

    uv run --no-sync python scripts/plot_story.py [--out plots/story_v2.7]

Writes ``story.png``: per lens, the three hallucination components (false claim = a wrong
proposition / concept; detail = a wrong specific with no concept of its own; category
substitution). Left: share of readout text, stacked. Right: share of readouts with at least one
segment of each component; these overlap (a readout can have both), so they are drawn side by
side, with a tick for "any hallucination". Whiskers: item-bootstrap 95% CI. Multi-layer lenses
are pooled over their 11 layers.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = "claude-opus-5-5"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
ARMS = [
    ("s3d-rl600-all11", "O-lens RL600\n(11 layers)"),
    ("mlnla-rl600-all11", "ML-NLA step 600\n(11 layers)"),
    ("nla-rl-L42", "NLA-RL\n(L42)"),
    ("jlens", "J-lens*\n(11 layers)"),
]
COMPONENTS = [  # same colours as plot_full_run.py
    ("false_claim", "False claim (wrong concept / proposition)", "#e34948"),
    ("detail", "Detail (wrong name, number, wording)", "#eda100"),
    ("substitution_category", "Category substitution", "#8b3fb5"),
]


def overall(root: Path, arm: str) -> dict:
    return json.loads((root / arm / "summary_v2.json").read_text())["overall"]


def pct(r: dict) -> tuple[float, float, float]:
    return 100 * r["value"], 100 * r["ci95"][0], 100 * r["ci95"][1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--story-out", type=Path, default=ROOT / "outputs")
    ap.add_argument("--out", type=Path, default=ROOT / "plots" / "story_v2.7")
    args = ap.parse_args()
    root = args.story_out / "story" / MODEL
    data = {a: overall(root, a) for a, _ in ARMS}

    plt.rcParams.update({"font.size": 15, "axes.edgecolor": MUTED, "text.color": TEXT})
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.8))
    x = np.arange(len(ARMS))

    # left: share of readout text, stacked components + CI of the total
    ax = axes[0]
    bottom = np.zeros(len(ARMS))
    for key, _label, color in COMPONENTS:
        v = np.array([pct(data[a]["char_share_leaf"][key])[0] for a, _ in ARMS])
        ax.bar(x, v, 0.6, bottom=bottom, color=color, edgecolor="white", linewidth=2)
        bottom += v
    tot = [pct(data[a]["char_share_verdict"]["hallucination"]) for a, _ in ARMS]
    ax.errorbar(
        x,
        [t[0] for t in tot],
        yerr=[[t[0] - t[1] for t in tot], [t[2] - t[0] for t in tot]],
        fmt="none",
        ecolor=TEXT,
        elinewidth=1.5,
        capsize=5,
    )
    for xi, t in zip(x, tot, strict=True):
        ax.text(
            xi, t[2] + 0.8, f"{t[0]:.0f}%", ha="center", va="bottom", fontsize=15, fontweight="bold"
        )
    ax.set_ylabel("% of readout text hallucinated")
    ax.set_title("Share of readout text", fontsize=18, loc="left", pad=12)
    ax.set_ylim(0, 52)

    # right: % of readouts with >= 1 segment, components side by side + "any" tick
    ax = axes[1]
    sub = 0.26
    for i, (key, _label, color) in enumerate(COMPONENTS):
        vals = [pct(data[a]["rates"][key]["all"]) for a, _ in ARMS]
        ax.bar(
            x + (i - 1) * sub,
            [v[0] for v in vals],
            sub * 0.9,
            color=color,
            yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]],
            error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 2},
        )
    tot = [pct(data[a]["rates"]["verdict_hallucination"]["all"]) for a, _ in ARMS]
    ax.scatter(x, [t[0] for t in tot], marker="_", s=2200, color=TEXT, linewidths=2.5, zorder=3)
    for xi, t in zip(x, tot, strict=True):
        ax.text(
            xi, t[0] + 1.5, f"{t[0]:.0f}%", ha="center", va="bottom", fontsize=15, fontweight="bold"
        )
    ax.set_ylabel("% of readouts with ≥1 such segment")
    ax.set_title("Share of readouts (— = any hallucination)", fontsize=18, loc="left", pad=12)
    ax.set_ylim(0, 108)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in ARMS])
        ax.tick_params(axis="x", length=0, pad=8)
        ax.yaxis.grid(True, color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in COMPONENTS]
    fig.legend(
        handles,
        [lab for _, lab, _ in COMPONENTS],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=14,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "Readout hallucination on “tell me a story” prompts — claim-map judge v2.7 (Opus 5.5)",
        fontsize=20,
        x=0.01,
        ha="left",
        y=0.995,
    )
    fig.text(
        0.01,
        0.935,
        "20 story items × 8 read sites. Whiskers: item-bootstrap 95% CI. *J-lens judged via its "
        "token-bag summary. Pre-gate (the held-out gate is chat gold).",
        fontsize=13,
        color=MUTED,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    args.out.mkdir(parents=True, exist_ok=True)
    p = args.out / "story.png"
    fig.savefig(p, dpi=170, bbox_inches="tight")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
