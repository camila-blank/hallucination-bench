"""Story vs chat under the same claim-map judge (plan 0007, claimmap-v2.7, scoring rev 20).

    uv run --no-sync python scripts/plot_story.py [--chat-out DIR] [--out plots/story_v2.7]

Writes ``story_vs_chat.png``: per lens, a chat bar and a story bar, each stacked by the three
hallucination components (false claim = a wrong proposition / concept; detail = a wrong specific
with no concept of its own; category substitution). Left: share of readout text. Right: share of
readouts with at least one segment of that component. On the right the components overlap (a
readout can have both), so they are drawn side by side, not stacked. Whiskers: item-bootstrap
95% CI of the total. Multi-layer lenses are pooled over their 11 layers.
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
BANKS = ("chat", "story")


def overall(root: Path, arm: str) -> dict:
    return json.loads((root / arm / "summary_v2.json").read_text())["overall"]


def pct(r: dict) -> tuple[float, float, float]:
    return 100 * r["value"], 100 * r["ci95"][0], 100 * r["ci95"][1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chat-out", type=Path, default=Path("/workspace/hb-taxonomy-v2/outputs"))
    ap.add_argument("--story-out", type=Path, default=ROOT / "outputs")
    ap.add_argument("--out", type=Path, default=ROOT / "plots" / "story_v2.7")
    args = ap.parse_args()
    roots = {
        "chat": args.chat_out / "wsbench_chat" / MODEL,
        "story": args.story_out / "story" / MODEL,
    }
    data = {(a, b): overall(roots[b], a) for a, _ in ARMS for b in BANKS}

    plt.rcParams.update({"font.size": 15, "axes.edgecolor": MUTED, "text.color": TEXT})
    fig, axes = plt.subplots(1, 2, figsize=(19, 8.2), gridspec_kw={"width_ratios": [1, 1.25]})
    x = np.arange(len(ARMS)) * 2.6
    off = {"chat": -0.55, "story": 0.55}
    w = 1.0

    # left: share of readout text, stacked components + CI of the total
    ax = axes[0]
    for b in BANKS:
        bottom = np.zeros(len(ARMS))
        for key, _label, color in COMPONENTS:
            v = np.array([pct(data[(a, b)]["char_share_leaf"][key])[0] for a, _ in ARMS])
            ax.bar(
                x + off[b],
                v,
                w,
                bottom=bottom,
                color=color,
                edgecolor="white",
                linewidth=2,
                alpha=1.0 if b == "story" else 0.45,
            )
            bottom += v
        tot = [pct(data[(a, b)]["char_share_verdict"]["hallucination"]) for a, _ in ARMS]
        lo = [t[0] - t[1] for t in tot]
        hi = [t[2] - t[0] for t in tot]
        ax.errorbar(
            x + off[b],
            [t[0] for t in tot],
            yerr=[lo, hi],
            fmt="none",
            ecolor=TEXT,
            elinewidth=1.5,
            capsize=4,
        )
        for xi, t in zip(x + off[b], tot, strict=True):
            ax.text(
                xi,
                t[2] + 1.0,
                f"{t[0]:.0f}",
                ha="center",
                va="bottom",
                fontsize=14,
                color=TEXT,
                fontweight="bold" if b == "story" else "normal",
            )
            ax.text(xi, -1.2, b, ha="center", va="top", fontsize=12.5, color=MUTED)
    ax.set_ylabel("% of readout text hallucinated")
    ax.set_title("Share of readout text", fontsize=18, loc="left", pad=12)
    ax.set_ylim(0, 52)

    # right: % of readouts with >= 1 segment, components side by side + total marker
    ax = axes[1]
    sub = 0.31
    for b in BANKS:
        for i, (key, _label, color) in enumerate(COMPONENTS):
            vals = [pct(data[(a, b)]["rates"][key]["all"]) for a, _ in ARMS]
            xs = x + off[b] + (i - 1) * sub
            ax.bar(
                xs,
                [v[0] for v in vals],
                sub * 0.92,
                color=color,
                alpha=1.0 if b == "story" else 0.45,
            )
        tot = [pct(data[(a, b)]["rates"]["verdict_hallucination"]["all"]) for a, _ in ARMS]
        ax.scatter(
            x + off[b], [t[0] for t in tot], marker="_", s=900, color=TEXT, linewidths=2.5, zorder=3
        )
        for xi, t in zip(x + off[b], tot, strict=True):
            ax.text(
                xi,
                t[0] + 1.5,
                f"{t[0]:.0f}",
                ha="center",
                va="bottom",
                fontsize=14,
                color=TEXT,
                fontweight="bold" if b == "story" else "normal",
            )
            ax.text(xi, -2.4, b, ha="center", va="top", fontsize=12.5, color=MUTED)
    ax.set_ylabel("% of readouts with ≥1 such segment")
    ax.set_title("Share of readouts (— = any hallucination)", fontsize=18, loc="left", pad=12)
    ax.set_ylim(0, 108)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in ARMS])
        ax.tick_params(axis="x", length=0, pad=26)
        ax.yaxis.grid(True, color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in COMPONENTS]
    labels = [lab for _, lab, _ in COMPONENTS]
    handles += [
        plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.45),
        plt.Rectangle((0, 0), 1, 1, color=MUTED),
    ]
    labels += ["chat bank (149 items, faded)", "story bank (20 items)"]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=14,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "Readout hallucination on stories vs chat — claim-map judge v2.7 (Opus 5.5)",
        fontsize=21,
        x=0.01,
        ha="left",
        y=0.99,
    )
    fig.text(
        0.01,
        0.925,
        "Same judge, same lenses; stories roughly double the O-lens rate. Whiskers: item-bootstrap "
        "95% CI. *J-lens judged via its token-bag summary. Pre-gate; story n = 20 items.",
        fontsize=13,
        color=MUTED,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.1, 1, 0.92))
    args.out.mkdir(parents=True, exist_ok=True)
    p = args.out / "story_vs_chat.png"
    fig.savefig(p, dpi=170, bbox_inches="tight")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
