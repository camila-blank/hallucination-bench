"""Hallucination per layer for O-lens RL / SFT and ML-NLA step 600 (line plot, two panels).

    uv run python scripts/plot_layer_hallucination.py [--out plots/full_run_v2.7]

Left: readout-level hallucination rate (summary_v2.json by_layer). Right: % of claims that are
hallucinations (same definition and item-bootstrap CI as plot_claim_hallucination_share.py).
The single-layer NLA-RL / NLA-SFT arms (L42) are drawn as reference points. Writes
layers_hallucination.png.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_claim_hallucination_share import ratio_ci, tally, with_hallu  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / "outputs" / "wsbench_chat" / "claude-opus-5-5"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
LINES = [
    ("s3d-rl600-all11", "O-lens RL", "#2a78d6"),
    ("s3d-sft251", "O-lens SFT", "#4a3aa7"),
    ("mlnla-rl600-all11", "ML-NLA step 600", "#eb6834"),
]
POINTS = [
    ("nla-rl-L42", "NLA-RL (L42)", "#e34948", "s"),
    ("nla-sft-L42", "NLA-SFT (L42)", "#008300", "D"),
]


def readout_rate(block: dict) -> tuple[float, float, float]:
    r = block["rates"]["hallucination"]["all"]
    return 100 * r["value"], 100 * r["ci95"][0], 100 * r["ci95"][1]


def summary(arm: str) -> dict:
    return json.loads((CELLS / arm / "summary_v2.json").read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "plots" / "full_run_v2.7"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), dpi=160)
    panels = [
        (axes[0], "Readouts with ≥ 1 hallucination (%)", "Readout-level hallucination rate"),
        (axes[1], "Hallucinated claims as % of all claims", "% of claims that are hallucinations"),
    ]
    for arm, name, color in LINES:
        bl = summary(arm)["by_layer"]
        layers = sorted(int(k) for k in bl)
        t = tally(arm)
        series = [
            [readout_rate(bl[str(L)]) for L in layers],
            [ratio_ci(with_hallu(t[L]), 4) for L in layers],
        ]
        for (ax, _, _), vals in zip(panels, series, strict=True):
            v, lo, hi = zip(*vals, strict=True)
            ax.fill_between(layers, lo, hi, color=color, alpha=0.15, linewidth=0)
            ax.plot(layers, v, color=color, linewidth=2.2, marker="o", markersize=5, label=name)
    for arm, name, color, marker in POINTS:
        vals = [readout_rate(summary(arm)["overall"]), ratio_ci(with_hallu(tally(arm)["all"]), 4)]
        for (ax, _, _), (v, lo, hi) in zip(panels, vals, strict=True):
            ax.errorbar(
                [42],
                [v],
                yerr=[[v - lo], [hi - v]],
                fmt=marker,
                color=color,
                markersize=8,
                markeredgecolor="white",
                capsize=3,
                label=name,
                zorder=4,
            )
    for ax, ylabel, title in panels:
        ax.set_xticks(range(20, 61, 4))
        ax.set_xlabel("Layer", fontsize=13, color=TEXT)
        ax.set_ylabel(ylabel + "  — lower is better", fontsize=12, color=TEXT)
        ax.set_title(title, fontsize=14, color=TEXT, loc="left")
        ax.set_ylim(0, 100 if ax is axes[0] else 60)
        ax.yaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=11, colors=MUTED, length=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.text(
        0.01,
        0.005,
        "Hallucination = false claim, detail or category substitution. Bands / whiskers: "
        "item-bootstrap 95% CI. NLA-RL / NLA-SFT are single-layer (L42) reference points.\n"
        "Claim-map judge v2.7 (scoring rev 20), pre-gate.",
        fontsize=9,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(out / "layers_hallucination.png")
    plt.close(fig)
    print("wrote", out / "layers_hallucination.png")


if __name__ == "__main__":
    main()
