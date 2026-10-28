"""Slide plot (plan 0016 rev 5): compliance-claim hallucination rate on unsuccessful items by layer,
one line per method, one panel per region (body / tail / turn); second row = the probe-checked
rate (claims at sites where refusal is represented). Reads probe_report.json.

    uv run --no-sync python scripts/strongreject/plot_rates.py
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hallucination_bench import REPO_ROOT

ARMS = {"s3d-rl600": "O-lens (s3d-rl600)", "mlnla-rl600": "multilayer NLA", "jlens": "J-lens"}
COLORS = {"s3d-rl600": "#1f77b4", "mlnla-rl600": "#ff7f0e", "jlens": "#2ca02c"}
REGIONS = ("body", "tail", "turn")
TITLES = {
    "body": "body (before last 64)",
    "tail": "tail (last 64, excl. turn)",
    "turn": "turn (last 4 tokens)",
}


def main() -> None:
    src = REPO_ROOT / "outputs/strongreject/claude-opus-5-5/probe_report.json"
    rep = json.loads(src.read_text())
    plt.rcParams.update({"font.size": 18})
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, dpi=160)
    rows = (
        ("claim_rate", "compliance claims\n(% of readouts)"),
        ("probe_checked_rate", "probe-checked: claims where\nrefusal is represented (%)"),
    )
    for r, (metric, ylab) in enumerate(rows):
        for c, reg in enumerate(REGIONS):
            ax = axes[r, c]
            for arm, label in ARMS.items():
                if arm not in rep["arms"]:
                    continue
                S = rep["arms"][arm]["standard"]
                layers = sorted(S, key=int)
                ys = [S[la][f"hc_{reg}"][metric] for la in layers]
                ax.plot(
                    [int(x) for x in layers],
                    [None if y is None else 100 * y for y in ys],
                    marker="o",
                    lw=3,
                    ms=9,
                    color=COLORS[arm],
                    label=label,
                )
            if r == 0:
                ax.set_title(TITLES[reg], fontsize=20)
            if c == 0:
                ax.set_ylabel(ylab, fontsize=18)
            if r == 1:
                ax.set_xlabel("layer", fontsize=18)
            ax.set_xticks([28, 36, 44, 52])
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.3)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=18, frameon=False)
    fig.suptitle(
        "strongreject: lens claims that Qwen3.6-27B will comply with a jailbreak it refused 10/10",
        fontsize=22,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = REPO_ROOT / "plots/strongreject/compliance_hallucination_by_layer.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
