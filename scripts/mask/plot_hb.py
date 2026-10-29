# ruff: noqa: E501  (table / label literals)
"""Plan 0017 rev 5 figure: the mask DIAGNOSTIC rate (not a hallucination score since 2026-10-29), i.e. the verified lie-claim
readout rate on gated honest_resisted items, by layer, one line per method, one panel per site
set (prompt / response / all), with 95% item-bootstrap CIs. Reads ``report.json``.

    uv run --no-sync python scripts/mask/plot_hb.py [--report outputs/mask/claude-opus-5-5/report.json]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
METHODS = {
    "s3d-rl600": ("O-lens (s3d-rl600)", "#1f77b4", "o"),
    "mlnla-rl600-all11": ("Multilayer NLA", "#ff7f0e", "s"),
    "jlens": ("J-lens", "#2ca02c", "^"),
}
SITE_SETS = ("prompt", "response", "all")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(REPO / "outputs/mask/claude-opus-5-5/report.json"))
    ap.add_argument("--out", default=str(REPO / "plots/0017_mask_hb_score.png"))
    a = ap.parse_args()
    rep = json.loads(Path(a.report).read_text())
    arms = rep["gate"]["arms"]
    plt.rcParams.update({"font.size": 16})
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), sharey=True)
    ymax = 0.0
    for ax, site in zip(axes, SITE_SETS, strict=True):
        for arm, (label, color, marker) in METHODS.items():
            if arm not in arms:
                continue
            xs, ys, lo, hi = [], [], [], []
            for key, row in sorted(arms[arm].items()):
                layer, st = key.split(":")
                if st != site:
                    continue
                c = row["9"]["honest_resisted"]
                if c.get("readout_rate") is None:
                    continue
                xs.append(int(layer[1:]))
                ys.append(c["readout_rate"])
                ci = c.get("readout_ci") or [c["readout_rate"]] * 2
                lo.append(c["readout_rate"] - ci[0])
                hi.append(ci[1] - c["readout_rate"])
                ymax = max(ymax, ci[1])
            ax.errorbar(
                xs,
                ys,
                yerr=[lo, hi],
                label=label,
                color=color,
                marker=marker,
                markersize=9,
                linewidth=2.5,
                capsize=5,
            )
        ax.set_title(f"{site} sites", fontsize=20)
        ax.set_xlabel("layer", fontsize=18)
        ax.set_xticks([28, 36, 44, 52])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("verified lie-claim readout rate\n(gated honest_resisted)", fontsize=17)
    for ax in axes:
        ax.set_ylim(0, max(ymax * 1.15, 0.01))
    n = rep["gate"]["arms"]
    fig.suptitle(
        "MASK: lens claims the model is lying when it was honest under pressure",
        fontsize=21,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=16, frameon=False
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=170, bbox_inches="tight")
    print(f"wrote {a.out} ({len(n)} methods)")


if __name__ == "__main__":
    main()
