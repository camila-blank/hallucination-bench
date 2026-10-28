"""Plan 0015 rev 8 (D): the HB headline figure from ``report.json`` (slide quality).

False-refusal-claim rate on ``never_refused`` (verified ``refusal_intent`` readouts / readouts,
prompt-weighted, 95% prompt-bootstrap CI) by layer, one line per method; panels = tail sites
(headline) and all prompt sites. Writes ``plots/overrefusal_hb_rate.png``.

    uv run --no-sync python scripts/overrefusal/plot_hb.py [outputs/overrefusal/<judge>/report.json]
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
LABEL = {
    "s3d-rl600": "O-lens (s3d-rl600)",
    "mlnla-rl600-all11": "Multilayer NLA",
    "jlens": "J-lens",
}
COLOR = {"s3d-rl600": "#1f6feb", "mlnla-rl600-all11": "#d4762a", "jlens": "#2a9d55"}


def main() -> None:
    src = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else ROOT / "outputs/overrefusal/claude-sonnet-5-5/report.json"
    )
    rep = json.loads(src.read_text())
    plt.rcParams.update({"font.size": 18})
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)
    for ax, sites, title in zip(
        axes, ("tail", "all"), ("Tail sites (headline)", "All prompt sites"), strict=True
    ):
        for arm, a in rep["arms"].items():
            sc = a["scores"]
            ls = sc["layers"]
            s = [
                sc["per_layer"][str(la)]["never_refused"]["refusal"][sites]["readout"] for la in ls
            ]
            m = [100 * x["mean"] for x in s]
            lo = [100 * (x["mean"] - x["lo"]) for x in s]
            hi = [100 * (x["hi"] - x["mean"]) for x in s]
            ax.errorbar(
                ls,
                m,
                yerr=[lo, hi],
                marker="o",
                ms=9,
                lw=3,
                capsize=5,
                color=COLOR.get(arm),
                label=LABEL.get(arm, arm),
            )
        ax.set_title(title, fontsize=22)
        ax.set_xlabel("Layer", fontsize=20)
        ax.set_xticks(ls)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("False refusal claims (% readouts)", fontsize=20)
    n = rep["populations"].get("never_refused")
    fig.suptitle(
        f"HallucinationBench overrefusal: claims of refusal on never-refused prompts (n={n})",
        fontsize=22,
    )
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=18, frameon=False)
    fig.tight_layout()
    out = ROOT / "plots/overrefusal_hb_rate.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
