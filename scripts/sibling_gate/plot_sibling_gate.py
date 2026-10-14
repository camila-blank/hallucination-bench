"""Plan 0009 figures: per arm, the share of sibling units by final status and by rung-1 status.

    uv run python scripts/sibling_gate/plot_sibling_gate.py [--out outputs/sibling_gate] \\
        [--plots plots/sibling_gate]

Reads ``<out>/report.json`` (``hbench sibling-gate report``). Writes final_status.png and
rung1_status.png. Gated (hallucination) statuses are the two warm segments on the left.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#ffffff"
FINAL = [  # gated first (warm), then the not-counted outcomes
    ("unstable", "Unstable (rung 1, gated)", "#e34948"),
    ("probe_unconfirmed", "Probe unconfirmed (gated)", "#eb6834"),
    ("probe_faithful", "Probe faithful", "#1baf7a"),
    ("probe_nearby", "Probe faithful, nearby layer", "#008300"),
    ("cross_method", "Cross-method agreement", "#2a78d6"),
    ("probe_na", "Probe n/a", "#b9b8b2"),
    ("resample_failed", "Resample failed", "#52514e"),
    ("pending", "Pending", "#e6e5e0"),
]
RUNG1 = [
    ("unstable", "Unstable", "#e34948"),
    ("stable", "Stable", "#2a78d6"),
    ("sparse", "Sparse", "#eda100"),
    ("deterministic", "Deterministic (token arm)", "#4a3aa7"),
    ("resample_failed", "Resample failed", "#52514e"),
]


def stacked(rows: dict, key: str, spec: list, title: str, out: Path) -> None:
    arms = sorted(rows, key=lambda a: -rows[a]["n_units"])
    fig, ax = plt.subplots(figsize=(12, 0.5 * len(arms) + 2.4), dpi=160)
    y = np.arange(len(arms))[::-1]
    left = np.zeros(len(arms))
    for k, label, color in spec:
        n = np.array([rows[a][key].get(k, [0, 0])[0] for a in arms], dtype=float)
        tot = np.array([max(rows[a]["n_units"], 1) for a in arms], dtype=float)
        w = 100 * n / tot
        ax.barh(
            y, w, left=left, height=0.66, color=color, edgecolor=SURFACE, linewidth=2, label=label
        )
        left += w
    for yi, a in zip(y, arms, strict=True):
        ax.text(101, yi, f"n={rows[a]['n_units']:,}", va="center", fontsize=11, color=MUTED)
    ax.set_yticks(y, arms, fontsize=12)
    ax.set_xlim(0, 112)
    ax.set_xlabel("% of sibling units", fontsize=13, color=TEXT)
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=12, colors=MUTED, length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, fontsize=16, color=TEXT, loc="left", pad=40)
    ax.legend(fontsize=10, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs" / "sibling_gate"))
    ap.add_argument("--plots", default=str(ROOT / "plots" / "sibling_gate"))
    a = ap.parse_args()
    rows = json.loads((Path(a.out) / "report.json").read_text())
    if not rows:
        raise SystemExit("report.json has no arms: run `hbench sibling-gate report` first")
    d = Path(a.plots)
    d.mkdir(parents=True, exist_ok=True)
    stacked(
        rows, "final", FINAL, "Sibling gate: final status of sibling units", d / "final_status.png"
    )
    stacked(
        rows,
        "rung1",
        RUNG1,
        "Sibling gate: rung-1 (10-sample resample) status",
        d / "rung1_status.png",
    )
    print(f"wrote {d / 'final_status.png'}, {d / 'rung1_status.png'}")


if __name__ == "__main__":
    main()
