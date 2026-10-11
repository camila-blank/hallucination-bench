"""Plots of the claim-map pilot (plan 0002) across lenses: readout-level rates and text shares.

    uv run python scripts/plot_claim_map_pilot.py [--out plots/claim_map_pilot]

Reads outputs/wsbench_chat/<model>/<arm>/summary_v2.json for the arms below and writes four
PNGs: {rates,share}_{hallucination,all}.png. Bars carry their values (relief for the low-contrast
palette slots); whiskers are item-bootstrap 95% CIs.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = "claude-opus-5-5"
ARMS = [("s3d-rl600", "O-lens RL (L44)"), ("nla-rl-L42", "NLA-RL (L42)"), ("jlens", "J-lens (L44)")]
LENS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # reference categorical slots 1-3 (all-pairs safe)
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"

HALLU = [("hallucination", "Hallucination"), ("false_claim", "False claim"), ("detail", "Detail")]
# readout-level rate keys / char-share keys and labels for the "all categories" views
ALL_RATES = [
    ("hallucination", "Hallucination"),
    ("false_claim", "↳ False claim"),
    ("detail", "↳ Detail"),
    ("verdict_probe_candidate", "Probe candidate"),
    ("verdict_correct", "Correct"),
    ("verdict_unresolved", "Unverifiable"),
    ("verdict_junk", "Junk (off-topic)"),
]
ALL_SHARES = [
    ("leaf:false_claim", "Hallucination: false claim"),
    ("leaf:detail", "Hallucination: detail"),
    ("verdict:probe_candidate", "Probe candidate"),
    ("verdict:correct", "Correct"),
    ("verdict:unresolved", "Unverifiable"),
    ("verdict:junk", "Junk (off-topic)"),
    ("verdict:non_claim", "Non-claim"),
]
STACK_COLORS = ["#e34948", "#eda100", "#4a3aa7", "#1baf7a", "#2a78d6", "#e87ba4", "#b9b8b2"]


def load(arm: str) -> dict:
    p = ROOT / "outputs" / "wsbench_chat" / MODEL / arm / "summary_v2.json"
    return json.loads(p.read_text())["overall"]


def rate(o: dict, key: str) -> tuple[float, float, float]:
    r = o["rates"][key]["all"]
    return 100 * r["value"], 100 * r["ci95"][0], 100 * r["ci95"][1]


def share(o: dict, key: str) -> tuple[float, float, float]:
    kind, name = key.split(":")
    r = o["char_share_leaf" if kind == "leaf" else "char_share_verdict"][name]
    return 100 * r["value"], 100 * r["ci95"][0], 100 * r["ci95"][1]


def style(ax, ylabel: str) -> None:
    ax.set_ylabel(ylabel, fontsize=15, color=TEXT)
    ax.tick_params(axis="both", labelsize=13, colors=MUTED, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)


def grouped(data: list[dict], cats, getter, ylabel: str, title: str, out: Path) -> None:
    n, k = len(cats), len(ARMS)
    fig, ax = plt.subplots(figsize=(max(8, 1.9 * n + 2), 6.2), dpi=160)
    width = 0.8 / k
    x = np.arange(n)
    ymax = 0.0
    for j, ((_, label), color, o) in enumerate(zip(ARMS, LENS_COLORS, data, strict=True)):
        vals = [getter(o, c) for c, _ in cats]
        v = np.array([a for a, _, _ in vals])
        lo = v - np.array([b for _, b, _ in vals])
        hi = np.array([c for _, _, c in vals]) - v
        xs = x + (j - (k - 1) / 2) * width
        ax.bar(xs, v, width * 0.92, color=color, label=label, edgecolor="white", linewidth=2)
        ax.errorbar(xs, v, yerr=[lo, hi], fmt="none", ecolor=TEXT, elinewidth=1.2, capsize=3)
        for xi, vi, hii in zip(xs, v, hi, strict=True):
            ax.text(
                xi,
                vi + hii + 1.2,
                f"{vi:.0f}" if vi >= 10 else f"{vi:.1f}",
                ha="center",
                va="bottom",
                fontsize=11,
                color=TEXT,
            )
        ymax = max(ymax, float((v + hi).max()))
    ax.set_xticks(x, [lab for _, lab in cats], fontsize=13)
    ax.set_ylim(0, min(100, ymax * 1.15 + 4))
    style(ax, ylabel)
    ax.set_title(title, fontsize=17, color=TEXT, loc="left", pad=44)
    ax.legend(fontsize=13, frameon=False, ncol=k, loc="lower left", bbox_to_anchor=(0, 1.01))
    fig.text(
        0.01,
        0.005,
        "Claim-map judge v2.7 (plan 0002 rev 17), pre-gate · 1,123 cells per lens"
             " · "
        "whiskers: item-bootstrap 95% CI",
        fontsize=10,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def stacked(data: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.2), dpi=160)
    y = np.arange(len(ARMS))[::-1]
    left = np.zeros(len(ARMS))
    for (key, label), color in zip(ALL_SHARES, STACK_COLORS, strict=True):
        v = np.array([share(o, key)[0] for o in data])
        ax.barh(
            y, v, left=left, color=color, label=label, edgecolor="white", linewidth=2, height=0.62
        )
        for yi, li, vi in zip(y, left, v, strict=True):
            if vi >= 3.5:
                ax.text(
                    li + vi / 2,
                    yi,
                    f"{vi:.0f}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="white" if color in ("#e34948", "#4a3aa7", "#2a78d6") else TEXT,
                )
        left += v
    ax.set_yticks(y, [lab for _, lab in ARMS], fontsize=14)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of readout text (%)", fontsize=15, color=TEXT)
    ax.tick_params(axis="both", labelsize=13, colors=MUTED, length=0)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_title(
        "Where each lens's readout text goes (all categories)",
        fontsize=17,
        color=TEXT,
        loc="left",
        pad=12,
    )
    ax.legend(fontsize=12, frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.16))
    fig.text(
        0.01,
        0.005,
        "Character-weighted, each readout weighted equally · claim-map judge"
             " v2.7, "
        "pre-gate · residual failed share ≤ 0.1%",
        fontsize=10,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "plots" / "claim_map_pilot"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    data = [load(a) for a, _ in ARMS]
    grouped(
        data,
        HALLU,
        rate,
        "Readouts with ≥ 1 such segment (%)",
        "Hallucination rate per lens (readout level)",
        out / "rates_hallucination.png",
    )
    grouped(
        data,
        ALL_RATES,
        rate,
        "Readouts with ≥ 1 such segment (%)",
        "All categories per lens (readout level)",
        out / "rates_all.png",
    )
    grouped(
        data,
        HALLU,
        lambda o, k: share(o, "verdict:hallucination" if k == "hallucination" else f"leaf:{k}"),
        "Share of readout text (%)",
        "Hallucinated share of readout text per lens",
        out / "share_hallucination.png",
    )
    stacked(data, out / "share_all.png")
    print("wrote", *sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
