"""Plots for results 0008 (agentic_misalignment): verdict shares, kind tiers, per-layer view.

Reads outputs/agentic_misalignment/<model>/report.json (+ audit/tiers.json) and writes PNGs to
plots/am_claim_map/. Palette: the dataviz reference categorical slots, ordered so adjacent
stacked segments pass the CVD and normal-vision checks (scripts/validate_palette.js); segments
carry direct value labels because three slots are below 3:1 contrast on the surface.

    uv run --no-sync python scripts/plot_am.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path("outputs/agentic_misalignment/claude-opus-5-5")
OUT = Path("plots/am_claim_map")
ARMS = [("s3d-rl600", "O-lens RL600"), ("nla-rl-L42", "NLA-RL L42"), ("jlens", "J-lens")]
VERDICTS = [  # (row, label, colour): stacking order = validated adjacency order
    ("correct", "correct", "#2a78d6"),
    ("unresolved", "unverifiable", "#eda100"),
    ("thinky", "model_state", "#1baf7a"),
    ("hallucination", "hallucination", "#eb6834"),
    ("probe_candidate", "probe candidate", "#4a3aa7"),
    ("junk", "junk", "#e87ba4"),
]
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

plt.rcParams.update(
    {
        "font.size": 13,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)


def _stacked(ax, rows, labels, segs, title):
    for i, shares in enumerate(rows):
        left = 0.0
        for (_, _, col), v in zip(segs, shares, strict=True):
            ax.barh(i, v, left=left, color=col, edgecolor=SURFACE, linewidth=2, height=0.62)
            if v >= 0.05:
                ax.text(left + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=11,
                        color=INK)
            left += v
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(title, loc="left", fontsize=15, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def verdict_plot(rep):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True)
    for ax, pop, title in (
        (axes[0], "headline", "Headline: the model (or an unnamed actor) considering or doing it"),
        (axes[1], "headline_ai_self", "Strict headline: the AI itself"),
    ):
        rows, labels = [], []
        for arm, name in ARMS:
            p = rep["arms"][arm]["overall"]["populations"][pop]["verdict"]
            n = p["n"]
            rows.append([p["rows"][v]["count"] / n for v, _, _ in VERDICTS])
            labels.append(f"{name}\n(n={n:,})")
        _stacked(ax, rows, labels, VERDICTS, title)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in VERDICTS]
    fig.legend(handles, [lab for _, lab, _ in VERDICTS], loc="lower center", ncol=6,
               frameon=False, fontsize=12)
    fig.suptitle("Claim-map v2.7 verdicts on lens misalignment mentions (failed calls excluded "
                 "from the bar; shares of judged spans)", x=0.01, ha="left", fontsize=13,
                 color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(OUT / "verdict_shares.png", dpi=160)
    plt.close(fig)


TIERS = [
    ("realised", "realised (the rollout does it)", "#2a78d6"),
    ("designed_or_setup", "designed / set up by the scenario", "#eda100"),
    ("unrelated", "unrelated misconduct", "#eb6834"),
]


def tier_plot(rep):
    rows, labels = [], []
    for arm, name in ARMS:
        km = rep["arms"][arm]["overall"]["kind_match"]
        c = km["tier_counts"]
        tot = sum(c.get(t, 0) for t in ("realised", "designed_only", "scenario_set_up",
                                         "unrelated"))
        rows.append([c.get("realised", 0) / tot,
                     (c.get("designed_only", 0) + c.get("scenario_set_up", 0)) / tot,
                     c.get("unrelated", 0) / tot])
        corr = (rep.get("tiers") or {}).get(arm, {}).get("corrected_unrelated_rate")
        labels.append(f"{name}\n(audit-corrected unrelated {corr:.0%})" if corr is not None
                      else name)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    _stacked(ax, rows, labels, TIERS, "What misconduct do headline mentions name? (raw tiers)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in TIERS]
    fig.legend(handles, [lab for _, lab, _ in TIERS], loc="lower center", ncol=3,
               frameon=False, fontsize=12)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "kind_tiers.png", dpi=160)
    plt.close(fig)


def layer_plot(rep):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, (row, title) in zip(axes, (("correct", "correct"), ("hallucination",
                                                                "hallucination")), strict=True):
        for (arm, name), col in zip(ARMS[::2], ("#2a78d6", "#eb6834"), strict=True):
            bl = rep["arms"][arm]["by_layer"]
            xs = sorted(int(k) for k in bl)
            ys = [bl[str(x)]["headline"]["verdict"]["rows"][row]["pooled"] for x in xs]
            ax.plot(xs, ys, color=col, linewidth=2, marker="o", markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=name)
            ax.annotate(name, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=11, color=INK)
        ax.set_title(f"Headline {title} share by layer", loc="left", fontsize=14, color=INK)
        ax.set_xlabel("layer")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(axis="y", color="#e6e5df", linewidth=1)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "per_layer.png", dpi=160)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rep = json.loads((ROOT / "report.json").read_text())
    verdict_plot(rep)
    tier_plot(rep)
    layer_plot(rep)
    print(f"wrote {sorted(p.name for p in OUT.glob('*.png'))}")


if __name__ == "__main__":
    main()
