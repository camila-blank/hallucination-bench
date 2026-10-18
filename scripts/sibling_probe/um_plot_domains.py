"""Combined simple plot across domains (plan 0011): the um_country_simple.png panel for each
domain whose judge met the 85% gold target (emotion is excluded: 0.71). Same rules as
um_plot_simple.py: committed mentions the probe can test at their own layer, no nearby tier.
"""

import json
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hallucination_bench import REPO_ROOT
from hallucination_bench.evals.sibling_probe.um_ladder import bootstrap_rate

UM = REPO_ROOT / "outputs" / "sibling" / "um"
OUT = REPO_ROOT / "plots" / "sibling_probe" / "um_domains_simple.png"
DOMAINS = [
    ("country", "Country"),
    ("occupation", "Occupation"),
    ("capital_city", "City (visiting)"),
    ("holiday", "Holiday"),
]
METHODS = [("s3d", "s3d\nO-lens"), ("mlnla-rl600", "ML-NLA\nRL600"), ("nla-rl-L42", "NLA-RL\nL42")]
CATS = [
    ("faithful", "faithful (probe fires)", "#2a78d6"),
    ("cross", "cross-method allowed", "#1baf7a"),
    ("hall", "hallucination", "#eb6834"),
]


def label(r: dict) -> str:
    if r["probe"]["fire_L"]:
        return "faithful"
    return "cross" if r["cross"] else "hall"


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "axes.edgecolor": "#52514e",
            "xtick.color": "#52514e",
            "ytick.color": "#52514e",
        }
    )
    avail = [
        (d, t)
        for d, t in DOMAINS
        if (UM / ("" if d == "country" else d) / "label" / "units_labeled.jsonl").exists()
    ]
    fig, axes = plt.subplots(1, len(avail), figsize=(5.2 * len(avail), 6.6), sharey=True)
    axes = axes if len(avail) > 1 else [axes]
    table = []
    for ax, (d, title) in zip(axes, avail, strict=True):
        p = UM / ("" if d == "country" else d) / "label" / "units_labeled.jsonl"
        rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        for x, (m, _name) in enumerate(METHODS):
            rs = [
                r
                for r in rows
                if r["method"] == m and r["committed"] and r["probe"]["na_reason"] is None
            ]
            if not rs:
                continue
            c = Counter(label(r) for r in rs)
            bottom = 0.0
            for key, lab, col in CATS:
                v = c[key] / len(rs)
                ax.bar(
                    x,
                    v,
                    bottom=bottom,
                    color=col,
                    edgecolor="#fcfcfb",
                    lw=2,
                    width=0.66,
                    label=lab if (x == 0 and ax is axes[0]) else None,
                )
                if v >= 0.07:
                    ax.text(
                        x,
                        bottom + v / 2,
                        f"{v:.0%}",
                        ha="center",
                        va="center",
                        color="#ffffff",
                        fontsize=12,
                    )
                bottom += v
            h = bootstrap_rate(rs, lambda r: float(label(r) == "hall"), lambda r: 1.0, "item")
            top = 1 - h["rate"]
            ax.errorbar(
                x + 0.38,
                top,
                yerr=[[h["hi"] - h["rate"]], [h["rate"] - h["lo"]]],
                fmt="none",
                ecolor="#0b0b0b",
                capsize=4,
                lw=1.3,
            )
            ax.text(x, 1.02, f"n={len(rs)}", ha="center", fontsize=11, color="#52514e")
            table.append((d, m, len(rs), c["faithful"], c["cross"], c["hall"], h))
        ax.set_xticks(range(len(METHODS)), [n for _, n in METHODS], fontsize=12)
        ax.set_title(title, loc="left", fontsize=16, pad=20)
        ax.set_ylim(0, 1.08)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].set_ylabel("share of committed sibling mentions\nthe probe could test")
    fig.legend(loc="lower center", ncol=3, frameon=False, fontsize=13)
    fig.suptitle(
        "Sibling mentions across user-modelling domains: faithful vs hallucinated",
        x=0.01,
        ha="left",
        fontsize=18,
    )
    fig.text(
        0.01,
        0.005,
        "Probe-n/a mentions excluded; faithful = probe fires at the mention's "
        "own layer; error bar = 95% CI on the hallucination share (bootstrap by item). "
        "J-lens omitted (too few testable units). Emotion omitted (judge below the 85% "
        "target).",
        fontsize=10,
        color="#52514e",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(OUT, dpi=160)
    print(OUT)
    for d, m, n, f, cr, _hl, h in table:
        print(
            f"{d:13s} {m:12s} n={n:4d} faithful {f / n:.0%} cross {cr / n:.0%} "
            f"hall {h['rate']:.2f} [{h['lo']:.2f}, {h['hi']:.2f}]"
        )


if __name__ == "__main__":
    main()
