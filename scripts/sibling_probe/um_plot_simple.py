"""Simplified summary bar plot (Camila 2026-10-13): committed sibling mentions the probe can
test at their own layer only (probe-n/a units excluded), no nearby-layer tier.

faithful = the sibling's probe beats the item's distractors at the mention's own layer;
otherwise cross-method allowed if another method entertains the country for the item within
+-4 layers; otherwise hallucination. Reads outputs/sibling/um/label/units_labeled.jsonl.
"""

import argparse
import json
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hallucination_bench import REPO_ROOT
from hallucination_bench.evals.sibling_probe.um_ladder import bootstrap_rate

LAB = REPO_ROOT / "outputs" / "sibling" / "um" / "label" / "units_labeled.jsonl"
OUT = REPO_ROOT / "plots" / "sibling_probe" / "um_country_simple.png"
METHODS = [("s3d", "s3d O-lens"), ("mlnla-rl600", "ML-NLA RL600"), ("nla-rl-L42", "NLA-RL L42")]
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="country")
    cat = ap.parse_args().category
    lab_path = (
        LAB
        if cat == "country"
        else (REPO_ROOT / "outputs" / "sibling" / "um" / cat / "label" / "units_labeled.jsonl")
    )
    out = REPO_ROOT / "plots" / "sibling_probe" / f"um_{cat}_simple.png"
    rows = [json.loads(x) for x in lab_path.read_text().splitlines() if x.strip()]
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "axes.edgecolor": "#52514e",
            "xtick.color": "#52514e",
            "ytick.color": "#52514e",
        }
    )
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for x, (m, _name) in enumerate(METHODS):
        rs = [
            r
            for r in rows
            if r["method"] == m and r["committed"] and r["probe"]["na_reason"] is None
        ]
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
                width=0.62,
                label=lab if x == 0 else None,
            )
            ax.text(
                x,
                bottom + v / 2,
                f"{v:.0%}\n({c[key]})",
                ha="center",
                va="center",
                color="#ffffff",
                fontsize=13,
            )
            bottom += v
        h = bootstrap_rate(rs, lambda r: float(label(r) == "hall"), lambda r: 1.0, "item")
        top = 1 - c["hall"] / len(rs)
        # on the boundary below the hallucination segment (at 1 - rate)
        ax.errorbar(
            x + 0.36,
            top,
            yerr=[[h["hi"] - h["rate"]], [h["rate"] - h["lo"]]],
            fmt="none",
            ecolor="#0b0b0b",
            capsize=5,
            lw=1.5,
        )
        ax.text(x, 1.02, f"n = {len(rs)}", ha="center", fontsize=13, color="#52514e")
    ax.set_xticks(range(len(METHODS)), [n for _, n in METHODS])
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("share of committed sibling mentions\nthe probe could test")
    title = {
        "country": "Country",
        "occupation": "Occupation",
        "emotion": "Emotion",
        "capital_city": "City (visiting)",
        "holiday": "Holiday",
    }.get(cat, cat)
    ax.set_title(f"{title} siblings: faithful vs hallucinated", loc="left", fontsize=18, pad=24)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=13)
    fig.text(
        0.01,
        0.01,
        "Probe-n/a mentions excluded; faithful = probe fires at the mention's own "
        "layer. Error bar: 95% CI on the hallucination share (bootstrap by item). "
        "J-lens omitted (1 testable unit).",
        fontsize=10,
        color="#52514e",
        wrap=True,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=160)
    print(out)
    for m, _ in METHODS:
        rs = [
            r
            for r in rows
            if r["method"] == m and r["committed"] and r["probe"]["na_reason"] is None
        ]
        h = bootstrap_rate(rs, lambda r: float(label(r) == "hall"), lambda r: 1.0, "item")
        print(m, len(rs), f"{h['rate']:.2f} [{h['lo']:.2f}, {h['hi']:.2f}]")


if __name__ == "__main__":
    main()
