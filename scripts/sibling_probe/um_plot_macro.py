"""Macro average across user-modelling domains (plan 0011): the unweighted mean over domains of
each label's share among committed, probe-testable sibling mentions (no nearby tier), over the
three final domains (country, city visiting, holiday; plan 0013). Error bar: 95% CI on the
macro hallucination share, items resampled within each domain (1,000 draws)."""

import json
import random
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hallucination_bench import REPO_ROOT

UM = REPO_ROOT / "outputs" / "sibling" / "um"
OUT = REPO_ROOT / "plots" / "sibling_probe" / "um_macro_simple.png"
DOMAINS = ["country", "capital_city", "holiday"]
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


def load() -> dict[tuple[str, str], list[dict]]:
    out = {}
    for d in DOMAINS:
        p = UM / ("" if d == "country" else d) / "label" / "units_labeled.jsonl"
        rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        for m, _ in METHODS:
            out[(d, m)] = [
                r
                for r in rows
                if r["method"] == m and r["committed"] and r["probe"]["na_reason"] is None
            ]
    return out


def macro_ci(units: dict, m: str, doms: list[str], n: int = 1000) -> tuple[float, float]:
    rng = random.Random(0)
    by = {d: defaultdict(list) for d in doms}
    for d in doms:
        for r in units[(d, m)]:
            by[d][r["item"]].append(label(r) == "hall")
    sims = []
    for _ in range(n):
        vals = []
        for d in doms:
            keys = list(by[d])
            draw = [v for k in (rng.choice(keys) for _ in keys) for v in by[d][k]]
            vals.append(sum(draw) / len(draw))
        sims.append(sum(vals) / len(vals))
    sims.sort()
    return sims[int(0.025 * n)], sims[int(0.975 * n) - 1]


def main() -> None:
    units = load()
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
    panels = [("Country, city (visiting), holiday", DOMAINS)]
    fig, ax0 = plt.subplots(figsize=(8, 6.6))
    axes = [ax0]
    for ax, (title, doms) in zip(axes, panels, strict=True):
        for x, (m, _n) in enumerate(METHODS):
            shares = {}
            for key, _, _ in CATS:
                shares[key] = sum(
                    Counter(label(r) for r in units[(d, m)])[key] / len(units[(d, m)]) for d in doms
                ) / len(doms)
            bottom = 0.0
            for key, lab, col in CATS:
                v = shares[key]
                ax.bar(
                    x,
                    v,
                    bottom=bottom,
                    color=col,
                    edgecolor="#fcfcfb",
                    lw=2,
                    width=0.62,
                    label=lab if (x == 0 and ax is axes[0]) else None,
                )
                ax.text(
                    x,
                    bottom + v / 2,
                    f"{v:.0%}",
                    ha="center",
                    va="center",
                    color="#ffffff",
                    fontsize=13,
                )
                bottom += v
            lo, hi = macro_ci(units, m, doms)
            h = shares["hall"]
            ax.errorbar(
                x + 0.36,
                1 - h,
                yerr=[[hi - h], [h - lo]],
                fmt="none",
                ecolor="#0b0b0b",
                capsize=5,
                lw=1.5,
            )
            n = sum(len(units[(d, m)]) for d in doms)
            ax.text(x, 1.02, f"n={n}", ha="center", fontsize=12, color="#52514e")
            print(
                f"{title:20s} {m:12s} faithful {shares['faithful']:.2f} cross "
                f"{shares['cross']:.2f} hall {h:.2f} [{lo:.2f}, {hi:.2f}]"
            )
        ax.set_xticks(range(len(METHODS)), [n for _, n in METHODS])
        ax.set_title(f"{title} ({len(doms)})", loc="left", fontsize=16, pad=20)
        ax.set_ylim(0, 1.08)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].set_ylabel("macro share of committed sibling mentions\nthe probe could test")
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.06), ncol=3, frameon=False, fontsize=13)
    fig.suptitle("Macro average over user-modelling domains", x=0.01, ha="left", fontsize=18)
    fig.text(
        0.01,
        0.005,
        "Domains: country, city (visiting), holiday; each weighted "
        "equally. Probe-n/a excluded; faithful = probe fires at the mention's own layer. "
        "Error bar: 95% CI on the macro hallucination share (items resampled within domain).",
        fontsize=10,
        color="#52514e",
        wrap=True,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    fig.savefig(OUT, dpi=160)
    print(OUT)


if __name__ == "__main__":
    main()
