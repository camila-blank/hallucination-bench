"""Macro plots with a matched-layer comparison (Camila 2026-10-13). NLA-RL reads ONE layer (L42);
the s3d O-lens and ML-NLA read 11 (L20..60), so the all-layer bars are not like for like.

* um_macro_single_layer.png: s3d / ML-NLA at ONE layer (default L44, the grid layer nearest
  NLA-RL's L42; there is no L42 readout for them) vs NLA-RL at L42;
* um_macro_all_layers.png: s3d / ML-NLA pooled over all 11 layers (NLA-RL excluded);
* um_macro_by_layer.png: the macro hallucination share at each layer for s3d / ML-NLA, with
  NLA-RL as one point at L42.

Macro = the unweighted mean over country, occupation, city (visiting) and holiday of each label's
share among committed, probe-testable sibling mentions (no nearby tier). CIs: items resampled
within each domain.
"""

import argparse
import json
import random
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hallucination_bench import REPO_ROOT

UM = REPO_ROOT / "outputs" / "sibling" / "um"
PLOTS = REPO_ROOT / "plots" / "sibling_probe"
DOMAINS = ["country", "occupation", "capital_city", "holiday"]
METHODS = [("s3d", "s3d O-lens"), ("mlnla-rl600", "ML-NLA RL600"), ("nla-rl-L42", "NLA-RL")]
GRID = (20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60)
CATS = [
    ("faithful", "faithful (probe fires)", "#2a78d6"),
    ("cross", "cross-method allowed", "#1baf7a"),
    ("hall", "hallucination", "#eb6834"),
]
FOOT = (
    "Macro over country, occupation, city (visiting), holiday (equal weight). Probe-n/a "
    "excluded; faithful = probe fires at the mention's own layer. NLA-RL reads a single layer "
    "(L42). Error bar: 95% CI on the macro hallucination share (items resampled within domain)."
)


def label(r: dict) -> str:
    if r["probe"]["fire_L"]:
        return "faithful"
    return "cross" if r["cross"] else "hall"


_ROWS: dict[str, list[dict]] = {}


def rows_of(d: str) -> list[dict]:
    if d not in _ROWS:
        p = UM / ("" if d == "country" else d) / "label" / "units_labeled.jsonl"
        _ROWS[d] = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    return _ROWS[d]


def units(m: str, layers: tuple[int, ...]) -> dict[str, list[dict]]:
    """Committed, probe-testable units per domain; s3d / ML-NLA restricted to ``layers``."""
    return {
        d: [
            r
            for r in rows_of(d)
            if r["method"] == m
            and r["committed"]
            and r["probe"]["na_reason"] is None
            and (m == "nla-rl-L42" or r["eff_layer"] in layers)
        ]
        for d in DOMAINS
    }


def macro(u: dict[str, list[dict]], key: str) -> float:
    return sum(Counter(label(r) for r in u[d])[key] / len(u[d]) for d in DOMAINS) / len(DOMAINS)


def macro_ci(u: dict[str, list[dict]], n: int = 1000) -> tuple[float, float]:
    rng = random.Random(0)
    by = {d: defaultdict(list) for d in DOMAINS}
    for d in DOMAINS:
        for r in u[d]:
            by[d][r["item"]].append(label(r) == "hall")
    sims = []
    for _ in range(n):
        vals = []
        for d in DOMAINS:
            keys = list(by[d])
            draw = [v for k in (rng.choice(keys) for _ in keys) for v in by[d][k]]
            vals.append(sum(draw) / len(draw))
        sims.append(sum(vals) / len(vals))
    sims.sort()
    return sims[int(0.025 * n)], sims[int(0.975 * n) - 1]


def tick(m: str, name: str, layers: tuple[int, ...]) -> str:
    if m == "nla-rl-L42":
        return f"{name}\nsingle layer\n(L42)"
    if len(layers) == 1:
        return f"{name}\nsingle layer\n(L{layers[0]})"
    return f"{name}\nall {len(layers)} layers\n(L{layers[0]}-{layers[-1]})"


def bars(layers: tuple[int, ...], out_name: str, title: str, methods: list | None = None) -> None:
    methods = methods or METHODS
    fig, ax = plt.subplots(figsize=(10, 6.9))
    for x, (m, _name) in enumerate(methods):
        u = units(m, layers)
        shares = {key: macro(u, key) for key, _, _ in CATS}
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
                label=lab if x == 0 else None,
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
        lo, hi = macro_ci(u)
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
        n = sum(len(v) for v in u.values())
        ax.text(x, 1.02, f"n={n}", ha="center", fontsize=12, color="#52514e")
        print(
            f"{out_name:28s} {m:12s} faithful {shares['faithful']:.2f} cross "
            f"{shares['cross']:.2f} hall {h:.2f} [{lo:.2f}, {hi:.2f}] n={n}"
        )
    ax.set_xticks(range(len(methods)), [tick(m, nm, layers) for m, nm in methods], fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("macro share of committed sibling mentions\nthe probe could test")
    ax.set_title(title, loc="left", fontsize=15, pad=22)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=12)
    foot = (
        FOOT
        if any(m == "nla-rl-L42" for m, _ in methods)
        else FOOT.replace(" NLA-RL reads a single layer (L42).", "")
    )
    fig.text(0.01, 0.01, foot, fontsize=9, color="#52514e", wrap=True)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(PLOTS / out_name, dpi=160)
    plt.close(fig)


def per_layer() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.3))
    cols = {"s3d": "#2a78d6", "mlnla-rl600": "#4a3aa7"}
    for m, name in METHODS[:2]:
        xs, ys, lo_e, hi_e = [], [], [], []
        for L in GRID:
            u = units(m, (L,))
            if any(not v for v in u.values()):
                continue
            h = macro(u, "hall")
            lo, hi = macro_ci(u, n=500)
            xs.append(L)
            ys.append(h)
            lo_e.append(h - lo)
            hi_e.append(hi - h)
        ax.errorbar(
            xs,
            ys,
            yerr=[lo_e, hi_e],
            color=cols[m],
            lw=2,
            marker="o",
            ms=8,
            mec="#fcfcfb",
            mew=2,
            capsize=3,
            label=name,
        )
        print("by layer", m, " ".join(f"L{x}:{y:.2f}" for x, y in zip(xs, ys, strict=True)))
    u = units("nla-rl-L42", GRID)
    h = macro(u, "hall")
    lo, hi = macro_ci(u, n=500)
    ax.errorbar(
        [42],
        [h],
        yerr=[[h - lo], [hi - h]],
        color="#eb6834",
        marker="D",
        ms=10,
        mec="#fcfcfb",
        mew=2,
        capsize=4,
        lw=2,
        ls="none",
        label="NLA-RL (single layer, L42)",
    )
    ax.set_xticks([*GRID, 42])
    ax.set_xlabel("layer")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("macro hallucination share")
    ax.grid(axis="y", color="#e6e5e0", lw=1)
    ax.set_title("Macro hallucination share by layer (4 domains)", loc="left", fontsize=16)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=12)
    fig.text(
        0.01,
        0.01,
        "Committed, probe-testable sibling mentions; each domain weighted "
        "equally; error bars: 95% CI (items resampled within domain).",
        fontsize=9,
        color="#52514e",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(PLOTS / "um_macro_by_layer.png", dpi=160)
    plt.close(fig)


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
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--match-layer",
        type=int,
        default=44,
        help="s3d / ML-NLA layer matched to NLA-RL's L42 (no L42 readout exists)",
    )
    a = ap.parse_args()
    bars(
        (a.match_layer,),
        "um_macro_single_layer.png",
        f"Single layer: s3d / ML-NLA at L{a.match_layer} vs NLA-RL at L42 (4 domains)",
    )
    bars(
        GRID,
        "um_macro_all_layers.png",
        "All 11 layers: s3d O-lens vs ML-NLA (4 domains)",
        methods=METHODS[:2],  # NLA-RL (single layer) excluded from the all-layer plot
    )
    per_layer()


if __name__ == "__main__":
    main()
