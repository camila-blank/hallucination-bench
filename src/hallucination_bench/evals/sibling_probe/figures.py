"""Plan 0004 figures (``plots/sibling_probe/``), slide quality, 150 dpi, legends outside.

* Figure 1, probe agreement per method: the share of claim units whose (gate-passing) probe
  detects the concept, item-clustered 95% CI, the probe-n/a share printed beside each bar; a
  second panel by layer (one line per method). Sets A and B in separate rows.
* Figure 2, categories: stacked claim units per category per method (n on every bar), and a
  second panel by layer depth (early / mid / late). Sets A and B in separate rows.

Colours: the dataviz reference categorical palette, fixed order (methods: blue, orange, aqua,
yellow; categories: aqua, blue, yellow, red, grey, violet, magenta), validated with the skill's
validator (adjacent pairs pass; contrast relief = the n labels and the summary tables).
"""

from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

METHOD_ORDER = ("s3d", "nla-rl-L42", "jlens", "mlnla-rl600")
METHOD_LABEL = {
    "s3d": "s3d O-lens",
    "nla-rl-L42": "NLA-RL L42",
    "jlens": "J-lens",
    "mlnla-rl600": "ML-NLA RL600 (side)",
}
METHOD_COLOR = {
    "s3d": "#2a78d6",
    "nla-rl-L42": "#eb6834",
    "jlens": "#1baf7a",
    "mlnla-rl600": "#eda100",
}
CAT_ORDER = (
    "validated",
    "consensus",
    "option",
    "hallucination",
    "unclear",
    "probe_na",
    "entity_substitution",
)
CAT_LABEL = {
    "validated": "validated (probe)",
    "consensus": "consensus (post hoc)",
    "option": "option",
    "hallucination": "hallucination",
    "unclear": "unclear",
    "probe_na": "probe n/a",
    "entity_substitution": "entity substitution",
}
CAT_COLOR = {
    "validated": "#1baf7a",
    "consensus": "#2a78d6",
    "option": "#eda100",
    "hallucination": "#e34948",
    "unclear": "#8b8a85",
    "probe_na": "#4a3aa7",
    "entity_substitution": "#e87ba4",
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LAYERS = (20, 24, 28, 32, 36, 40, 42, 44, 48, 52, 56, 60)


def _style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 17,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "axes.edgecolor": INK2,
            "axes.labelcolor": INK,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
        }
    )


def item_boot(items: Sequence[str], x: Sequence[float], n_boot: int = 1000, seed: int = 0):
    if not len(x):
        return float("nan"), float("nan"), float("nan")
    ids = sorted(set(items))
    ix = {k: i for i, k in enumerate(ids)}
    g = np.array([ix[i] for i in items])
    xs = np.asarray(x, float)
    s = np.bincount(g, weights=xs, minlength=len(ids))
    c = np.bincount(g, minlength=len(ids)).astype(float)
    W = np.random.default_rng(seed).multinomial(len(ids), np.full(len(ids), 1 / len(ids)), n_boot)
    b = (W @ s) / np.maximum(W @ c, 1e-12)
    return float(xs.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def fig1(rows: Sequence[dict], out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    sets = [s for s in ("A", "B") if any(r["set"] == s for r in rows)]
    fig, axes = plt.subplots(
        len(sets), 2, figsize=(17, 5.2 * len(sets)), squeeze=False, width_ratios=[1, 1.6]
    )
    for si, st in enumerate(sets):
        rs = [r for r in rows if r["set"] == st]
        ax = axes[si][0]
        ms = [m for m in METHOD_ORDER if any(r["method"] == m for r in rs)]
        for i, m in enumerate(ms):
            mr = [r for r in rs if r["method"] == m]
            ok = [r for r in mr if r["probe"]["status"] == "ok"]
            est, lo, hi = item_boot(
                [r["item"] for r in ok], [float(r["probe"]["detected"]) for r in ok]
            )
            na = 1 - len(ok) / len(mr) if mr else float("nan")
            if ok:
                ax.bar(i, est, width=0.6, color=METHOD_COLOR[m], edgecolor="#fcfcfb", linewidth=2)
                ax.errorbar(i, est, yerr=[[est - lo], [hi - est]], color=INK, capsize=5, lw=1.5)
            ax.text(
                i,
                (hi if ok else 0) + 0.03,
                f"n={len(ok)}\nn/a {na:.0%}",
                ha="center",
                va="bottom",
                fontsize=12,
                color=INK2,
            )
        ax.set_xticks(range(len(ms)), [METHOD_LABEL[m].replace(" (side)", "\n(side)") for m in ms])
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("probe detects the concept")
        ax.set_title(f"Set {st}: probe agreement per method")
        ax.yaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
        ax = axes[si][1]
        for m in ms:
            xs, ys, los, his = [], [], [], []
            for L in LAYERS:
                ok = [
                    r
                    for r in rs
                    if r["method"] == m and r["layer"] == L and r["probe"]["status"] == "ok"
                ]
                if len(ok) < 5:
                    continue
                est, lo, hi = item_boot(
                    [r["item"] for r in ok], [float(r["probe"]["detected"]) for r in ok]
                )
                xs.append(L)
                ys.append(est)
                los.append(lo)
                his.append(hi)
            if xs:
                ax.plot(xs, ys, "-o", color=METHOD_COLOR[m], lw=2, ms=8, label=METHOD_LABEL[m])
                ax.fill_between(xs, los, his, color=METHOD_COLOR[m], alpha=0.12, lw=0)
        ax.set_xticks(LAYERS)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("lens layer (cells with >= 5 units)")
        ax.set_title(f"Set {st}: by layer (item-clustered 95% CI)")
        ax.yaxis.grid(True, color=GRID)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout()
    p = out / "fig1_probe_agreement.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def _stack(ax, groups: list[tuple[str, Counter]], title: str, share: bool = True) -> None:
    for i, (_lab, c) in enumerate(groups):
        tot = sum(c.values())
        bottom = 0.0
        for cat in CAT_ORDER:
            v = c.get(cat, 0) / tot if (tot and share) else c.get(cat, 0)
            if v:
                ax.bar(
                    i,
                    v,
                    bottom=bottom,
                    width=0.62,
                    color=CAT_COLOR[cat],
                    edgecolor="#fcfcfb",
                    linewidth=2,
                )
                if share and v >= 0.06:
                    ax.text(
                        i,
                        bottom + v / 2,
                        f"{v:.0%}",
                        ha="center",
                        va="center",
                        fontsize=11,
                        color="#ffffff"
                        if cat in ("consensus", "hallucination", "probe_na")
                        else INK,
                    )
            bottom += v
        ax.text(i, bottom + 0.01, f"n={tot}", ha="center", va="bottom", fontsize=12, color=INK2)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    ax.set_ylim(0, 1.1 if share else None)
    ax.set_ylabel("share of claim units")
    ax.set_title(title)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)


def fig2(rows: Sequence[dict], out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    _style(plt)
    sets = [s for s in ("A", "B") if any(r["set"] == s for r in rows)]
    fig, axes = plt.subplots(
        len(sets), 2, figsize=(17, 5.6 * len(sets)), squeeze=False, width_ratios=[1, 1.6]
    )
    for si, st in enumerate(sets):
        rs = [r for r in rows if r["set"] == st]
        ms = [m for m in METHOD_ORDER if any(r["method"] == m for r in rs)]
        groups = [
            (
                METHOD_LABEL[m].replace(" (side)", "\n(side)"),
                Counter(r["category"] for r in rs if r["method"] == m),
            )
            for m in ms
        ]
        _stack(axes[si][0], groups, f"Set {st}: categories per method")
        by = defaultdict(Counter)
        for r in rs:
            by[(r["method"], r["depth"])][r["category"]] += 1
        g2 = [
            (f"{METHOD_LABEL[m].split(' (')[0].replace(' ', chr(10), 1)}\n{d}", by[(m, d)])
            for m in ms
            for d in ("early", "mid", "late")
            if by[(m, d)]
        ]
        _stack(axes[si][1], g2, f"Set {st}: by layer depth (early <= 36, mid 40-42, late >= 44)")
        axes[si][1].tick_params(axis="x", labelsize=10)
    handles = [Patch(color=CAT_COLOR[c], label=CAT_LABEL[c]) for c in CAT_ORDER]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.tight_layout()
    p = out / "fig2_categories.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def make_all(rows: Sequence[dict], out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    ps = [fig1(rows, out), fig2(rows, out)]
    for p in ps:
        print(f"[figures] {p}")
    return ps
