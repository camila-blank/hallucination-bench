"""Country ladder on the SHARED set (Camila 2026-10-13): the (item, sibling country) pairs that
s3d O-lens, ML-NLA RL600 and NLA-RL L42 all commit to and the probe can test at the mention's
own layer. On this set cross-method support is automatic (all three methods name the pair), so
the verdict shown is the probe's alone: fires (faithful) vs does not fire.
Reads outputs/sibling/um/label/units_labeled.jsonl.
"""

import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hallucination_bench import REPO_ROOT
from hallucination_bench.evals.sibling_probe.um_ladder import bootstrap_rate

LAB = REPO_ROOT / "outputs" / "sibling" / "um" / "label" / "units_labeled.jsonl"
OUT = REPO_ROOT / "plots" / "sibling_probe" / "um_country_shared.png"
METHODS = [("s3d", "s3d O-lens"), ("mlnla-rl600", "ML-NLA RL600"), ("nla-rl-L42", "NLA-RL L42")]


def ok(r: dict) -> bool:
    return r["committed"] and r["probe"]["na_reason"] is None


def main() -> None:
    rows = [json.loads(x) for x in LAB.read_text().splitlines() if x.strip()]
    sets = [
        {(r["item"], r["concept"]) for r in rows if r["method"] == m and ok(r)} for m, _ in METHODS
    ]
    shared = set.intersection(*sets)
    by = defaultdict(list)
    for r in rows:
        if ok(r) and (r["item"], r["concept"]) in shared:
            by[r["method"]].append(r)
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
        rs = by[m]
        n_fire = sum(r["probe"]["fire_L"] for r in rs)
        f, nf = n_fire / len(rs), 1 - n_fire / len(rs)
        ax.bar(
            x,
            f,
            color="#2a78d6",
            edgecolor="#fcfcfb",
            lw=2,
            width=0.62,
            label="probe fires (faithful)" if x == 0 else None,
        )
        ax.bar(
            x,
            nf,
            bottom=f,
            color="#eb6834",
            edgecolor="#fcfcfb",
            lw=2,
            width=0.62,
            label="probe does not fire" if x == 0 else None,
        )
        ax.text(
            x, f / 2, f"{f:.0%}\n({n_fire})", ha="center", va="center", color="#fff", fontsize=13
        )
        ax.text(
            x,
            f + nf / 2,
            f"{nf:.0%}\n({len(rs) - n_fire})",
            ha="center",
            va="center",
            color="#fff",
            fontsize=13,
        )
        h = bootstrap_rate(rs, lambda r: float(not r["probe"]["fire_L"]), lambda r: 1.0, "item")
        # on the boundary between the segments (at 1 - rate): the CI of the orange share
        ax.errorbar(
            x + 0.36,
            f,
            yerr=[[h["hi"] - h["rate"]], [h["rate"] - h["lo"]]],
            fmt="none",
            ecolor="#0b0b0b",
            capsize=5,
            lw=1.5,
        )
        ax.text(x, 1.02, f"n = {len(rs)} mentions", ha="center", fontsize=13, color="#52514e")
        print(m, len(rs), f"not-fire {h['rate']:.2f} [{h['lo']:.2f}, {h['hi']:.2f}]")
    ax.set_xticks(range(len(METHODS)), [n for _, n in METHODS])
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("share of committed, probe-testable mentions")
    ax.set_title(
        f"Shared set: {len(shared)} (item, sibling) pairs all three lenses commit to",
        loc="left",
        fontsize=17,
        pad=24,
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=13)
    fig.text(
        0.01,
        0.01,
        "Probe fires = the sibling's probe beats the item's 8 distractors at the "
        "mention's own layer. Cross-method is not shown: on this set every pair has it. "
        "Error bar: 95% CI on 'does not fire' (bootstrap by item, 11 items).",
        fontsize=10,
        color="#52514e",
        wrap=True,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT, dpi=160)
    print(OUT, sorted(shared))


if __name__ == "__main__":
    main()
