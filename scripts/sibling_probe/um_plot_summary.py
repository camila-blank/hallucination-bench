"""One-figure summary of the plan 0006 country ladder from outputs/sibling/um/label/ (no recompute).

Left: what committed sibling mentions turn out to be, per method. Right: hallucination rate
(committed adjudicable units) with item-bootstrap 95% CIs, and the same rate when nearby-layer
fires are not counted as faithful.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hallucination_bench import REPO_ROOT  # noqa: E402

LAB = REPO_ROOT / "outputs" / "sibling" / "um" / "label"
OUT = REPO_ROOT / "plots" / "sibling_probe" / "um_country_summary.png"
METHODS = [("s3d", "s3d O-lens"), ("mlnla-rl600", "ML-NLA RL600"), ("nla-rl-L42", "NLA-RL L42")]
SEGS = [
    ("probe-faithful (same layer)", "faithful: probe fires at L", "#2a78d6", None),
    ("probe-faithful (nearby layer)", "faithful: probe fires nearby", "#4a3aa7", None),
    ("cross-method allowed (adjudicable)", "cross-method allowed", "#1baf7a", None),
    ("cross-method allowed (probe n/a)", "cross-method allowed (probe n/a)", "#1baf7a", "///"),
    ("hallucination", "hallucination", "#eb6834", None),
    ("hallucination (probe n/a)", "hallucination (probe n/a)", "#eb6834", "///"),
]


def main() -> None:
    s = json.loads((LAB / "summary.json").read_text())
    plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                         "axes.edgecolor": "#52514e", "xtick.color": "#52514e",
                         "ytick.color": "#52514e"})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(17, 5.6), gridspec_kw={"width_ratios": [1.7, 1]})
    for y, (m, name) in enumerate(METHODS):
        counts = s["labels"][m]["counts"]
        n = s["labels"][m]["n_committed"]
        left = 0.0
        for key, lab, col, hatch in SEGS:
            v = counts.get(key, 0) / n
            a1.barh(y, v, left=left, color=col, hatch=hatch, edgecolor="#fcfcfb", lw=2,
                    height=0.62, label=lab if y == 0 else None)
            if v >= 0.07:
                a1.text(left + v / 2, y, f"{v:.0%}", ha="center", va="center", color="#ffffff",
                        fontsize=12)
            left += v
        share = n / s["labels"][m]["n_units"]
        a1.text(1.01, y, f"{n} committed\n({share:.0%} of {s['labels'][m]['n_units']} mentions)",
                va="center", fontsize=11, color="#52514e")
    a1.set_yticks(range(len(METHODS)), [n for _, n in METHODS])
    a1.invert_yaxis()
    a1.set_xlim(0, 1)
    a1.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    a1.set_xlabel("share of committed sibling-country mentions")
    a1.set_title("What committed sibling mentions turn out to be", loc="left", fontsize=16)
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False, fontsize=11)
    for y, (m, _name) in enumerate(METHODS):
        h = s["headline"][m]
        for key, off, face, lab in (("hallucination_rate", -0.12, "#eb6834", "nearby fires = faithful"),
                                    ("nearby_not_faithful", 0.12, "#fcfcfb", "nearby fires = hallucination")):
            r = h[key]
            a2.errorbar(r["rate"], y + off, xerr=[[r["rate"] - r["lo"]], [r["hi"] - r["rate"]]],
                        fmt="o", ms=10, mfc=face, mec="#eb6834", mew=2, color="#eb6834",
                        capsize=4, lw=2, label=lab if y == 0 else None)
            a2.text(r["hi"] + 0.02, y + off, f"{r['rate']:.2f}", va="center", fontsize=12)
    a2.set_yticks(range(len(METHODS)), [""] * len(METHODS))
    a2.invert_yaxis()
    a2.set_xlim(0, 1)
    a2.grid(axis="x", color="#e6e5e0", lw=1)
    a2.set_xlabel("hallucinations ÷ probed committed siblings\n(95% CI by item)")
    a2.set_title("Hallucination rate", loc="left", fontsize=16)
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, frameon=False, fontsize=11)
    fig.suptitle("Country siblings (32 user-modelling items): commitment judge → distractor-rank "
                 "probe → cross-method", x=0.01, ha="left", fontsize=18)
    fig.text(0.01, 0.005, "J-lens omitted (5 committed units). 'Nearby' = probe fires at a lens "
             "layer within ±4 (rev 8 definition; the L±1 version is being computed).",
             fontsize=11, color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(OUT, dpi=160)
    print(OUT)


if __name__ == "__main__":
    main()
