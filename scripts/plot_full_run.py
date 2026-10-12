"""Plots of the full claim-map run (plan 0002, judge claimmap-v2.7): every method, plus per-layer
curves for the 11-layer arms.

    uv run python scripts/plot_full_run.py [--out plots/full_run_v2.7]

Writes:
  methods_rates_hallucination.png  readout rate of hallucination / false claim / detail per method
  methods_rates_all.png            readout rate of every verdict per method (small multiples)
  methods_share_hallucination.png  share of readout text: hallucination / false claim / detail
  methods_share_all.png            share of readout text, all categories (stacked, sums to 100%)
  layers_rates.png / layers_share.png   per-layer curves for O-lens and multilayer NLA 600 / 800
Multi-layer methods are pooled over their layers in the method plots. Whiskers / bands are
item-bootstrap 95% CIs.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = "claude-opus-5-5"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
FOOT = (
    "Claim-map judge v2.7 (plan 0002 rev 17; scoring rev 18: junk = non-claim + off-topic), "
    "pre-gate · whiskers/bands: item-bootstrap 95% CI"
)

# (arm dir, label, group) - prose lenses first, then token lenses (judged via their summary)
METHODS = [
    ("s3d-rl600-all11", "O-lens RL (11 layers)", "prose"),
    ("s3d-rl600", "O-lens RL (5 layers)", "prose"),
    ("s3d-sft251", "O-lens SFT (5 layers)", "prose"),
    ("mlnla-rl600-all11", "ML-NLA step 600 (11 layers)", "prose"),
    ("mlnla-rl600-all11@44", "ML-NLA step 600 (L44)", "prose"),
    ("mlnla-rl2-800-all11", "ML-NLA step 800 (11 layers)", "prose"),
    ("nla-rl-L42", "NLA-RL (L42)", "prose"),
    ("nla-sft-L42", "NLA-SFT (L42)", "prose"),
    ("patchscopes", "Patchscopes (5 layers)", "prose"),
    ("jlens", "J-lens (5 layers)", "tokens"),
    ("rlens", "R-lens (5 layers)", "tokens"),
    ("tuned", "Tuned lens (5 layers)", "tokens"),
    ("logit", "Logit lens (5 layers)", "tokens"),
    ("template-v3", "Template lens (5 layers)", "tokens"),
    ("sae-l42", "SAE (L42)", "tokens"),
]
METRIC_COLORS = {"hallucination": "#e34948", "false_claim": "#2a78d6", "detail": "#eda100"}
HALLU = [("hallucination", "Hallucination"), ("false_claim", "False claim"), ("detail", "Detail")]
VERDICT_RATES = [
    ("hallucination", "Hallucination"),
    ("false_claim", "↳ False claim"),
    ("detail", "↳ Detail"),
    ("verdict_probe_candidate", "Probe candidate"),
    ("verdict_correct", "Correct"),
    ("verdict_unresolved", "Unverifiable"),
]
STACK = [
    ("leaf:false_claim", "Hallucination: false claim", "#e34948"),
    ("leaf:detail", "Hallucination: detail", "#eda100"),
    ("verdict:probe_candidate", "Probe candidate", "#4a3aa7"),
    ("verdict:correct", "Correct", "#1baf7a"),
    ("verdict:unresolved", "Unverifiable", "#2a78d6"),
    ("verdict:junk", "Junk / non-claim", "#b9b8b2"),  # rev 18: non_claim + off_topic
    ("verdict:failed", "Judge call failed", "#52514e"),
]
LAYER_ARMS = [
    ("s3d-rl600-all11", "O-lens RL", "#2a78d6"),
    ("mlnla-rl600-all11", "ML-NLA step 600", "#eb6834"),
    ("mlnla-rl2-800-all11", "ML-NLA step 800", "#1baf7a"),
]


def block_of(arm: str) -> dict:
    """``arm`` or ``arm@layer`` (one layer of a multi-layer arm) -> its metrics block."""
    name, _, layer = arm.partition("@")
    s = summary(name)
    return s["by_layer"][layer] if layer else s["overall"]


def summary(arm: str) -> dict:
    return json.loads(
        (ROOT / "outputs" / "wsbench_chat" / MODEL / arm / "summary_v2.json").read_text()
    )


def rate(block: dict, key: str) -> tuple[float, float, float]:
    r = block["rates"][key]["all"]
    return 100 * r["value"], 100 * r["ci95"][0], 100 * r["ci95"][1]


def share(block: dict, key: str) -> tuple[float, float, float]:
    kind, name = key.split(":")
    r = block["char_share_leaf" if kind == "leaf" else "char_share_verdict"].get(name)
    if r is None:
        return 0.0, 0.0, 0.0
    return 100 * r["value"], 100 * r["ci95"][0], 100 * r["ci95"][1]


def hshare(block: dict, key: str) -> tuple[float, float, float]:
    return share(block, "verdict:hallucination" if key == "hallucination" else f"leaf:{key}")


def clean(ax) -> None:
    ax.tick_params(axis="both", labelsize=12, colors=MUTED, length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)


def panels(data, metrics, getter, xlabel, title, out, colors=None) -> None:
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4.3 * n + 3.2, 7.4), dpi=160, sharey=True)
    y = np.arange(len(METHODS))[::-1]
    for i, (ax, (key, label)) in enumerate(zip(np.atleast_1d(axes), metrics, strict=True)):
        vals = [getter(s, key) for s in data]
        v = np.array([a for a, _, _ in vals])
        lo = v - np.array([b for _, b, _ in vals])
        hi = np.array([c for _, _, c in vals]) - v
        color = (colors or {}).get(key, "#2a78d6")
        ax.barh(y, v, height=0.68, color=color, edgecolor="white", linewidth=1.5)
        ax.errorbar(v, y, xerr=[lo, hi], fmt="none", ecolor=TEXT, elinewidth=1, capsize=2)
        for yi, vi, hii in zip(y, v, hi, strict=True):
            ax.text(
                vi + hii + 1,
                yi,
                f"{vi:.0f}" if vi >= 10 else f"{vi:.1f}",
                va="center",
                fontsize=10,
                color=TEXT,
            )
        ax.set_title(label, fontsize=14, color=TEXT)
        ax.set_xlim(0, min(100, float((v + hi).max()) * 1.18 + 3))
        ax.xaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
        ax.axhline(
            y[len([m for m in METHODS if m[2] == "prose"]) - 1] - 0.5,
            color=MUTED,
            linewidth=0.8,
            linestyle=":",
        )
        clean(ax)
        if i == 0:
            ax.set_yticks(y, [m[1] for m in METHODS], fontsize=12)
    fig.supxlabel(xlabel, fontsize=13, color=TEXT)
    fig.suptitle(title, fontsize=17, color=TEXT, x=0.01, ha="left")
    fig.text(
        0.01,
        0.005,
        FOOT + " · dotted line: prose lenses above, token lenses (judged via "
        "summary) below · multi-layer methods pooled over layers",
        fontsize=9,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(out)
    plt.close(fig)


def stacked(data, out) -> None:
    fig, ax = plt.subplots(figsize=(14, 8.2), dpi=160)
    y = np.arange(len(METHODS))[::-1]
    left = np.zeros(len(METHODS))
    for key, label, color in STACK:
        v = np.array([share(s, key)[0] for s in data])
        ax.barh(
            y, v, left=left, color=color, label=label, edgecolor="white", linewidth=1.5, height=0.68
        )
        for yi, li, vi in zip(y, left, v, strict=True):
            if vi >= 4:
                ax.text(
                    li + vi / 2,
                    yi,
                    f"{vi:.0f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if color in ("#e34948", "#4a3aa7", "#2a78d6") else TEXT,
                )
        left += v
    for yi, s in zip(y, data, strict=True):  # rev 18: the off-topic part of the junk segment
        ot = (s.get("char_share_junk_reason") or {}).get("off_topic") or {}
        if ot.get("value") is not None:
            ax.text(
                101,
                yi,
                f"off-topic {100 * ot['value']:.1f}%",
                va="center",
                fontsize=9,
                color=MUTED,
                clip_on=False,
            )
    ax.set_yticks(y, [m[1] for m in METHODS], fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of readout text (%)", fontsize=13, color=TEXT)
    clean(ax)
    ax.spines["left"].set_visible(False)
    ax.set_title(
        "Where each method's readout text goes (all categories)",
        fontsize=17,
        color=TEXT,
        loc="left",
    )
    ax.legend(fontsize=11, frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.08))
    fig.text(
        0.01,
        0.005,
        "Character-weighted, each readout weighted equally · " + FOOT,
        fontsize=9,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.03, 0.93, 1))
    fig.savefig(out)
    plt.close(fig)


def layer_curves(getter, metrics, ylabel, title, out) -> None:
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 5.4), dpi=160)
    for ax, (key, label) in zip(np.atleast_1d(axes), metrics, strict=True):
        for arm, name, color in LAYER_ARMS:
            bl = summary(arm)["by_layer"]
            layers = sorted(int(k) for k in bl)
            vals = [getter(bl[str(L)], key) for L in layers]
            v = [a for a, _, _ in vals]
            ax.fill_between(
                layers,
                [b for _, b, _ in vals],
                [c for _, _, c in vals],
                color=color,
                alpha=0.15,
                linewidth=0,
            )
            ax.plot(layers, v, color=color, linewidth=2, marker="o", markersize=5, label=name)
        ax.set_title(label, fontsize=14, color=TEXT)
        ax.set_xlabel("Layer", fontsize=12, color=TEXT)
        ax.set_xticks(range(20, 61, 8))
        ax.yaxis.grid(True, color=GRID)
        ax.set_ylim(bottom=0)
        clean(ax)
    np.atleast_1d(axes)[0].set_ylabel(ylabel, fontsize=13, color=TEXT)
    np.atleast_1d(axes)[0].legend(fontsize=11, frameon=False, loc="best")
    fig.suptitle(title, fontsize=16, color=TEXT, x=0.01, ha="left")
    fig.text(0.01, 0.005, FOOT + " · 1,123 cells per layer", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "plots" / "full_run_v2.7"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    data = [block_of(a) for a, _, _ in METHODS]
    ylab = "Readouts with ≥ 1 such segment (%)"
    panels(
        data,
        HALLU,
        rate,
        ylab,
        "Hallucination rate per method (readout level)",
        out / "methods_rates_hallucination.png",
        METRIC_COLORS,
    )
    panels(
        data,
        VERDICT_RATES,
        rate,
        ylab,
        "All categories per method (readout level)",
        out / "methods_rates_all.png",
    )
    panels(
        data,
        HALLU,
        hshare,
        "Share of readout text (%)",
        "Hallucinated share of readout text per method",
        out / "methods_share_hallucination.png",
        METRIC_COLORS,
    )
    stacked(data, out / "methods_share_all.png")
    layer_metrics = [
        *HALLU,
        ("verdict_probe_candidate", "Probe candidate"),
        ("verdict_correct", "Correct"),
    ]
    layer_curves(
        rate, layer_metrics, ylab, "By layer: readout-level rates", out / "layers_rates.png"
    )
    layer_curves(
        hshare,
        HALLU,
        "Share of readout text (%)",
        "By layer: hallucinated share of text",
        out / "layers_share.png",
    )
    headline(out / "hallucinationbench_per_method.png")
    headline(
        out / "hallucinationbench_per_method_and_layer.png",
        PER_LAYER_METHODS,
        height=17,
        title_suffix=" and layer",
    )
    print("wrote", *sorted(p.name for p in out.glob("*.png")))


LAYERS11 = range(20, 61, 4)
PER_LAYER_METHODS = [
    m for m in METHODS if m[0] not in ("s3d-rl600-all11", "mlnla-rl600-all11@44")
] + [
    *((f"s3d-rl600-all11@{L}", f"O-lens RL L{L}", "prose") for L in LAYERS11),
    *((f"mlnla-rl600-all11@{L}", f"ML-NLA step 600 L{L}", "prose") for L in LAYERS11),
    *((f"mlnla-rl2-800-all11@{L}", f"ML-NLA step 800 L{L}", "prose") for L in LAYERS11),
]


def headline(out: Path, methods=METHODS, height: float = 7.6, title_suffix: str = "") -> None:
    """One-panel HallucinationBench headline: readout-level hallucination rate per method (sorted,
    lower is better), with false-claim and detail rates as markers."""
    rows = []
    for arm, label, group in methods:
        o = block_of(arm)
        rows.append(
            (
                label,
                group,
                rate(o, "hallucination"),
                rate(o, "false_claim")[0],
                rate(o, "detail")[0],
            )
        )
    rows.sort(key=lambda r: r[2][0])
    fig, ax = plt.subplots(figsize=(11.5, height), dpi=160)
    y = np.arange(len(rows))[::-1]
    v = np.array([r[2][0] for r in rows])
    lo = v - np.array([r[2][1] for r in rows])
    hi = np.array([r[2][2] for r in rows]) - v
    ax.barh(
        y,
        v,
        height=0.66,
        color="#e34948",
        edgecolor="white",
        linewidth=1.5,
        label="Hallucination (≥ 1 false claim or detail)",
    )
    ax.errorbar(v, y, xerr=[lo, hi], fmt="none", ecolor=TEXT, elinewidth=1, capsize=2)
    ax.scatter(
        [r[3] for r in rows],
        y,
        s=60,
        color="#2a78d6",
        edgecolor="white",
        linewidth=1.5,
        zorder=3,
        label="False claim",
    )
    ax.scatter(
        [r[4] for r in rows],
        y,
        s=60,
        color="#eda100",
        edgecolor="white",
        linewidth=1.5,
        zorder=3,
        marker="D",
        label="Detail",
    )
    for yi, vi, hii in zip(y, v, hi, strict=True):
        ax.text(vi + hii + 1.2, yi, f"{vi:.1f}", va="center", fontsize=11, color=TEXT)
    labels = [f"{r[0]}{' †' if r[1] == 'tokens' else ''}" for r in rows]
    ax.set_yticks(y, labels, fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Readouts with ≥ 1 hallucination (%)  — lower is better", fontsize=13, color=TEXT)
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    clean(ax)
    ax.spines["left"].set_visible(False)
    ax.set_title(
        (
            "HallucinationBench v2 (wsbench_chat): hallucination rate per method" + title_suffix
        ).replace("(wsbench_chat): ", "(wsbench_chat):\n" if title_suffix else "(wsbench_chat): "),
        fontsize=16,
        color=TEXT,
        loc="left",
        pad=34,
    )
    ax.legend(fontsize=11, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    fig.text(
        0.01,
        0.005,
        "Hallucination = segment obviously contradicted by the text (quote required). "
        "† token lens, judged via its summary. Multi-layer methods pooled over layers.\n"
        "Claim-map judge v2.7 (plan 0002 rev 18), pre-gate · whiskers: item-bootstrap 95% CI",
        fontsize=9,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out)
    plt.close(fig)


if __name__ == "__main__":
    main()
