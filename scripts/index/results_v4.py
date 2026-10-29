"""Tables and figure for docs/results/0019_index_v4.md from a built hindex-v4 index.

Reads ``index.json`` (``hbench index --version v4``) and the overrefusal root's
``sj_metrics_full.json`` (plan 0020: metric 2 and digest health), prints the doc's markdown
tables and writes the figure. Nothing is recomputed: every number is the index's or the eval's.

    python scripts/index/results_v4.py --index outputs/index/claude-opus-5-5/index.json \
        --sj /workspace/hb-overrefusal/outputs/overrefusal/claude-opus-5-5/sj_metrics_full.json \
        --fig plots/0019_index_v4.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMS = ("main", "agentic_misalignment", "sibling", "story", "overrefusal")
MACROS = ("full", "matched", "full_v3", "matched_v3")
# methods carrying overrefusal (allowlisted readers); colours fixed per method (validated
# categorical slots 1-3 of the dataviz reference palette), markers as secondary encoding
SAFETY_METHODS = {
    "s3d-rl600": ("#2a78d6", "o"),
    "mlnla-rl600": ("#eb6834", "s"),
    "jlens": ("#1baf7a", "^"),
}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"


def pct(c: dict[str, Any] | None) -> str:
    if not c:
        return "—"
    lo, hi = c.get("ci") or (None, None)
    v = 100 * c["v"]
    return f"{v:.1f}" if lo is None else f"{v:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"


def rows_by(index: dict[str, Any]) -> dict[str, dict[int, dict[str, Any]]]:
    out: dict[str, dict[int, dict[str, Any]]] = {}
    for r in index["rows"]:
        out.setdefault(r["method"], {})[int(r["layer"])] = r
    return out


def macro_table(by: dict[str, dict[int, dict[str, Any]]], methods: list[str]) -> str:
    head = "| method | layer | " + " | ".join(MACROS) + " | " + " | ".join(FAMS) + " |"
    lines = [head, "|---|---:|" + "---|" * (len(MACROS) + len(FAMS))]
    for m in methods:
        for layer, r in sorted(by[m].items()):
            if not any(k in r for k in MACROS):
                continue
            cells = [pct(r.get(k)) for k in MACROS] + [pct(r["families"].get(f)) for f in FAMS]
            lines.append(f"| {m} | {layer} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def overrefusal_table(
    by: dict[str, dict[int, dict[str, Any]]], sj: dict[str, Any], arm_of: dict[str, str]
) -> str:
    lines = [
        "| method | layer | raw | probe-validated | n judged | refusal verdicts | "
        "contradicted | unprobed | unjudged | Sonnet digests (all pops) | metric 2 never_refused | "
        "metric 2 all populations |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for m in SAFETY_METHODS:
        arm = arm_of[m]
        for layer, r in sorted(by[m].items()):
            o = r["families"].get("overrefusal")
            if o is None:
                continue
            s = sj[arm][str(layer)]
            h = s["health"]
            m2n, m2a = s["m2_off_topic_never_refused"], s["m2_off_topic_all"]

            def ci(x: dict[str, Any]) -> str:
                return (
                    f"{100 * x['mean']:.1f} [{100 * x['lo']:.1f}, {100 * x['hi']:.1f}] (n={x['n']})"
                )

            lines.append(
                f"| {m} | {layer} | {100 * o['raw']:.1f} | {pct(o)} | {o['n_claims']} | "
                f"{o['n_false']} | {o['n_contradicted']} | {o['n_unprobed']} | "
                f"{o['health']['n_failed_cells']} | {h['n_digest_sonnet']} | {ci(m2n)} | "
                f"{ci(m2a)} |"
            )
    return "\n".join(lines)


def figure(by: dict[str, dict[int, dict[str, Any]]], dest: Path) -> None:
    plt.rcParams.update({"font.size": 15, "axes.edgecolor": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2), dpi=200, facecolor="#fcfcfb")
    panels = (
        (axes[0], "Macros: full (solid) and matched (dashed)", ("full", "matched")),
        (axes[1], "Overrefusal family (probe-validated)", ("overrefusal",)),
    )
    for ax, title, keys in panels:
        ax.set_facecolor("#fcfcfb")
        for m, (col, mk) in SAFETY_METHODS.items():
            for k, ls in zip(keys, ("-", "--"), strict=False):
                pts = [
                    (layer, r.get(k) or r["families"].get(k)) for layer, r in sorted(by[m].items())
                ]
                pts = [(x, c) for x, c in pts if c]
                if not pts:
                    continue
                xs = [x for x, _ in pts]
                ax.fill_between(
                    xs,
                    [100 * c["ci"][0] for _, c in pts],
                    [100 * c["ci"][1] for _, c in pts],
                    color=col,
                    alpha=0.10,
                    lw=0,
                )
                ax.plot(
                    xs,
                    [100 * c["v"] for _, c in pts],
                    ls=ls,
                    lw=2,
                    marker=mk,
                    ms=8,
                    color=col,
                    mec="#fcfcfb",
                    mew=1.5,
                    label=f"{m} · {k}",
                )
        ax.set_title(title, fontsize=17, loc="left", color=INK)
        ax.set_xlabel("layer")
        ax.set_ylabel("% hallucinated (95% CI)")
        ax.set_xticks(range(20, 61, 4))
        ax.grid(axis="y", color=GRID, lw=1)
        ax.set_ylim(bottom=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED)
    h0, l0 = axes[0].get_legend_handles_labels()
    fig.legend(
        h0,
        l0,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=14,
    )
    fig.suptitle(
        "hindex-v4 by layer (jlens has no sibling score, so no full)",
        fontsize=19,
        x=0.01,
        ha="left",
        color=INK,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--sj", type=Path, required=True)
    ap.add_argument("--fig", type=Path, required=True)
    a = ap.parse_args()
    index = json.loads(a.index.read_text())
    sj = json.loads(a.sj.read_text())
    by = rows_by(index)
    arm_of = {m: by[m][min(by[m])]["families"]["overrefusal"]["arm"] for m in SAFETY_METHODS}
    qwen = sorted(m for m in by if not m.startswith("v4-"))
    print("## macro table (Qwen)\n")
    print(macro_table(by, qwen))
    print("\n## macro table (V4)\n")
    print(macro_table(by, sorted(m for m in by if m.startswith("v4-"))))
    print("\n## overrefusal\n")
    print(overrefusal_table(by, sj, arm_of))
    figure(by, a.fig)
    print(f"\nwrote {a.fig}")


if __name__ == "__main__":
    main()
