"""Tables and figure for docs/results/0021_index_v5.md from a built hindex-v5 index.

Reads the v5 ``index.json`` (``hbench index``, default ``--version v5``) and the hindex-v4
baseline (the #42 run; ``tests/fixtures/index_v4_devbox.json`` is a copy), prints the doc's
markdown tables and writes the figure. Nothing is recomputed: every number is an index's.

    python scripts/index/results_v5.py --v5 outputs/index/claude-opus-5-5/index.json \
        --v4 tests/fixtures/index_v4_devbox.json --fig plots/0021_index_v5.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

AM = "agentic_misalignment"
FAMS = ("main", AM, "sibling", "story", "overrefusal")
# headline methods; colours fixed per method (validated categorical slots 1-3 of the dataviz
# reference palette, as in 0019), markers as secondary encoding
HEADLINE = {
    "s3d-rl600": ("#2a78d6", "o"),
    "mlnla-rl600": ("#eb6834", "s"),
    "jlens": ("#1baf7a", "^"),
}
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3dc", "#fcfcfb"


def pct(c: dict[str, Any] | None, n: bool = False) -> str:
    if not c:
        return "—"
    s = f"{100 * c['v']:.1f} [{100 * c['ci'][0]:.1f}, {100 * c['ci'][1]:.1f}]"
    return f"{s} n={c['n']}" if n else s


def rows_by(index: dict[str, Any]) -> dict[str, dict[int, dict[str, Any]]]:
    out: dict[str, dict[int, dict[str, Any]]] = {}
    for r in index["rows"]:
        out.setdefault(r["method"], {})[int(r["layer"])] = r
    return out


def am_methods(by: dict[str, dict[int, dict[str, Any]]]) -> list[str]:
    return sorted(m for m, ls in by.items() if any(AM in r["families"] for r in ls.values()))


def v5_table(by: dict[str, dict[int, dict[str, Any]]], methods: list[str]) -> str:
    head = "| method | layer | full | matched | AM n cand / n all | " + " | ".join(FAMS) + " |"
    lines = [head, "|---|---:|---|---|---:|" + "---|" * len(FAMS)]
    for m in methods:
        for layer, r in sorted(by[m].items()):
            am = r["families"].get(AM)
            if am is None and AM not in r.get("absent", {}):
                continue
            nn = f"{am['n']} / {am['n_all_claims']}" if am else "0 (absent)"
            cells = [pct(r.get("full")), pct(r.get("matched")), nn]
            cells += [pct(r["families"].get(f), n=True) for f in FAMS]
            lines.append(f"| {m} | {layer} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def delta_table(
    b4: dict[str, dict[int, dict[str, Any]]],
    b5: dict[str, dict[int, dict[str, Any]]],
    methods: list[str],
) -> str:
    lines = [
        "| method | layer | AM v4 | AM v5 | Δ AM | full v4 | full v5 | Δ full "
        "| matched v4 | matched v5 | Δ matched |",
        "|---|---:|---|---|---:|---|---|---:|---|---|---:|",
    ]

    def d(a: dict[str, Any] | None, b: dict[str, Any] | None) -> str:
        return f"{100 * (b['v'] - a['v']):+.1f}" if a and b else "—"

    for m in methods:
        for layer, r5 in sorted(b5[m].items()):
            r4 = b4.get(m, {}).get(layer)
            if r4 is None or AM not in r4["families"]:
                continue
            cells = []
            for k in (AM, "full", "matched"):
                a = r4["families"].get(k) if k == AM else r4.get(k)
                b = r5["families"].get(k) if k == AM else r5.get(k)
                if k != AM and a is None and b is None:  # V4 / methods without v4 macros
                    a, b = r4.get(f"{k}_v3"), r5.get(f"{k}_v3")
                cells += [pct(a), pct(b), d(a, b)]
            lines.append(f"| {m} | {layer} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def figure(
    b4: dict[str, dict[int, dict[str, Any]]], b5: dict[str, dict[int, dict[str, Any]]], dest: Path
) -> None:
    plt.rcParams.update({"font.size": 15, "axes.edgecolor": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2), dpi=200, facecolor=SURFACE)
    for ax, key in zip(axes, ("full", "matched"), strict=True):
        ax.set_facecolor(SURFACE)
        for m, (col, mk) in HEADLINE.items():
            for by, ls, ver in ((b4, "--", "v4"), (b5, "-", "v5")):
                pts = [(x, r.get(key)) for x, r in sorted(by.get(m, {}).items())]
                pts = [(x, c) for x, c in pts if c]
                if not pts:
                    continue
                xs = [x for x, _ in pts]
                ys = [100 * c["v"] for _, c in pts]
                if ver == "v5":
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
                    ys,
                    ls=ls,
                    lw=2,
                    marker=mk,
                    ms=8 if ver == "v5" else 6,
                    color=col,
                    mec=SURFACE,
                    mew=1.5,
                    alpha=1.0 if ver == "v5" else 0.65,
                    label=f"{m} · {ver}",
                )
                if ver == "v5":  # direct label at the line end
                    ax.annotate(
                        m,
                        (xs[-1], ys[-1]),
                        xytext=(8, 0),
                        textcoords="offset points",
                        va="center",
                        fontsize=13,
                        color=INK,
                    )
        ax.set_title(f"`{key}`", fontsize=17, loc="left", color=INK)
        ax.set_xlabel("layer")
        ax.set_ylabel("% hallucinated (95% CI, v5)")
        ax.set_xticks(range(20, 61, 4))
        ax.set_xlim(18, 68)
        ax.grid(axis="y", color=GRID, lw=1)
        ax.set_ylim(bottom=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED)
    h, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        h,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=14,
    )
    fig.suptitle(
        "hindex-v5 (solid, AM on candidate claims) vs hindex-v4 (dashed) by layer; "
        "jlens has no `full`",
        fontsize=18,
        x=0.01,
        ha="left",
        color=INK,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v5", type=Path, required=True)
    ap.add_argument("--v4", type=Path, required=True)
    ap.add_argument("--fig", type=Path, required=True)
    a = ap.parse_args()
    i5, i4 = json.loads(a.v5.read_text()), json.loads(a.v4.read_text())
    assert (i5["version"], i4["version"]) == ("hindex-v5", "hindex-v4")
    b5, b4 = rows_by(i5), rows_by(i4)
    ms = am_methods(b5)
    print("## v5 table (methods with an AM value)\n")
    print(v5_table(b5, ms))
    print("\n## v4 -> v5\n")
    print(delta_table(b4, b5, ms))
    figure(b4, b5, a.fig)
    print(f"\nwrote {a.fig}")


if __name__ == "__main__":
    main()
