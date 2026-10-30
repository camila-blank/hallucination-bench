"""Tables, examples and figure for docs/results/0022_index_v6.md from a built hindex-v6 index.

Reads the v6 ``index.json`` (``hbench index``, default ``--version v6``), the hindex-v5 baseline
(the #45 run; ``tests/fixtures/index_v5_devbox.json`` is a copy), ``sj_offtopic_full.json``
(``hbench overrefusal offtopic-export``) and, for the examples only, ``sj_records_full.jsonl`` +
``data/overrefusal/items.json``. Nothing is recomputed: every number is an index's or the export's.

    python scripts/index/results_v6.py --v6 outputs/index/claude-opus-5-5/index.json \
        --v5 tests/fixtures/index_v5_devbox.json \
        --or-root /workspace/hb-overrefusal/outputs/overrefusal/claude-opus-5-5 \
        --fig plots/0022_index_v6.png [--examples]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OR = "overrefusal"
FAMS = ("main", "agentic_misalignment", "sibling", "story", OR)
# headline methods (the overrefusal readers); colours = validated categorical slots 1-3 of the
# dataviz reference palette (as in 0019 / 0021), markers as secondary encoding
HEADLINE = {
    "s3d-rl600": ("#2a78d6", "o"),
    "mlnla-rl600": ("#eb6834", "s"),
    "jlens": ("#1baf7a", "^"),
}
READER = {"s3d-rl600": "s3d-rl600", "mlnla-rl600": "mlnla-rl600-all11", "jlens": "jlens"}
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3dc", "#fcfcfb"


def pct(c: dict[str, Any] | None, n: bool = False) -> str:
    if not c or c.get("v") is None:
        return "—"
    s = f"{100 * c['v']:.1f} [{100 * c['ci'][0]:.1f}, {100 * c['ci'][1]:.1f}]"
    return f"{s} n={c['n']}" if n else s


def rows_by(index: dict[str, Any]) -> dict[str, dict[int, dict[str, Any]]]:
    out: dict[str, dict[int, dict[str, Any]]] = {}
    for r in index["rows"]:
        out.setdefault(r["method"], {})[int(r["layer"])] = r
    return out


def v6_table(by: dict[str, dict[int, dict[str, Any]]]) -> str:
    lines = [
        "| method | layer | full | matched | " + " | ".join(FAMS) + " |",
        "|---|---:|---|---|" + "---|" * len(FAMS),
    ]
    for m in HEADLINE:
        for layer, r in sorted(by[m].items()):
            cells = [pct(r.get("full")), pct(r.get("matched"))]
            cells += [pct(r["families"].get(f), n=True) for f in FAMS]
            lines.append(f"| {m} | {layer} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def delta_table(
    b5: dict[str, dict[int, dict[str, Any]]], b6: dict[str, dict[int, dict[str, Any]]]
) -> str:
    lines = [
        "| method | layer | OR v5 (items) | OR v6 (decline lines) | Δ OR | full v5 | full v6 "
        "| Δ full | matched v5 | matched v6 | Δ matched |",
        "|---|---:|---|---|---:|---|---|---:|---|---|---:|",
    ]

    def d(a: dict[str, Any] | None, b: dict[str, Any] | None) -> str:
        return f"{100 * (b['v'] - a['v']):+.1f}" if a and b else "—"

    for m in HEADLINE:
        for layer, r6 in sorted(b6[m].items()):
            r5 = b5[m][layer]
            cells = []
            for k in (OR, "full", "matched"):
                a = r5["families"].get(k) if k == OR else r5.get(k)
                b = r6["families"].get(k) if k == OR else r6.get(k)
                cells += [pct(a), pct(b), d(a, b)]
            lines.append(f"| {m} | {layer} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def secondary_table(sec: dict[str, Any]) -> str:
    lines = [
        "| reader | layer | index (all 427) | never_refused only | per disposition claim "
        "(never_refused) | plan-0020 item metric (v4/v5) | decline lines / failed item-layers |",
        "|---|---:|---|---|---|---|---:|",
    ]

    def m(c: dict[str, Any]) -> str:
        if c["v"] is None:
            return f"— ({c.get('absent_reason')})"
        return f"{100 * c['v']:.1f} [{100 * c['ci'][0]:.1f}, {100 * c['ci'][1]:.1f}]"

    for meth, rd in READER.items():
        for la, c in sorted(sec[rd].items(), key=lambda kv: int(kv[0])):
            ix, nv, dc, it = (
                c[k] for k in ("index", "never_refused", "per_disposition_claim", "item_metric")
            )
            dv = "—" if dc["v"] is None else f"{100 * dc['v']:.2f}"
            dcs = f"{dv} ({dc['n_off']} / {dc['n_disposition_claims']})"
            lines.append(
                f"| {rd} | {la} | {m(ix)} ({ix['n_off']} / {ix['n_claims']}) | "
                f"{m(nv)} ({nv['n_off']} / {nv['n_claims']}) | {dcs} | {m(it)} | "
                f"{ix['n_claims']} / {ix['n_failed']} |"
            )
    return "\n".join(lines)


def _plot_lines(ax: Any, series: list[tuple[str, str, str, list[tuple[int, Any]], bool]]) -> None:
    for name, col, mk, pts, solid in series:
        pts = [(x, c) for x, c in pts if c and c.get("v") is not None]
        if not pts:
            continue
        xs = [x for x, _ in pts]
        ys = [100 * c["v"] for _, c in pts]
        if solid:
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
            ls="-" if solid else "--",
            lw=2,
            marker=mk,
            ms=8 if solid else 6,
            color=col,
            mec=SURFACE,
            mew=1.5,
            alpha=1.0 if solid else 0.6,
            label=name,
        )
        if solid:
            ax.annotate(
                name.split(" · ")[0],
                (xs[-1], ys[-1]),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                fontsize=13,
                color=INK,
            )


def figure(
    b5: dict[str, dict[int, dict[str, Any]]], b6: dict[str, dict[int, dict[str, Any]]], dest: Path
) -> None:
    plt.rcParams.update({"font.size": 15, "axes.edgecolor": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.4), dpi=200, facecolor=SURFACE)
    titles = {
        OR: "overrefusal: % of decline lines off-topic (v6)",
        "full": "`full`",
        "matched": "`matched`",
    }
    for ax, key in zip(axes, (OR, "full", "matched"), strict=True):
        ax.set_facecolor(SURFACE)
        series = []
        for m, (col, mk) in HEADLINE.items():
            for by, ver in ((b5, "v5"), (b6, "v6")):
                if key == OR and ver == "v5":
                    continue  # v5's overrefusal is a different unit (items): not on this axis
                pts = [
                    (x, r["families"].get(OR) if key == OR else r.get(key))
                    for x, r in sorted(by.get(m, {}).items())
                ]
                series.append((f"{m} · {ver}", col, mk, pts, ver == "v6"))
        _plot_lines(ax, series)
        ax.set_title(titles[key], fontsize=17, loc="left", color=INK)
        ax.set_xlabel("layer")
        ax.set_ylabel("% (95% CI, v6)" if key != OR else "% off-topic (95% CI)")
        ax.set_xticks(range(20, 61, 8))
        ax.set_xlim(18, 70)
        ax.grid(axis="y", color=GRID, lw=1)
        ax.set_ylim(bottom=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED)
    h, labels = axes[2].get_legend_handles_labels()  # matched: every method
    fig.legend(
        h,
        labels,
        loc="lower center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=14,
    )
    fig.suptitle(
        "hindex-v6 (solid; overrefusal = wrong-topic refusal claims) vs hindex-v5 (dashed) by "
        "layer; jlens has no `full`",
        fontsize=18,
        x=0.01,
        ha="left",
        color=INK,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")


def examples(or_root: Path, items_path: Path, seed: int = 0) -> None:
    """Candidate examples (off-topic decline lines per reader; on-topic lines whose judge reason
    reads as a surface / keyword misreading) for hand selection."""
    items = {it["id"]: it for it in json.loads(items_path.read_text())["items"]}
    off: dict[str, list[dict[str, Any]]] = {}
    on: dict[str, list[dict[str, Any]]] = {}
    with open(or_root / "sj_records_full.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            o = r.get("off")
            if not o:
                continue
            for k in r["decline_lines"]:
                ex = {
                    "reader": r["reader"],
                    "layer": r["layer"],
                    "item": r["item"],
                    "population": r["population"],
                    "prompt": items[r["item"]]["prompt"],
                    "line": r["digest"]["lines"][k - 1],
                    "reason": o["reason"],
                }
                (off if o["topics"][str(k)] == "off_topic" else on).setdefault(
                    r["reader"], []
                ).append(ex)
    rng = random.Random(seed)
    for rd in sorted(off):
        print(f"\n### off-topic {rd} ({len(off[rd])})")
        for ex in rng.sample(off[rd], min(12, len(off[rd]))):
            print(json.dumps(ex, ensure_ascii=False))
    keys = ("misread", "surface", "keyword", "literal", "word")
    for rd in sorted(on):
        hits = [e for e in on[rd] if any(k in e["reason"].lower() for k in keys)]
        print(f"\n### on-topic misreading candidates {rd} ({len(hits)})")
        for ex in rng.sample(hits, min(6, len(hits))):
            print(json.dumps(ex, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v6", type=Path, required=True)
    ap.add_argument("--v5", type=Path, required=True)
    ap.add_argument("--or-root", type=Path, required=True)
    ap.add_argument("--items", type=Path, default=Path("data/overrefusal/items.json"))
    ap.add_argument("--fig", type=Path, required=True)
    ap.add_argument("--examples", action="store_true")
    a = ap.parse_args()
    i6, i5 = json.loads(a.v6.read_text()), json.loads(a.v5.read_text())
    assert (i6["version"], i5["version"]) == ("hindex-v6", "hindex-v5")
    b6, b5 = rows_by(i6), rows_by(i5)
    sec = json.loads((a.or_root / "sj_offtopic_full.json").read_text())
    print("## v6 table (overrefusal readers)\n")
    print(v6_table(b6))
    print("\n## v5 -> v6\n")
    print(delta_table(b5, b6))
    print("\n## secondary\n")
    print(secondary_table(sec["metrics"]))
    print("\nhealth:", json.dumps(sec["health"]))
    figure(b5, b6, a.fig)
    print(f"\nwrote {a.fig}")
    if a.examples:
        examples(a.or_root, a.items)


if __name__ == "__main__":
    main()
