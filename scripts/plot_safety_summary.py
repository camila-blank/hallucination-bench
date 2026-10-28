"""Combined HB summary figure for the three safety evals (plans 0015, 0016, 0017).

Top row: the HB score (hallucination rate, % of readouts) per (method, layer); bottom row: the
probe-checked rate (claims at sites where the eval's probe contradicts the claim). Every
(method, layer) is its own point; nothing is averaged across layers.

Reads the generated report JSONs of the three eval worktrees (not in git; paths overridable):

- overrefusal: report.json + probe_report.json (never_refused n = 160, tail sites, refusal
  probe v2 threshold), 95% prompt-bootstrap CIs from the report.
- strongreject: report.json + probe_report.json (150 unsuccessful items, all prompt positions,
  pooled per-readout `harmful_compliance` claim rate). The report has no CI on the pooled rate.
- mask: report.json (155 gated honest_resisted items, site set `all`), 95% item-bootstrap CIs.
  Probe-checked uses the sensitivity threshold the results doc presents (LOAO OOF-honest q95;
  the pre-registered threshold is degenerate), pooled over prompt + response readouts.

Palette: dataviz reference slots 1/3/2 (blue / aqua / orange), matching scripts/plot_am.py's
method colours; validated all-pairs with scripts/validate_palette.js. Aqua is below 3:1
contrast on the surface, so each method also has its own marker and the doc carries tables.

    .venv/bin/python scripts/plot_safety_summary.py
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAYERS = [28, 36, 44, 52]
METHODS = [  # (key, label, colour, marker)
    ("olens", "O-lens (s3d-rl600)", "#2a78d6", "o"),
    ("mlnla", "Multilayer NLA (mlnla-rl600-all11)", "#1baf7a", "s"),
    ("jlens", "J-lens", "#eb6834", "D"),
]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

Series = dict[str, dict[str, list[float | None]]]  # method -> {"y", "lo", "hi"} (percent)


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def overrefusal(report: Path, probe: Path) -> tuple[Series, Series]:
    rep, prb = _load(report), _load(probe)
    arms = {"olens": "s3d-rl600", "mlnla": "mlnla-rl600-all11", "jlens": "jlens"}
    hb: Series = {}
    pc: Series = {}
    for key, arm in arms.items():
        hb[key] = {"y": [], "lo": [], "hi": []}
        pc[key] = {"y": [], "lo": [], "hi": []}
        for layer in LAYERS:
            pl = rep["arms"][arm]["scores"]["per_layer"][str(layer)]
            r = pl["never_refused"]["refusal"]["tail"]["readout"]
            t = prb["tagging"][arm][f"refusal-dm-v2/L{layer}"]["tail"]["probe_checked"]
            for out, src in ((hb[key], r), (pc[key], t)):
                out["y"].append(100 * src["mean"])
                out["lo"].append(100 * src["lo"])
                out["hi"].append(100 * src["hi"])
    return hb, pc


def strongreject(report: Path, probe: Path) -> tuple[Series, Series]:
    rep, prb = _load(report), _load(probe)
    arms = {"olens": "s3d-rl600", "mlnla": "mlnla-rl600", "jlens": "jlens"}
    hb: Series = {}
    pc: Series = {}
    for key, arm in arms.items():
        hb[key] = {"y": [], "lo": [], "hi": []}
        pc[key] = {"y": [], "lo": [], "hi": []}
        for layer in LAYERS:
            pooled = rep["arms"][arm]["layers"][str(layer)]["pooled"]["hc"]["all"]
            std = prb["arms"][arm]["standard"][str(layer)]["hc_all"]
            hb[key]["y"].append(100 * pooled["rate"])
            pc[key]["y"].append(100 * std["probe_checked_rate"])
            for s in (hb[key], pc[key]):  # no CI on the pooled rate in the report
                s["lo"].append(None)
                s["hi"].append(None)
    return hb, pc


def mask(report: Path) -> tuple[Series, Series]:
    rep = _load(report)
    arms = {"olens": "s3d-rl600", "mlnla": "mlnla-rl600-all11", "jlens": "jlens"}
    hb: Series = {}
    pc: Series = {}
    for key, arm in arms.items():
        hb[key] = {"y": [], "lo": [], "hi": []}
        pc[key] = {"y": [], "lo": [], "hi": []}
        for layer in LAYERS:
            v = rep["arms"][arm][f"L{layer:03d}:all"]["verified"]["honest_resisted"]
            hb[key]["y"].append(100 * v["readout_rate"])
            hb[key]["lo"].append(100 * v["readout_ci"][0])
            hb[key]["hi"].append(100 * v["readout_ci"][1])
            probe = rep["probe"]["arms"][arm]
            p, r = probe[f"L{layer:03d}:prompt"], probe[f"L{layer:03d}:response"]
            n = p["loao_oof_hr_readouts"] + r["loao_oof_hr_readouts"]
            contradicted = p["loao_oof_hr_contradicted"] + r["loao_oof_hr_contradicted"]
            claims = contradicted + p["loao_oof_hr_supported"] + r["loao_oof_hr_supported"]
            # pooling must reproduce the HB `all` rate exactly, or the denominators differ
            assert abs(claims / n - v["readout_rate"]) < 5e-5, (arm, layer)
            pc[key]["y"].append(100 * contradicted / n)
            pc[key]["lo"].append(None)
            pc[key]["hi"].append(None)
    return hb, pc


def draw(ax: Any, series: Series, *, title: str | None, ylabel: str | None) -> None:
    for key, label, colour, marker in METHODS:
        s = series[key]
        ys = s["y"]
        ax.plot(
            LAYERS,
            ys,
            color=colour,
            lw=2.5,
            marker=marker,
            ms=10,
            mec="#fcfcfb",
            mew=2,
            label=label,
            zorder=3,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
        if all(v is not None for v in s["lo"]):
            lo = [y - lo for y, lo in zip(ys, s["lo"], strict=True)]  # type: ignore[operator]
            hi = [hi - y for y, hi in zip(ys, s["hi"], strict=True)]  # type: ignore[operator]
            ax.errorbar(
                LAYERS,
                ys,
                yerr=[lo, hi],
                fmt="none",
                ecolor=colour,
                elinewidth=1.8,
                capsize=5,
                capthick=1.8,
                zorder=2,
            )
    if title:
        ax.set_title(title, fontsize=17, color=INK, loc="left", pad=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=15, color=INK2)
    ax.set_xticks(LAYERS)
    ax.set_xlabel("layer", fontsize=15, color=INK2)
    ax.tick_params(labelsize=14, colors=INK2, length=0)
    ax.set_ylim(bottom=0)
    ax.margins(x=0.08)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--or-dir",
        type=Path,
        default=Path("/workspace/hb-overrefusal/outputs/overrefusal/claude-sonnet-5-5"),
    )
    ap.add_argument(
        "--sr-dir",
        type=Path,
        default=Path("/workspace/hb-strongreject/outputs/strongreject/claude-opus-5-5"),
    )
    ap.add_argument(
        "--mask-dir", type=Path, default=Path("/workspace/hb-mask/outputs/mask/claude-opus-5-5")
    )
    ap.add_argument("--out", type=Path, default=Path("plots/safety_evals_hb_summary.png"))
    args = ap.parse_args()

    panels = [
        (
            "Overrefusal: false refusal claims\nn = 160 never-refused prompts · tail (4) sites",
            "Probe-checked: claims where the\nrefusal probe (v2) says refusal is absent",
            *overrefusal(args.or_dir / "report.json", args.or_dir / "probe_report.json"),
        ),
        (
            "StrongREJECT: false compliance claims\nn = 150 failed jailbreaks · all prompt sites",
            "Probe-checked: claims where the\nrefusal probe (v2) says refusal is present",
            *strongreject(args.sr_dir / "report.json", args.sr_dir / "probe_report.json"),
        ),
        (
            "MASK: false lie claims\nn = 155 honest-resisted items · all sites",
            "Probe-checked: claims where no lie\nis represented (sensitivity τ)",
            *mask(args.mask_dir / "report.json"),
        ),
    ]

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(2, 3, figsize=(19, 10.5), facecolor="#fcfcfb")
    for col, (title, pc_title, hb, pc) in enumerate(panels):
        draw(
            axes[0, col],
            hb,
            title=title,
            ylabel="HB hallucination rate\n(% of readouts)" if col == 0 else None,
        )
        draw(
            axes[1, col],
            pc,
            title=pc_title,
            ylabel="probe-checked rate\n(% of readouts)" if col == 0 else None,
        )
        axes[1, col].title.set_fontsize(14)
        axes[1, col].title.set_color(INK2)
    for ax in axes.flat:
        ax.set_facecolor("#fcfcfb")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        fontsize=15,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
        handlelength=2.5,
    )
    fig.text(
        0.01,
        0.005,
        "Error bars: 95% bootstrap CIs from the reports (overrefusal, MASK HB score). "
        "StrongREJECT has no CI on the pooled rate. Y-axes are independent per panel.\n"
        "MASK probe: LOAO OOF-honest q95 (pre-registered threshold is degenerate).",
        fontsize=12,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94), h_pad=3)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170, facecolor=fig.get_facecolor())
    print(f"wrote {args.out}")
    for title, _, hb, pc in panels:
        print(title.splitlines()[0])
        for key, label, _, _ in METHODS:
            fmt = " ".join(f"{y:.2f}" for y in hb[key]["y"])
            fmt_pc = " ".join(f"{y:.2f}" for y in pc[key]["y"])
            print(f"  {label:36s} HB {fmt} | probe-checked {fmt_pc}")


if __name__ == "__main__":
    main()
