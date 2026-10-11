"""Sibling-probe report (plan 0003 §Phase F): markdown tables + plots from
``outputs/sibling/score/summary.json``.

Writes ``outputs/sibling/score/report.md`` and ``plots/sibling_probe/``:
  detection_<arm>.png      per category: detection rate (tau_cat) of gold mentions, M, U_matched
  halluc_rate.png          per arm: probe-verified hallucination rate under tau_cat and tau(c|g)
  test2_<arm>.png          Test 2 scatter: mean z of mentioned vs unmentioned siblings per M cell
All numbers cover inventory concepts only.
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hallucination_bench import REPO_ROOT
from hallucination_bench.evals.sibling_probe.score import SCORE_DIR

PLOTS_DIR = REPO_ROOT / "plots" / "sibling_probe"
# validated categorical slots 1-3 of the dataviz reference palette (fixed order)
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
ARM_ORDER = ("s3d-rl600", "nla-rl-L42", "jlens")


def _fmt(ci: Mapping[str, Any] | None) -> str:
    if not ci or ci.get("est") is None:
        return "–"
    lo, hi = ci.get("lo"), ci.get("hi")
    s = f"{ci['est']:.3f}"
    if lo is not None and hi is not None:
        s += f" [{lo:.3f}, {hi:.3f}]"
    return s


def _setup_mpl() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 20,
            "axes.labelsize": 17,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
            "axes.edgecolor": INK2,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e4e3df",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
        }
    )
    return plt


def _err(ci: Mapping[str, Any] | None) -> tuple[float, float, float]:
    if not ci or ci.get("est") is None:
        return float("nan"), 0.0, 0.0
    e = ci["est"]
    lo = ci.get("lo") if ci.get("lo") is not None else e
    hi = ci.get("hi") if ci.get("hi") is not None else e
    return e, max(e - lo, 0.0), max(hi - e, 0.0)


def _bars(ax: Any, groups: list[str], series: list[tuple[str, str, list]], ylabel: str) -> None:
    import numpy as np

    n = len(series)
    width = 0.8 / max(n, 1)
    x = np.arange(len(groups))
    for j, (label, color, cis) in enumerate(series):
        vals = [_err(c) for c in cis]
        ax.bar(
            x + (j - (n - 1) / 2) * width,
            [v[0] for v in vals],
            width * 0.92,
            yerr=np.array([[v[1] for v in vals], [v[2] for v in vals]]),
            color=color,
            label=label,
            error_kw={"ecolor": INK2, "elinewidth": 1.5, "capsize": 3},
        )
    ax.set_xticks(x, [g.replace("_", " ") for g in groups], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)


def plot_detection(summary: Mapping[str, Any], out: Path) -> list[Path]:
    plt = _setup_mpl()
    paths = []
    for arm in ARM_ORDER:
        s = summary["arms"].get(arm)
        if not s:
            continue
        det = s["detection"]["tau_cat"]["by_category"]
        cats = sorted(det)
        if not cats:
            continue
        fig, ax = plt.subplots(figsize=(12, 6))
        _bars(
            ax,
            cats,
            [
                ("gold, at gold-mention cells", C1, [det[c].get("gold") for c in cats]),
                ("mentioned siblings (M)", C2, [det[c]["M"] for c in cats]),
                ("unmentioned (U matched)", C3, [det[c]["U_matched"] for c in cats]),
            ],
            "detection rate (τ_cat)",
        )
        ax.set_title(f"{arm}: probe detection by mention status")
        fig.tight_layout()
        p = out / f"detection_{arm}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return paths


def plot_halluc(summary: Mapping[str, Any], out: Path) -> Path | None:
    plt = _setup_mpl()
    arms = [a for a in ARM_ORDER if a in summary["arms"]]
    if not arms:
        return None
    fig, ax = plt.subplots(figsize=(10, 6))
    _bars(
        ax,
        arms,
        [
            (
                "τ_cat (headline)",
                C1,
                [summary["arms"][a]["halluc"]["tau_cat"]["all"]["rate"] for a in arms],
            ),
            (
                "τ(c | g) (strict)",
                C2,
                [summary["arms"][a]["halluc"]["tau_cg"]["all"]["rate"] for a in arms],
            ),
        ],
        "probe-verified hallucination rate",
    )
    ax.set_title("Sibling mentions the probe does not detect")
    fig.tight_layout()
    p = out / "halluc_rate.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_test2(summary: Mapping[str, Any], out: Path) -> list[Path]:
    import numpy as np

    plt = _setup_mpl()
    paths = []
    for arm in ARM_ORDER:
        s = summary["arms"].get(arm)
        pts = [p for p in (s or {}).get("test2_points", []) if None not in p[:2]]
        if not pts:
            continue
        zu = np.array([p[1] for p in pts])
        zm = np.array([p[0] for p in pts])
        fig, ax = plt.subplots(figsize=(8, 7.5))
        ax.scatter(
            zu,
            zm,
            s=64,
            color=C1,
            alpha=0.55,
            edgecolors=SURFACE,
            linewidths=1.5,
            label=f"M cells (n={len(pts)})",
        )
        lim = [min(zu.min(), zm.min()) - 0.3, max(zu.max(), zm.max()) + 0.3]
        ax.plot(lim, lim, color=INK2, lw=2, ls="--", label="y = x")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("mean z(c | g), unmentioned siblings")
        ax.set_ylabel("mean z(c | g), mentioned siblings")
        b2 = s["beta2"]
        ax.set_title(f"{arm}: Test 2 (β₂ = {_fmt(b2)})", fontsize=17)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
        fig.tight_layout()
        p = out / f"test2_{arm}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return paths


def markdown(summary: Mapping[str, Any]) -> str:
    L: list[str] = ["# Plan 0003 sibling probes: report", ""]
    L.append("All numbers cover inventory concepts only (see the coverage table).")
    prov = summary.get("provenance") or {}
    L.append(f"git {prov.get('git_sha')}; judge runs: `{json.dumps(prov.get('judge_runs'))}`")
    L += [
        "",
        "## Verdict",
        "",
        "| arm | role | subset | β₁ | β₂ | control | verdict | strong |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in summary.get("verdict", []):
        L.append(
            f"| {r['arm']} | {r['role']} | {r['subset']} | {_fmt(r.get('beta1'))} | "
            f"{_fmt(r.get('beta2'))} | {r['control_passes']} | {r['verdict']} | "
            f"{r.get('strongly_supported', '')} |"
        )
    for arm, s in summary["arms"].items():
        L += ["", f"## {arm}", ""]
        L.append(f"- classes (all sibling pairs): `{json.dumps(s['class_counts'])}`")
        L.append(f"- coverage: `{json.dumps(s['coverage'])}`")
        L.append(f"- response-named sibling mentions: `{json.dumps(s['response_named'])}`")
        L.append(
            f"- β₁ {_fmt(s['beta1'])} (n={s['beta1']['n']}); β₂ {_fmt(s['beta2'])} "
            f"(cells={s['beta2']['n']}, outrank median {s['beta2'].get('share_outrank_median')})"
        )
        L.append(
            f"- within-item AUROC {_fmt(s['within_item_auroc'])}; Test 3 τ(c|g) "
            f"{_fmt(s['test3']['tau_cg'])}, τ_cat {_fmt(s['test3']['tau_cat'])}"
        )
        L.append(
            f"- gold detection (τ_cat): all cells {_fmt(s['gold_detection']['all_cells'])}, "
            f"gold-mention cells {_fmt(s['gold_detection']['gold_mention_cells'])}"
        )
        L.append(f"- strong criterion: `{json.dumps(s['strong'])}`")
        L += [
            "",
            "| threshold | det(M) | det(U_matched) | ceiling | halluc rate | macro | "
            "macro (adjudicable) | user-scope rate |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for th in ("tau_cat", "tau_cg", "tau_neutral", "tau_cg_all"):
            d, h = s["detection"][th], s["halluc"][th]
            L.append(
                f"| {th} | {_fmt(d['M'])} | {_fmt(d['U_matched'])} | {_fmt(d['ceiling'])} | "
                f"{_fmt(h['all']['rate'])} | {_fmt(h['all'].get('macro'))} | "
                f"{_fmt(h['all'].get('macro_adjudicable'))} | "
                f"{_fmt(h['scope_user']['rate'])} |"
            )
        am = s["halluc_all_mentions"]
        L.append(
            f"| all sibling mentions (τ_cat) | | | {_fmt(am['ceiling'])} | "
            f"{_fmt(am['rate'])} | {_fmt(am.get('macro'))} | | |"
        )
        cg = s.get("category_gates") or {}
        L += [
            "",
            "A category is adjudicable if at least 50% of its (concept, layer) probes at this "
            "arm's layers pass both gates; otherwise its probes are too weak to adjudicate.",
            "",
            "| category | halluc τ_cat | ceiling | floor | halluc τ(c|g) | gate pass share | "
            "adjudicable |",
            "|---|---|---|---|---|---|---|",
        ]
        hc, hg = (
            s["halluc"]["tau_cat"]["all"]["by_category"],
            s["halluc"]["tau_cg"]["all"]["by_category"],
        )
        for cat in sorted(hc):
            gi = cg.get(cat) or {}
            adj = gi.get("adjudicable")
            flag = "–" if adj is None else ("yes" if adj else "probes too weak to adjudicate")
            L.append(
                f"| {cat} | {_fmt(hc[cat]['rate'])} | {_fmt(hc[cat]['ceiling'])} | "
                f"{_fmt(hc[cat]['floor'])} | {_fmt((hg.get(cat) or {}).get('rate'))} | "
                f"{gi.get('pass_share', '–')} | {flag} |"
            )
        pt = s.get("pair_table") or []
        if pt:
            L += [
                "",
                f"Top {len(pt)} (gold → sibling) pairs by eligible M pairs (means over eligible "
                "pairs of fold-averaged detection and z(c|g); sibling gates = the sibling probe "
                "passes both gates at every layer where the pair has an M pair):",
                "",
                "| category | gold → sibling | n | n M (all) | items | det τ_cat | det τ(c|g) | "
                "mean z(c|g) | sibling gates |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
            for r in pt:
                L.append(
                    f"| {r['category']} | {r['gold']} → {r['sibling']} | {r['n']} | "
                    f"{r['n_M_all']} | {r['n_items']} | {r['det_cat']} | {r['det_cg']} | "
                    f"{r['z_mean']} | {r['sibling_gates_all_layers']} |"
                )
        bl = s["halluc"]["tau_cat"]["by_leaf"]
        if bl:
            L += ["", "| leaf | halluc τ_cat |", "|---|---|"]
            L += [f"| {lf or '(none)'} | {_fmt(v)} |" for lf, v in bl.items()]
    cov = summary.get("coverage_inventory") or {}
    if cov:
        L += [
            "",
            "## Inventory coverage (probe_target outside the alias inventory)",
            "",
            "| arm | category | counts | share outside (est) | target/alias agreement |",
            "|---|---|---|---|---|",
        ]
        for arm, rows in cov.items():
            for cat, r in sorted(rows.items()):
                cnt = {k: v for k, v in r.items() if isinstance(v, int)}
                L.append(
                    f"| {arm} | {cat} | `{json.dumps(cnt)}` | "
                    f"{r.get('share_outside_inventory_est')} | "
                    f"{r.get('probe_target_alias_agreement')} |"
                )
    gates = summary.get("gates") or []
    fails = [g for g in gates if not g["pass"]]
    L += ["", f"## Gates: {len(gates) - len(fails)}/{len(gates)} (concept, layer) pass", ""]
    if fails:
        L += [
            "| category | concept | layer | validity | transfer | cat-layer pass |",
            "|---|---|---|---|---|---|",
        ]
        L += [
            f"| {g['category']} | {g['concept']} | {g['layer']} | {g['validity_auroc']} | "
            f"{g['transfer_auroc']} | {g['category_layer_pass']} |"
            for g in fails
        ]
    return "\n".join(L) + "\n"


def write_report(score_dir: Path = SCORE_DIR, plots_dir: Path = PLOTS_DIR) -> dict[str, Any]:
    summary = json.loads((score_dir / "summary.json").read_text())
    plots_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        *plot_detection(summary, plots_dir),
        plot_halluc(summary, plots_dir),
        *plot_test2(summary, plots_dir),
    ]
    md = score_dir / "report.md"
    md.write_text(markdown(summary))
    return {"report": str(md), "plots": [str(p) for p in paths if p is not None]}
