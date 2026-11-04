"""``hbench final`` (plan 0029 §3.9): the single source of truth for hindex-v8 and its plots.

``data/index/v8_manifest.json`` (in git; written in the results PR) lists, per family x method x
unit (``L20`` ... ``L60``, ``all``), the root, the file and its sha256, the prompt / instrument
version, the judge model, ``n_cells`` and ``n_failed``::

    {"index_version": "hindex-v8",
     "max_failed_share": 0.01,
     "entries": [{"family": "main", "method": "s3d-rl600", "arm": "s3d-rl600", "unit": "L20",
                  "root": "outputs/v8/wsbench_chat/claude-opus-5-5",
                  "file": "s3d-rl600/cells.jsonl", "sha256": "...",
                  "prompt_version": "claimmap-v2.8", "judge_model": "claude-opus-5-5",
                  "n_cells": 1600, "n_failed": 3}, ...]}

``file`` is relative to ``root``; a safety family's file is the root's ``index_counts.jsonl``
(``index_counts_joint.jsonl`` for ``all``). :func:`main` checks every sha256 and completeness
(``n_failed / n_cells <= max_failed_share``, default 1%, per entry), and that every claim-map
entry (main, story, am_claims) records the **same** accepted v8 prompt (a registered
``claimmap-v2.8*``); any failure exits 2 naming the hole. It then builds hindex-v8 from exactly
the manifest's arm dirs (staged as symlinks under ``<out>/_sources``) and roots, writes
``index.json`` / ``index.md`` to ``--out`` and draws the plots to ``--plots``. Nothing outside the
manifest is plotted.

Plots (150 dpi, slide fonts, fixed colours, legends outside): ``index_per_layer.png`` (``full``
per layer with its CI band, ``full_j6`` thin, ``full_joint`` dotted horizontal),
``families_per_method.png`` (one panel per family; per-layer lines, the joint value dotted where
the family has one) and ``macro_all11.png`` (per method: the over-layers means of ``full``,
``matched``, ``full_j6`` and ``full_joint``, with CIs; the macro and the layers behind each mean
labelled). Per-layer uses k = 3 samples (J-lens 1); the joint unit 1 per layer.
"""

import hashlib
import json
import os
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from hallucination_bench import index as ix
from hallucination_bench.judges import claimmap_v2 as cm

MAX_FAILED_SHARE = 0.01
CLAIM_MAP_FAMILIES: tuple[str, ...] = (ix.MAIN, ix.STORY, ix.AM_CLAIMS)
ARM_DIR_FAMILIES: tuple[str, ...] = (ix.MAIN, ix.STORY, ix.AM_CLAIMS, ix.AM_SITUATION)
COLOURS: dict[str, str] = {
    "s3d-rl600": "#1f77b4",  # O-lens
    "mlnla-rl600": "#d62728",  # NLA
    "jlens": "#2ca02c",  # J-lens
}
LABELS: dict[str, str] = {"s3d-rl600": "O-lens", "mlnla-rl600": "NLA", "jlens": "J-lens"}
DPI = 150


class ManifestError(ValueError):
    """A manifest check failed (exit 2)."""


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    if m.get("index_version") != ix.INDEX_VERSION_V8:
        raise ManifestError(f"{path}: index_version must be {ix.INDEX_VERSION_V8}")
    if not isinstance(m.get("entries"), list) or not m["entries"]:
        raise ManifestError(f"{path}: no entries")
    need = ("family", "method", "unit", "root", "file", "sha256", "n_cells", "n_failed")
    for i, e in enumerate(m["entries"]):
        if miss := [k for k in need if k not in e]:
            raise ManifestError(f"{path}: entry {i} misses {miss}")
        if e["family"] not in ix.FAMILIES_V8:
            raise ManifestError(f"{path}: entry {i}: unknown family {e['family']!r}")
        if not (e["unit"] == ix.JOINT_LABEL or str(e["unit"]).startswith("L")):
            raise ManifestError(f"{path}: entry {i}: unit must be L<layer> or all")
    return m


def _where(e: Mapping[str, Any]) -> str:
    return f"{e['family']} x {e['method']} x {e['unit']}"


def check(m: Mapping[str, Any], base: Path) -> str:
    """Every sha256, completeness, one accepted prompt version; returns that version."""
    share = float(m.get("max_failed_share", MAX_FAILED_SHARE))
    for e in m["entries"]:
        p = (base / e["root"] / e["file"]).resolve()
        if not p.is_file():
            raise ManifestError(f"{_where(e)}: missing file {p}")
        if (got := sha256_file(p)) != e["sha256"]:
            raise ManifestError(f"{_where(e)}: sha256 mismatch for {p} ({got} != {e['sha256']})")
        n, f = int(e["n_cells"]), int(e["n_failed"])
        if n <= 0:
            raise ManifestError(f"{_where(e)}: no cells (hole)")
        if f / n > share:
            raise ManifestError(
                f"{_where(e)}: {f} of {n} cells failed ({100 * f / n:.2f}% > "
                f"{100 * share:.0f}%): the hole must be re-run"
            )
    versions = {e.get("prompt_version") for e in m["entries"] if e["family"] in CLAIM_MAP_FAMILIES}
    if len(versions) != 1:
        raise ManifestError(f"claim-map roots record different prompt versions: {versions}")
    (v,) = versions
    if v not in cm.VERSIONS or not cm.is_v28(v):
        raise ManifestError(f"claim-map prompt {v!r} is not an accepted v8 version (v2.8*)")
    return v


def stage_sources(m: Mapping[str, Any], base: Path, stage: Path) -> dict[str, list[Path]]:
    """Index sources from exactly the manifest: claim / situation arm dirs as symlinks under
    ``stage/<family>/<arm>``; other families' roots as listed."""
    sources: dict[str, list[Path]] = defaultdict(list)
    for e in m["entries"]:
        root = (base / e["root"]).resolve()
        fam = e["family"]
        if fam in ARM_DIR_FAMILIES:
            arm = e.get("arm") or Path(e["file"]).parts[0]
            dest = stage / fam / arm
            src = root / arm
            if dest.is_symlink() or dest.exists():
                if dest.resolve() != src.resolve():
                    raise ManifestError(f"{fam}: arm {arm} listed under two roots")
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(src, dest, target_is_directory=True)
            if stage / fam not in sources[fam]:
                sources[fam].append(stage / fam)
        elif root not in sources[fam]:
            sources[fam].append(root)
    return dict(sources)


def _layer(unit: str) -> int | str:
    return ix.JOINT_LABEL if unit == ix.JOINT_LABEL else int(str(unit)[1:])


def allowed(m: Mapping[str, Any]) -> dict[tuple[str, str], set[int | str]]:
    """(method, family) -> the units the manifest lists (what may be plotted)."""
    out: dict[tuple[str, str], set[int | str]] = defaultdict(set)
    for e in m["entries"]:
        out[(e["method"], e["family"])].add(_layer(e["unit"]))
    return dict(out)


# ---------------------------------------------------------------- plots


def _plt():  # type: ignore[no-untyped-def]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
        }
    )
    return plt


def _colour(method: str) -> str:
    return COLOURS.get(method, "#7f7f7f")


def _label(method: str) -> str:
    return f"{LABELS.get(method, method)} ({method})"


def plot_index_per_layer(index: Mapping[str, Any], keep: Mapping, dest: Path) -> Path:
    plt = _plt()
    fig, axis = plt.subplots(figsize=(13, 7))
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in index["rows"]:
        by[r["method"]].append(r)
    for method, rows in sorted(by.items()):
        if not any(k[0] == method for k in keep):
            continue
        c = _colour(method)
        pts = [
            (r["layer"], r["full"]) for r in rows if r["layer"] != ix.JOINT_LABEL and "full" in r
        ]
        if pts:
            xs = [p[0] for p in pts]
            axis.plot(
                xs,
                [100 * p[1]["v"] for p in pts],
                "-o",
                color=c,
                lw=2.5,
                label=f"{_label(method)} full",
            )
            axis.fill_between(
                xs,
                [100 * p[1]["ci"][0] for p in pts],
                [100 * p[1]["ci"][1] for p in pts],
                color=c,
                alpha=0.15,
            )
        j6 = [
            (r["layer"], r["full_j6"])
            for r in rows
            if r["layer"] != ix.JOINT_LABEL and "full_j6" in r
        ]
        if j6:
            axis.plot(
                [p[0] for p in j6],
                [100 * p[1]["v"] for p in j6],
                "-",
                color=c,
                lw=1,
                label=f"{_label(method)} full_j6",
            )
        joint = next((r for r in rows if r["layer"] == ix.JOINT_LABEL and "full_joint" in r), None)
        if joint is not None:
            axis.axhline(
                100 * joint["full_joint"]["v"],
                color=c,
                ls=":",
                lw=2,
                label=f"{_label(method)} full_joint (all 11 layers, 1 sample each)",
            )
    axis.set_xlabel("layer")
    axis.set_ylabel("hindex-v8 (% hallucinated)")
    axis.set_title("Hallucination index per layer (per-layer k = 3, J-lens 1; joint 1 per layer)")
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    p = dest / "index_per_layer.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_families(index: Mapping[str, Any], keep: Mapping, dest: Path) -> Path:
    plt = _plt()
    fams = [f for f in ix.FAMILIES_V8 if any(k[1] == f for k in keep)]
    ncol = 4
    nrow = max(1, -(-len(fams) // ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 5 * nrow), squeeze=False)
    for ax, f in zip(axes.flat, fams, strict=False):
        for method in sorted({r["method"] for r in index["rows"]}):
            units = keep.get((method, f))
            if not units:
                continue
            rows = [r for r in index["rows"] if r["method"] == method and f in r["families"]]
            pts = [
                (r["layer"], r["families"][f])
                for r in rows
                if r["layer"] != ix.JOINT_LABEL and r["layer"] in units
            ]
            c = _colour(method)
            if pts:
                ax.plot(
                    [p[0] for p in pts],
                    [100 * p[1]["v"] for p in pts],
                    "-o",
                    color=c,
                    lw=2,
                    label=_label(method),
                )
            jr = next((r for r in rows if r["layer"] == ix.JOINT_LABEL), None)
            if jr is not None and ix.JOINT_LABEL in units:
                ax.axhline(100 * jr["families"][f]["v"], color=c, ls=":", lw=2)
        ax.set_title(f)
        ax.set_xlabel("layer")
        ax.set_ylabel("%")
    for ax in list(axes.flat)[len(fams) :]:
        ax.axis("off")
    handles, labels = [], []
    for ax in axes.flat:
        h, lab = ax.get_legend_handles_labels()
        for hh, ll in zip(h, lab, strict=True):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("Families per method (dotted: the joint unit, all 11 layers)")
    fig.tight_layout()
    p = dest / "families_per_method.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_macro(index: Mapping[str, Any], keep: Mapping, dest: Path) -> Path:
    plt = _plt()
    over = {m: o for m, o in index.get("over_layers", {}).items() if any(k[0] == m for k in keep)}
    keys = ("full", "matched", "full_j6", "full_joint")
    fig, ax = plt.subplots(figsize=(13, 7))
    width = 0.8 / max(1, len(over))
    for i, (method, o) in enumerate(sorted(over.items())):
        xs, ys, lo, hi, notes = [], [], [], [], []
        for j, k in enumerate(keys):
            if k not in o:
                continue
            v = o[k]
            ci = v.get("ci") or [v["v"], v["v"]]
            xs.append(j + (i - (len(over) - 1) / 2) * width)
            ys.append(100 * v["v"])
            lo.append(100 * (v["v"] - ci[0]))
            hi.append(100 * (ci[1] - v["v"]))
            layers = v.get("layers")
            notes.append(f"L{layers[0]}-{layers[-1]} (n={len(layers)})" if layers else "joint")
        ax.bar(
            xs,
            ys,
            width=width,
            color=_colour(method),
            yerr=[lo, hi],
            capsize=4,
            label=_label(method),
        )
        for x, y, t in zip(xs, ys, notes, strict=True):
            ax.text(x, y, t, ha="center", va="bottom", fontsize=10, rotation=90)
    ax.set_xticks(
        range(len(keys)),
        [
            "full\n(8 families)",
            "matched\n(no sibling)",
            "full_j6\n(6 joint families)",
            "full_joint\n(joint row)",
        ],
    )
    ax.set_ylabel("% hallucinated")
    ax.set_title("Mean over the 11 layers (shared item bootstrap) and the joint unit")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    p = dest / "macro_all11.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


def filter_index(index: dict[str, Any], keep: Mapping) -> dict[str, Any]:
    """Only the (method, family, unit)s the manifest lists stay in the rows' families."""
    out = dict(index)
    rows = []
    for r in index["rows"]:
        fams = {
            f: v for f, v in r["families"].items() if r["layer"] in keep.get((r["method"], f), ())
        }
        if fams:
            rows.append({**r, "families": fams})
    out["rows"] = rows
    return out


def main(
    manifest: Path,
    *,
    out: Path,
    plots: Path,
    methods: Path,
    workers: int = 1,
    base: Path | None = None,
) -> int:
    """Exit 0, or 2 (with the reason on stderr) when a manifest check fails."""
    base = Path(base) if base is not None else Path.cwd()
    try:
        m = load_manifest(manifest)
        version = check(m, base)
        stage = Path(out) / "_sources"
        sources = stage_sources(m, base, stage)
    except ManifestError as e:
        print(f"hbench final: {e}", file=sys.stderr)
        return 2
    keep = allowed(m)
    built = ix.build_index_v8(sources, ix.load_methods(methods), workers)
    built["manifest"] = {
        "path": str(Path(manifest).resolve()),
        "sha256": sha256_file(manifest),
        "prompt_version": version,
    }
    js, md = ix.write_index(built, Path(out))
    plots = Path(plots)
    plots.mkdir(parents=True, exist_ok=True)
    shown = filter_index(built, keep)
    written = [
        plot_index_per_layer(shown, keep, plots),
        plot_families(shown, keep, plots),
        plot_macro(shown, keep, plots),
    ]
    print(f"wrote {js}, {md}; plots {', '.join(str(p) for p in written)}")
    return 0


def entry(
    family: str,
    method: str,
    unit: str,
    root: Path,
    file: str,
    *,
    base: Path | None = None,
    **kw: Any,
) -> dict[str, Any]:
    """A manifest entry with the file's sha256 (results PR helper)."""
    p = (Path(base or Path.cwd()) / root / file).resolve()
    return {
        "family": family,
        "method": method,
        "unit": unit,
        "root": str(root),
        "file": file,
        "sha256": sha256_file(p),
        **kw,
    }


__all__: Sequence[str] = ("check", "entry", "load_manifest", "main")
