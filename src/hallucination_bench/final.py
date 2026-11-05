"""``hbench final`` (plan 0029 §3.9): the single source of truth for hindex-v8 and its plots.

``data/index/v8_manifest.json`` (in git; written in the results PR) lists, per family x method x
unit (``L20`` ... ``L60``, ``all``), the root, the file and its sha256, the prompt / instrument
version, the judge model, ``n_cells`` and ``n_failed``::

    {"index_version": "hindex-v8",
     "max_failed_share": 0.01,
     "excuse_verifier": "cverify-v1",
     "entries": [{"family": "main", "method": "s3d-rl600", "arm": "s3d-rl600", "unit": "L20",
                  "root": "outputs/v8/wsbench_chat/claude-opus-5-5",
                  "file": "s3d-rl600/cells.jsonl", "sha256": "...",
                  "prompt_version": "claimmap-v2.9", "judge_model": "claude-opus-5-5",
                  "n_cells": 1600, "n_failed": 3}, ...]}

``file`` is relative to ``root``; a safety family's file is the root's ``index_counts.jsonl``
(``index_counts_joint.jsonl`` for ``all``). Required keys: family, method, unit, root, file,
sha256, prompt_version, judge_model (``arm`` defaults to the method; ``n_cells`` / ``n_failed``
are optional and must match the recount). :func:`main` checks every sha256, recounts cells and
failures from the hashed file itself (<= ``max_failed_share``, default 1%, per entry; no layer
outside the grid), requires the full unit grid per family x method (L20 ... L60, plus ``all``
for the joint families), and that every claim-map entry (main, story, am_claims) declares the
**same** accepted v8 prompt (:data:`V8_FINAL_PROMPTS`: only ``claimmap-v2.9``, plan 0029
amendment 1; a v2.8 or mixed manifest is refused; AM ``+am3`` versions normalised to their chat
version) and that each claim-map arm dir's ``arm.json`` records it (a main arm dir also records
``bank: x3``, plan 0028's bank). Plan 0029 Amendment 2: the manifest's top-level
``excuse_verifier`` and every claim-map arm dir's ``arm.json`` (and, for a chat / story ``all``
entry, ``joint.json``) must record the continuation verifier ``cverify-v1``; a cell whose
verifier verdict is failed / pending counts as failed in the recount, and any such cell refuses
the manifest outright (no ``max_failed_share`` allowance: re-run until clean); a chat / story
``all`` entry whose dir has no ``joint.json`` is refused. Any failure exits 2 naming
the hole. It then builds hindex-v8 from exactly the manifest's files (symlinked into a fresh
temp dir per run; a joint file is read only if an ``all`` entry hashes it) and nothing else,
writes ``index.json`` / ``index.md`` to ``--out`` (sources point at the manifest roots) and draws
the plots to ``--plots``. Nothing outside the manifest is plotted.

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
# plan 0029: the main family is judged on plan 0028's bank (arm.json `bank` / `bank_source`)
MAIN_BANK = "x3"
MAIN_BANK_SOURCE_PREFIX = "plan0028-x3:"
CLAIM_MAP_FAMILIES: tuple[str, ...] = (ix.MAIN, ix.STORY, ix.AM_CLAIMS)
# plan 0029 amendment 1: the claim-map prompt versions `hbench final` accepts for hindex-v8
# (chat versions; AM `+am3` normalised). v2.8 roots are pilot-only and never final.
V8_FINAL_PROMPTS: frozenset[str] = frozenset({"claimmap-v2.9"})
# plan 0029 Amendment 2: the continuation verifier every v8 claim-map root must have used (the
# manifest's top-level `excuse_verifier` and every claim-map arm dir's arm.json; joint.json too
# for a chat / story `all` entry)
V8_EXCUSE_VERIFIER = "cverify-v1"
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


REQUIRED_KEYS: tuple[str, ...] = (
    "family",
    "method",
    "unit",
    "root",
    "file",
    "sha256",
    "prompt_version",
    "judge_model",
)
# the per-layer unit grid every (family, method) must cover (plus `all` for the joint families)
EXPECTED_LAYERS: tuple[int, ...] = cm.JOINT_LAYERS


def load_manifest(path: Path) -> dict[str, Any]:
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    if m.get("index_version") != ix.INDEX_VERSION_V8:
        raise ManifestError(f"{path}: index_version must be {ix.INDEX_VERSION_V8}")
    if not isinstance(m.get("entries"), list) or not m["entries"]:
        raise ManifestError(f"{path}: no entries")
    for i, e in enumerate(m["entries"]):
        if miss := [k for k in REQUIRED_KEYS if k not in e]:
            raise ManifestError(f"{path}: entry {i} misses {miss}")
        if e["family"] not in ix.FAMILIES_V8:
            raise ManifestError(f"{path}: entry {i}: unknown family {e['family']!r}")
        if not (e["unit"] == ix.JOINT_LABEL or str(e["unit"]).startswith("L")):
            raise ManifestError(f"{path}: entry {i}: unit must be L<layer> or all")
    return m


def _where(e: Mapping[str, Any]) -> str:
    return f"{e['family']} x {e['method']} x {e['unit']}"


def _arm(e: Mapping[str, Any]) -> str:
    return str(e.get("arm") or e["method"])


def chat_version(v: Any) -> str:
    """A declared or recorded claim-map version as its chat version (``claimmap-v2.8+am3`` ->
    ``claimmap-v2.8``); unknown -> :class:`ManifestError`."""
    from hallucination_bench.judges import claimmap_am as cam

    try:
        return cam.spec(str(v)).chat_version
    except ValueError:
        raise ManifestError(f"unknown claim-map prompt version {v!r}") from None


def _unit_layer(v: Any) -> int | str:
    return ix.JOINT_LABEL if v == ix.JOINT_LABEL else int(v)


def recount(e: Mapping[str, Any], path: Path) -> tuple[int, int, set[int | str]]:
    """``(n_cells, n_failed, layers in the file)`` of one entry, from the hashed file itself
    (rows of the entry's arm only, in files shared across arms). A failed cell is anything not
    settled: a failed or pending map / detection / judge, or an unjudged safety row."""
    fam, unit, arm = e["family"], _layer(e["unit"]), _arm(e)
    n = f = 0
    seen: set[int | str] = set()
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if fam == ix.SIBLING and r.get("method") != arm:
                continue
            if fam in ix.SAFETY_FAMILIES_V8 and r.get("arm") != arm:
                continue
            la = _unit_layer(r["layer"])
            seen.add(la)
            if la != unit:
                continue
            if fam in (ix.MAIN, ix.STORY):
                if r.get("status") == "empty":
                    continue
                n += 1
                st = r.get("stages") or {}
                # plan 0029 Amendment 2: an unsettled verifier verdict is a failed cell too
                f += st.get("map") != "ok" or st.get("cverify") in ("failed", "pending")
            elif fam == ix.AM_CLAIMS:
                if r.get("status") == "empty":
                    continue
                n += 1
                bad_det = r.get("status") in ("pending", "failed")
                bad_cv = r.get("cverify_status") in ("failed", "pending")  # Amendment 2
                f += bad_det or r.get("map_status") not in (None, "ok") or bad_cv
            elif fam == ix.AM_SITUATION:
                if r.get("status") == "empty":
                    continue
                n += 1
                f += r.get("status") != "ok"
            elif fam == ix.SIBLING:
                n += 1
            else:
                n += 1
                f += int(r.get("n_failed_cells") or 0) > 0
    return n, f, seen


def unsettled_verdicts(e: Mapping[str, Any], path: Path) -> int:
    """Plan 0029 Amendment 2: the cells of one claim-map entry (its arm and unit, from the
    hashed file) whose continuation-verifier status is ``failed`` or ``pending`` (chat / story
    ``stages.cverify``, AM ``cverify_status``). :func:`check` requires zero: unlike other failed
    cells they get no ``max_failed_share`` allowance (re-run the cverify stage until clean)."""
    fam, unit = e["family"], _layer(e["unit"])
    if fam not in CLAIM_MAP_FAMILIES:
        return 0
    n = 0
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if _unit_layer(r["layer"]) != unit or r.get("status") == "empty":
                continue
            st = (
                r.get("cverify_status")
                if fam == ix.AM_CLAIMS
                else (r.get("stages") or {}).get("cverify")
            )
            n += st in ix.UNSETTLED_VERIFIER
    return n


def check(m: Mapping[str, Any], base: Path) -> str:
    """Every sha256; ``n_cells`` / ``n_failed`` recomputed from the hashed file (a declared
    count must agree); <= ``max_failed_share`` failed per entry; no layer outside the grid in
    a file; the full unit grid per family x method; one accepted v8 claim-map version, declared
    by every claim-map entry and recorded in each claim-map arm dir's ``arm.json``; a main arm
    dir's ``arm.json`` records the x3 bank. Returns the accepted (chat) version."""
    share = float(m.get("max_failed_share", MAX_FAILED_SHARE))
    grid = {f"L{la}" for la in EXPECTED_LAYERS}
    allowed_layers: set[int | str] = {*EXPECTED_LAYERS, ix.JOINT_LABEL}
    for e in m["entries"]:
        p = (base / e["root"] / e["file"]).resolve()
        if not p.is_file():
            raise ManifestError(f"{_where(e)}: missing file {p}")
        if (got := sha256_file(p)) != e["sha256"]:
            raise ManifestError(f"{_where(e)}: sha256 mismatch for {p} ({got} != {e['sha256']})")
        n, f, seen = recount(e, p)
        if extra := seen - allowed_layers:
            raise ManifestError(f"{_where(e)}: {p} holds layers outside the grid: {extra}")
        for k, v in (("n_cells", n), ("n_failed", f)):
            if k in e and int(e[k]) != v:
                raise ManifestError(f"{_where(e)}: declared {k} {e[k]} != {v} in {p}")
        if n <= 0:
            raise ManifestError(f"{_where(e)}: no cells (hole)")
        if u := unsettled_verdicts(e, p):  # plan 0029 Amendment 2: no allowance
            raise ManifestError(
                f"{_where(e)}: {u} cells with a failed / pending continuation-verifier verdict in "
                f"{p}: re-run the cverify stage until none is left (no failed-share allowance)"
            )
        if f / n > share:
            raise ManifestError(
                f"{_where(e)}: {f} of {n} cells failed ({100 * f / n:.2f}% > "
                f"{100 * share:.0f}%): the hole must be re-run"
            )
    units: dict[tuple[str, str], set[str]] = defaultdict(set)
    for e in m["entries"]:
        units[(e["family"], e["method"])].add(str(e["unit"]))
    holes = []
    for (fam, method), got_units in sorted(units.items()):
        want = grid | ({ix.JOINT_LABEL} if fam in ix.JOINT_FAMILIES_V8 else set())
        if missing := sorted(want - got_units):
            holes.append(f"{fam} x {method}: {missing}")
    if holes:
        raise ManifestError("incomplete unit grid: " + "; ".join(holes))
    claim = [e for e in m["entries"] if e["family"] in CLAIM_MAP_FAMILIES]
    versions = {chat_version(e["prompt_version"]) for e in claim}
    if len(versions) != 1:
        raise ManifestError(f"claim-map roots record different prompt versions: {versions}")
    (v,) = versions
    if v not in cm.VERSIONS or v not in V8_FINAL_PROMPTS:
        raise ManifestError(
            f"claim-map prompt {v!r} is not an accepted v8 version ({sorted(V8_FINAL_PROMPTS)})"
        )
    if (mv := m.get("excuse_verifier")) != V8_EXCUSE_VERIFIER:
        raise ManifestError(
            f"the manifest declares excuse_verifier {mv!r}; v8 needs {V8_EXCUSE_VERIFIER!r} "
            "(plan 0029 amendment 2)"
        )
    for e in claim:
        d = (base / e["root"] / e["file"]).resolve().parent
        a = d / "arm.json"
        if not a.is_file():
            raise ManifestError(f"{_where(e)}: no arm.json in {d} (cannot verify the prompt)")
        arm_rec = json.loads(a.read_text(encoding="utf-8"))
        rec = arm_rec.get("map_prompt") or cm.PROMPT_VERSION
        if chat_version(rec) != v:
            raise ManifestError(f"{_where(e)}: {a} records map_prompt {rec!r}, declared {v!r}")
        recs = [(a, arm_rec)]
        j = d / "joint.json"
        if e["unit"] == ix.JOINT_LABEL:
            if not j.is_file() and e["family"] in (ix.MAIN, ix.STORY):
                # a chat / story joint run records its own verifier setting in joint.json
                raise ManifestError(
                    f"{_where(e)}: no joint.json in {d} (cannot verify the joint excuse verifier)"
                )
            if j.is_file():
                recs.append((j, json.loads(j.read_text(encoding="utf-8"))))
        for path, r in recs:
            if (got := r.get("excuse_verifier")) != V8_EXCUSE_VERIFIER:
                raise ManifestError(
                    f"{_where(e)}: {path} records excuse_verifier {got!r}; v8 needs "
                    f"{V8_EXCUSE_VERIFIER!r} (plan 0029 amendment 2)"
                )
        if e["family"] == ix.MAIN:  # the v8 main family is plan 0028's x3 bank (missing = wsb)
            b, src = arm_rec.get("bank") or "wsb", str(arm_rec.get("bank_source") or "")
            if b != MAIN_BANK or not src.startswith(MAIN_BANK_SOURCE_PREFIX):
                raise ManifestError(
                    f"{_where(e)}: {a} records bank {b!r} (bank_source {src or None!r}); "
                    f"the v8 main family must be judged on --bank {MAIN_BANK}"
                )
    return v


def stage_sources(
    m: Mapping[str, Any], base: Path, stage: Path
) -> tuple[dict[str, list[Path]], dict[str, str]]:
    """Index sources holding exactly the manifest's files: each listed file is symlinked at its
    path relative to its root under ``stage/<family>/r<k>/`` (a safety root's
    ``index_meta.json`` beside it, for the unit guard). Returns the sources and a map from each
    staged root to the manifest root it stands for."""
    sources: dict[str, list[Path]] = defaultdict(list)
    roots: dict[tuple[str, Path], Path] = {}
    back: dict[str, str] = {}
    for e in m["entries"]:
        fam = e["family"]
        root = (base / e["root"]).resolve()
        if (fam, root) not in roots:
            roots[(fam, root)] = stage / fam / f"r{sum(f == fam for f, _ in roots)}"
            sources[fam].append(roots[(fam, root)])
            back[str(roots[(fam, root)].resolve())] = str(root)
        sroot = roots[(fam, root)]
        files = [e["file"]]
        if fam in ix.SAFETY_FAMILIES_V8 and (root / ix.SAFETY_META).is_file():
            files.append(ix.SAFETY_META)
        for rel in files:
            dest = sroot / rel
            if dest.is_symlink() or dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            if fam == ix.AM_SITUATION:  # one file holds every unit: keep the listed ones only
                units = {
                    _layer(x["unit"])
                    for x in m["entries"]
                    if x["family"] == fam
                    and x["file"] == rel
                    and (base / x["root"]).resolve() == root
                }
                with (root / rel).open(encoding="utf-8") as fi, dest.open("w") as fo:
                    for line in fi:
                        if line.strip() and _unit_layer(json.loads(line)["layer"]) in units:
                            fo.write(line)
                continue
            os.symlink((root / rel).resolve(), dest)
    return dict(sources), back


def _restore_paths(built: dict[str, Any], back: Mapping[str, str]) -> None:
    """Point ``sources`` / safety ``meta`` roots at the manifest roots, not the temp stage."""
    built["sources"] = {
        f: [back.get(p, p) for p in ps] for f, ps in built.get("sources", {}).items()
    }
    for blk in built.get("safety", {}).values():
        for mm in blk.get("meta", []):
            mm["root"] = back.get(mm["root"], mm["root"])


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
            "full\n(7 families)",
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
    import tempfile

    base = Path(base) if base is not None else Path.cwd()
    try:
        m = load_manifest(manifest)
        version = check(m, base)
    except ManifestError as e:
        print(f"hbench final: {e}", file=sys.stderr)
        return 2
    keep = allowed(m)
    with tempfile.TemporaryDirectory(prefix="hbench-final-") as tmp:  # fresh every run
        try:
            sources, back = stage_sources(m, base, Path(tmp))
        except ManifestError as e:
            print(f"hbench final: {e}", file=sys.stderr)
            return 2
        built = ix.build_index_v8(
            sources,
            ix.load_methods(methods),
            workers,
            prompt_version=version,
            excuse_verifier=V8_EXCUSE_VERIFIER,  # check() required it everywhere
        )
        _restore_paths(built, back)
    built["manifest"] = {
        "path": str(Path(manifest).resolve()),
        "sha256": sha256_file(manifest),
        "prompt_version": version,
        "excuse_verifier": V8_EXCUSE_VERIFIER,
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
