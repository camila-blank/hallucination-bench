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
``all`` entry whose dir has no ``joint.json`` is refused. Plan 0029 Amendment 4: the manifest's
optional ``persistent_failures`` (``{path, sha256}``) names a list of persistent judge failures
(``scripts/v8/persistent_failures.py``); a listed cell must be a failed record of its entry in the
exact hashed file (``file_sha256``), is excluded from the failed-share and unsettled-verdict
gates (<= :data:`MAX_EXCLUDED_SHARE` per entry) and is reported (``index.json``, ``index.md``,
``excluded.md``); unlisted failures keep every rule above. Any failure exits 2 naming
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


def _cell_id(e: Mapping[str, Any], r: Mapping[str, Any]) -> str:
    """A failed cell's id in the persistent-failures list (plan 0029 Amendment 4): the record's
    ``key`` (claim-map / situation families; ``id__L<layer>__p<pos>`` when it has none) or the
    safety row's ``item``."""
    if e["family"] in ix.SAFETY_FAMILIES_V8:
        return str(r.get("item"))
    if r.get("key") is not None:
        return str(r["key"])
    return f"{r.get('id')}__L{r.get('layer')}__p{r.get('pos')}"


def scan(e: Mapping[str, Any], path: Path) -> tuple[int, list[tuple[str, bool]], set[int | str]]:
    """``(n_cells, failed, layers in the file)`` of one entry, from the hashed file itself (rows
    of the entry's arm only, in files shared across arms). ``failed`` holds one ``(cell id,
    verifier verdict unsettled)`` pair per failed record (:func:`_cell_id`). A failed cell is
    anything not settled: a failed or pending map / detection / judge, or an unjudged safety row;
    plan 0029 Amendment 2: an unsettled verifier verdict makes the cell failed too."""
    fam, unit, arm = e["family"], _layer(e["unit"]), _arm(e)
    n = 0
    failed: list[tuple[str, bool]] = []
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
            bad = cv = False
            if fam in (ix.MAIN, ix.STORY):
                if r.get("status") == "empty":
                    continue
                st = r.get("stages") or {}
                cv = st.get("cverify") in ix.UNSETTLED_VERIFIER
                bad = st.get("map") != "ok" or cv
            elif fam == ix.AM_CLAIMS:
                if r.get("status") == "empty":
                    continue
                bad_det = r.get("status") in ("pending", "failed")
                cv = r.get("cverify_status") in ix.UNSETTLED_VERIFIER  # Amendment 2
                bad = bad_det or r.get("map_status") not in (None, "ok") or cv
            elif fam == ix.AM_SITUATION:
                if r.get("status") == "empty":
                    continue
                bad = r.get("status") != "ok"
            elif fam != ix.SIBLING:
                bad = int(r.get("n_failed_cells") or 0) > 0
            n += 1
            if bad:
                failed.append((_cell_id(e, r), cv))
    return n, failed, seen


def recount(e: Mapping[str, Any], path: Path) -> tuple[int, int, set[int | str]]:
    """``(n_cells, n_failed, layers in the file)`` of one entry, from the hashed file itself
    (rows of the entry's arm only, in files shared across arms). A failed cell is anything not
    settled: a failed or pending map / detection / judge, or an unjudged safety row."""
    n, failed, seen = scan(e, path)
    return n, len(failed), seen


def unsettled_verdicts(e: Mapping[str, Any], path: Path) -> int:
    """Plan 0029 Amendment 2: the cells of one claim-map entry (its arm and unit, from the
    hashed file) whose continuation-verifier status is ``failed`` or ``pending`` (chat / story
    ``stages.cverify``, AM ``cverify_status``). :func:`check` requires zero: unlike other failed
    cells they get no ``max_failed_share`` allowance (re-run the cverify stage until clean)."""
    if e["family"] not in CLAIM_MAP_FAMILIES:
        return 0
    return sum(cv for _c, cv in scan(e, path)[1])


# ---------------------------------------------------------------- plan 0029 Amendment 4

PERSISTENT_VERSION = "v8-persistent-1"
# a cell is persistent only after this many failed attempts under its current fingerprint
PERSISTENT_MIN_ATTEMPTS = 3
# what a persistent failure may be: both models refused / an invalid answer after the in-run
# retries (never a transport error: those are re-run)
PERSISTENT_REASONS: frozenset[str] = frozenset({"refused", "invalid"})
# a broken-run bound on the excluded share per entry, not a quality gate
MAX_EXCLUDED_SHARE = 0.15
PERSISTENT_CELL_KEYS: tuple[str, ...] = (
    "family",
    "method",
    "unit",
    "cell",
    "reason",
    "attempts",
    "file_sha256",
)
# an entry whose excluded share is above this is flagged for review in excluded.md / index.md
REVIEW_EXCLUDED_SHARE = 0.05


def load_persistent(
    m: Mapping[str, Any], base: Path
) -> dict[tuple[str, str, str], dict[str, tuple[str, str]]]:
    """Plan 0029 Amendment 4: the manifest's ``persistent_failures`` list (``{"path", "sha256"}``,
    the path relative to ``base``) as ``(family, method, unit) -> {cell id: (reason, sha256 of the
    entry file the list was built from)}``; ``{}``
    when the manifest names none. Refused (:class:`ManifestError`): a sha256 mismatch, another
    list version, a cell missing a key, a reason outside :data:`PERSISTENT_REASONS`, fewer than
    :data:`PERSISTENT_MIN_ATTEMPTS` attempts, a cell listed twice, or a (family, method, unit) no
    manifest entry has."""
    ref = m.get("persistent_failures")
    if ref is None:
        return {}
    if not isinstance(ref, Mapping) or not {"path", "sha256"} <= set(ref):
        raise ManifestError("persistent_failures must be {'path': ..., 'sha256': ...}")
    p = (base / str(ref["path"])).resolve()
    if not p.is_file():
        raise ManifestError(f"persistent_failures: missing file {p}")
    if (got := sha256_file(p)) != ref["sha256"]:
        raise ManifestError(f"persistent_failures: sha256 mismatch for {p} ({got})")
    doc = json.loads(p.read_text(encoding="utf-8"))
    if doc.get("version") != PERSISTENT_VERSION:
        raise ManifestError(f"{p}: version must be {PERSISTENT_VERSION!r}")
    units = {(e["family"], e["method"], str(e["unit"])) for e in m["entries"]}
    out: dict[tuple[str, str, str], dict[str, tuple[str, str]]] = defaultdict(dict)
    for i, c in enumerate(doc.get("cells") or []):
        if miss := [k for k in PERSISTENT_CELL_KEYS if k not in c]:
            raise ManifestError(f"{p}: cell {i} misses {miss}")
        if c["reason"] not in PERSISTENT_REASONS:
            raise ManifestError(
                f"{p}: cell {i} ({c['cell']}): reason {c['reason']!r} is not persistent "
                f"({sorted(PERSISTENT_REASONS)}); re-run it"
            )
        if int(c["attempts"]) < PERSISTENT_MIN_ATTEMPTS:
            raise ManifestError(
                f"{p}: cell {i} ({c['cell']}): {c['attempts']} attempts < "
                f"{PERSISTENT_MIN_ATTEMPTS}; re-run it"
            )
        u = (str(c["family"]), str(c["method"]), str(c["unit"]))
        if u not in units:
            raise ManifestError(f"{p}: cell {i} names {u}, which no manifest entry has")
        if str(c["cell"]) in out[u]:
            raise ManifestError(f"{p}: cell {c['cell']!r} listed twice for {u}")
        out[u][str(c["cell"])] = (str(c["reason"]), str(c["file_sha256"]))
    return dict(out)


def check(m: Mapping[str, Any], base: Path, excluded: list[dict[str, Any]] | None = None) -> str:
    """Every sha256; ``n_cells`` / ``n_failed`` recomputed from the hashed file (a declared
    count must agree); <= ``max_failed_share`` failed per entry; no layer outside the grid in
    a file; the full unit grid per family x method; one accepted v8 claim-map version, declared
    by every claim-map entry and recorded in each claim-map arm dir's ``arm.json``; a main arm
    dir's ``arm.json`` records the x3 bank. Returns the accepted (chat) version."""
    share = float(m.get("max_failed_share", MAX_FAILED_SHARE))
    grid = {f"L{la}" for la in EXPECTED_LAYERS}
    allowed_layers: set[int | str] = {*EXPECTED_LAYERS, ix.JOINT_LABEL}
    persistent = load_persistent(m, base)  # plan 0029 Amendment 4 ({} without a list)
    for e in m["entries"]:
        p = (base / e["root"] / e["file"]).resolve()
        if not p.is_file():
            raise ManifestError(f"{_where(e)}: missing file {p}")
        if (got := sha256_file(p)) != e["sha256"]:
            raise ManifestError(f"{_where(e)}: sha256 mismatch for {p} ({got} != {e['sha256']})")
        n, failed, seen = scan(e, p)
        f = len(failed)
        if extra := seen - allowed_layers:
            raise ManifestError(f"{_where(e)}: {p} holds layers outside the grid: {extra}")
        listed = persistent.get((e["family"], e["method"], str(e["unit"])), {})
        for k, v in (("n_cells", n), ("n_failed", f), ("n_excluded", len(listed))):
            if k in e and int(e[k]) != v:
                raise ManifestError(f"{_where(e)}: declared {k} {e[k]} != {v} in {p}")
        if n <= 0:
            raise ManifestError(f"{_where(e)}: no cells (hole)")
        # Amendment 4: the list was built from exactly this hashed file, and a listed cell must
        # be a failed cell of this entry (a stale list refuses)
        if bad := sorted({h for _r, h in listed.values()} - {e["sha256"]}):
            raise ManifestError(
                f"{_where(e)}: the persistent-failure list was built from another {e['file']} "
                f"({bad[0][:12]}... != {e['sha256'][:12]}...): rebuild the list"
            )
        ids = [c for c, _cv in failed]
        if dup := sorted({c for c in ids if c in listed and ids.count(c) > 1}):
            raise ManifestError(
                f"{_where(e)}: listed cell ids are not unique among the failed records: {dup[:3]}"
            )
        if stale := sorted(set(listed) - set(ids)):
            raise ManifestError(
                f"{_where(e)}: {len(stale)} persistent-failure cells are not failed in {p} "
                f"(stale list): {stale[:3]}"
            )
        rest = [cv for c, cv in failed if c not in listed]
        # plan 0029 Amendment 2, retained by Amendment 4: no allowance for an unlisted
        # unsettled verdict
        if u := sum(rest):
            raise ManifestError(
                f"{_where(e)}: {u} cells with a failed / pending continuation-verifier verdict in "
                f"{p}: re-run the cverify stage until none is left (no failed-share allowance)"
            )
        if len(rest) / n > share:
            raise ManifestError(
                f"{_where(e)}: {len(rest)} of {n} cells failed ({100 * len(rest) / n:.2f}% > "
                f"{100 * share:.0f}%): the hole must be re-run"
            )
        if len(listed) / n > MAX_EXCLUDED_SHARE:
            raise ManifestError(
                f"{_where(e)}: {len(listed)} of {n} cells excluded as persistent failures "
                f"({100 * len(listed) / n:.2f}% > {100 * MAX_EXCLUDED_SHARE:.0f}%): a broken run"
            )
        if excluded is not None and listed:
            reasons: dict[str, int] = defaultdict(int)
            for r, _h in listed.values():
                reasons[r] += 1
            excluded.append(
                {
                    "family": e["family"],
                    "method": e["method"],
                    "unit": str(e["unit"]),
                    "n_cells": n,
                    "n_excluded": len(listed),
                    "share": len(listed) / n,
                    "by_reason": dict(sorted(reasons.items())),
                    "n_unsettled_verdicts": sum(cv for c, cv in failed if c in listed),
                }
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


EXCLUDED_MD = "excluded.md"


def excluded_md(excluded: Sequence[Mapping[str, Any]]) -> str:
    """Plan 0029 Amendment 4: the per-entry table of the cells excluded as persistent judge
    failures (written beside ``index.md`` when the manifest names a list)."""
    lines = [
        "# hindex-v8: cells excluded as persistent judge failures (plan 0029 Amendment 4)",
        "",
        "A cell that stayed failed after at least "
        f"{PERSISTENT_MIN_ATTEMPTS} attempts (both models refused, or an invalid answer after "
        "the in-run retries) is dropped from the index, never scored; counted here.",
        "",
        "Exclusions are not random: judges refuse harmful-sounding readouts, so each method is "
        "scored on its own (method-dependent) item set; safety `n_items` / bootstrap clusters "
        "still include the excluded all-zero rows. `review` = excluded share above "
        f"{100 * REVIEW_EXCLUDED_SHARE:.0f}%.",
        "",
        "| family | method | unit | cells | excluded | % | reasons | review |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for x in excluded:
        why = ", ".join(f"{k} {v}" for k, v in x["by_reason"].items())
        flag = "review" if x["share"] > REVIEW_EXCLUDED_SHARE else ""
        lines.append(
            f"| {x['family']} | {x['method']} | {x['unit']} | {x['n_cells']} | "
            f"{x['n_excluded']} | {100 * x['share']:.2f} | {why} | {flag} |"
        )
    if not excluded:
        lines.append("| (none) | | | | | | | |")
    lines += ["", f"Total excluded: {sum(x['n_excluded'] for x in excluded)}", ""]
    return "\n".join(lines)


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
        excluded: list[dict[str, Any]] = []
        version = check(m, base, excluded)
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
    if (ref := m.get("persistent_failures")) is not None:  # plan 0029 Amendment 4: reported
        built["manifest"]["persistent_failures"] = {
            "path": str(ref["path"]),
            "sha256": str(ref["sha256"]),
            "n_excluded": sum(x["n_excluded"] for x in excluded),
            "entries": excluded,
        }
    js, md = ix.write_index(built, Path(out))
    if ref is not None:  # Amendment 4: disclosed in index.md too, not only in excluded.md
        text = excluded_md(excluded)
        (Path(out) / EXCLUDED_MD).write_text(text, encoding="utf-8")
        with Path(md).open("a", encoding="utf-8") as fh:
            fh.write("\n" + text.replace("# hindex-v8: cells", "## Cells", 1))
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
