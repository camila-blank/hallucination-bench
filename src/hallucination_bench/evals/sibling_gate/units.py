"""Plan 0009 §1: sibling units, normalisation, ``item_sibling_names`` and the label join.

A **sibling unit** groups the ``substitution_sibling`` segments (scored leaf ``leaf``, segment
``status == "ok"``) of one cell (arm, id, layer, pos) that share the normalised ``pair.said`` and
``pair.actual``. Repeats across the cell's 3 samples are one unit; the same pair at another layer
or pos is another unit. Disputed segments, and segments whose ``alt_leaf`` alone is a sibling,
form no unit.

This module is imported by the rebuild path (``run.rebuild_arm``) and ``report_v2``, so it imports
nothing heavy and nothing from ``wsbench_chat.run``.
"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from hallucination_bench import REPO_ROOT

SIBLING_LEAF = "substitution_sibling"
OUT = REPO_ROOT / "outputs" / "sibling_gate"
CELLS_ROOT_DEFAULT = REPO_ROOT / "outputs" / "wsbench_chat" / "claude-opus-5-5"

# §1: the 13 arms of docs/results/0002_report_full_v2.7.md plus mlnla-rl600-all11 (the plan's
# "14 arms of the report" counts 13 report arms + this one); the mlnla-ck* sweep stays ungated.
PROSE_ARMS: tuple[str, ...] = (
    "s3d-rl600",
    "s3d-rl600-all11",
    "s3d-sft251",
    "mlnla-rl600-all11",
    "mlnla-rl2-800-all11",
    "nla-rl-L42",
    "nla-sft-L42",
    "patchscopes",
)
TOKEN_ARMS: tuple[str, ...] = ("jlens", "logit", "tuned", "rlens", "template-v3", "sae-l42")
ARMS_IN_SCOPE: tuple[str, ...] = (*PROSE_ARMS, *TOKEN_ARMS)
DETERMINISTIC_ARMS = frozenset(TOKEN_ARMS)  # §2a: token bags, no sampling -> rung 2 directly

# rung-1 / final statuses (§4)
RUNG1 = ("unstable", "stable", "sparse", "deterministic", "resample_failed")
FINALS = (
    "unstable",
    "probe_faithful",
    "probe_nearby",
    "cross_method",
    "probe_unconfirmed",
    "probe_na",
    "resample_failed",
)

SegKey = tuple[str, int, int, int]  # (cell key, readout idx, start, end)


def family_of(arm: str) -> str:
    """§3e lens families: s3d-*, mlnla-*, nla-*, patchscopes, and each token arm on its own."""
    for p in ("s3d", "mlnla", "nla"):
        if arm.startswith(p + "-"):
            return p
    return arm


# ---------------------------------------------------------------- normalisation


def norm(s: str | None) -> str:
    """NFKC, casefold, punctuation and whitespace stripped; if that leaves nothing, the
    casefolded raw string."""
    raw = s or ""
    t = unicodedata.normalize("NFKC", raw).casefold()
    out = "".join(
        ch for ch in t if not ch.isspace() and not unicodedata.category(ch).startswith("P")
    )
    return out if out else raw.casefold()


def cell_key(id_: str, layer: int, pos: int) -> str:
    return f"{id_}__L{int(layer):03d}__p{int(pos)}"


def unit_key(arm: str, ck: str, said_n: str, actual_n: str) -> str:
    return f"{arm}:{ck}:{said_n}:{actual_n}"


def fill_cache_key(u: dict[str, Any]) -> str:
    """§2b: ``fill:{arm}:{cellkey}:{said}:{actual}`` (normalised said / actual)."""
    return f"fill:{u['arm']}:{u['cell']}:{u['said_norm']}:{u['actual_norm']}"


# ---------------------------------------------------------------- units


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


def is_unit_segment(seg: dict[str, Any]) -> bool:
    return seg.get("leaf") == SIBLING_LEAF and seg.get("status") == "ok"


def units_of_cell(arm: str, rec: dict[str, Any]) -> list[dict[str, Any]]:
    """The sibling units of one ``cells.jsonl`` record (mapped cells only)."""
    m = rec.get("map")
    if not m or (rec.get("stages") or {}).get("map") != "ok":
        return []
    by: dict[tuple[str, str], dict[str, Any]] = {}
    for s in m.get("segments") or []:
        if not is_unit_segment(s):
            continue
        pair = s.get("pair") or {}
        said, actual = pair.get("said") or "", pair.get("actual") or ""
        k = (norm(said), norm(actual))
        u = by.get(k)
        if u is None:
            u = by[k] = {
                "unit": unit_key(arm, rec["key"], *k),
                "arm": arm,
                "id": rec["id"],
                "layer": int(rec["layer"]),
                "pos": int(rec["pos"]),
                "cell": rec["key"],
                "said": said,
                "actual": actual,
                "said_norm": k[0],
                "actual_norm": k[1],
                "segments": [],
            }
        u["segments"].append(
            {
                "key": rec["key"],
                "readout": int(s["readout"]),
                "start": int(s["start"]),
                "end": int(s["end"]),
                "text": s.get("text"),
                "unit_id": s.get("unit"),
                "contradicted_by": s.get("contradicted_by"),
            }
        )
    return list(by.values())


def build_units(arm: str, cells_path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Stream ``cells.jsonl`` (32 GB cap: never hold the records) -> units + counts."""
    units: list[dict[str, Any]] = []
    c = Counter()
    for rec in iter_jsonl(cells_path):
        us = units_of_cell(arm, rec)
        if us:
            c["n_cells"] += 1
            c["n_units"] += len(us)
            c["n_segments"] += sum(len(u["segments"]) for u in us)
            units.extend(us)
    return units, dict(c)


def units_path(arm: str, out: Path = OUT) -> Path:
    return Path(out) / "units" / f"{arm}.jsonl"


def load_units(arms: Iterable[str], out: Path = OUT) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for a in arms:
        p = units_path(a, out)
        if p.exists():
            rows.extend(iter_jsonl(p))
    return rows


def item_sibling_names(
    units: Iterable[dict[str, Any]], canonical: dict[tuple[str, str], str] | None = None
) -> dict[str, dict[str, dict[str, Any]]]:
    """item -> said object -> ``{arms, layers, sites}`` over every in-scope arm's units.

    Before §3a it is keyed by the normalised said; afterwards it is **rebuilt** keyed by
    ``said_canonical`` (``canonical``: (said_norm, item) -> said_canonical; units without an
    entry keep their normalised said)."""
    out: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for u in units:
        name = u["said_norm"]
        if canonical is not None:
            name = canonical.get((u["said_norm"], u["id"]), name)
        row = out[u["id"]].setdefault(name, {"arms": set(), "layers": set(), "sites": set()})
        row["arms"].add(u["arm"])
        row["layers"].add(int(u["layer"]))
        row["sites"].add((u["arm"], int(u["layer"]), int(u["pos"])))
    return dict(out)


# ---------------------------------------------------------------- labels (the scoring join)


def labels_dir(out_root: Path | None = None, override: Path | None = None) -> Path:
    """``<out_root>/sibling_gate/labels`` (the rebuild's ``--out``), unless overridden."""
    if override is not None:
        return Path(override)
    root = Path(out_root) if out_root is not None else REPO_ROOT / "outputs"
    return root / "sibling_gate" / "labels"


def load_labels(arm: str, labels_dir: Path | None = None) -> dict[SegKey, str | None] | None:
    """``{(cell key, readout idx, start, end): final}`` for one arm, or ``None`` when the arm has
    no labels file (rev-20 scoring). A null ``final`` (rung 2 unfinished) keeps the raw leaf."""
    p = Path(labels_dir if labels_dir is not None else OUT / "labels") / f"{arm}.jsonl"
    if not p.exists():
        return None
    out: dict[SegKey, str | None] = {}
    for row in iter_jsonl(p):
        for s in row.get("segments") or []:
            out[(s["key"], int(s["readout"]), int(s["start"]), int(s["end"]))] = row.get("final")
    return out


def seg_status(
    labels: dict[SegKey, str | None] | None, rec: dict[str, Any], seg: dict[str, Any]
) -> str | None:
    """A raw segment's ``sibling_status`` (``None``: no unit, no label, or a non-ok segment)."""
    if not labels or seg.get("status") != "ok":
        return None
    return labels.get((rec["key"], int(seg["readout"]), int(seg["start"]), int(seg["end"])))


# ---------------------------------------------------------------- label summaries (report)

SENSITIVITY_ROWS: tuple[tuple[str, str], ...] = (
    ("rung1_rev1", "(i) rev-1 rung-1 rule"),
    ("rung1_branch_a_only", "(ii) branch A only"),
    ("rung1_said_le1", "(iii) n_said <= 1"),
    ("rung1_said_le3", "(iv) n_said <= 3 (stable >= 4)"),
    ("cross_same_pos", "cross-method at the same pos"),
    ("cross_same_family", "same-family agreement counts"),
    ("cross_shared_prior", "shared-prior discount"),
)
HALLUCINATION_FINALS = frozenset({"unstable", "probe_unconfirmed"})  # = tx.SIBLING_HALLUCINATION


def summarize_label_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Units and segments by ``rung1`` and by ``final``, the unstable branch split, the
    ``no_gold`` / ``no_class`` counts, and per sensitivity row the gated (hallucination) units
    and segments."""
    out: dict[str, Any] = {
        "n_units": 0,
        "n_segments": 0,
        "rung1": {k: [0, 0] for k in RUNG1},
        "final": {k: [0, 0] for k in (*FINALS, "pending")},
        "branch": {"A": 0, "B": 0, "both": 0},
        "no_gold": 0,
        "no_class": 0,
        "diverse": 0,
        "gated": [0, 0],
        "sensitivity": {k: [0, 0] for k, _ in SENSITIVITY_ROWS},
    }
    for r in rows:
        n = len(r.get("segments") or [])
        out["n_units"] += 1
        out["n_segments"] += n
        r1 = r.get("rung1") or "resample_failed"
        out["rung1"].setdefault(r1, [0, 0])
        out["rung1"][r1][0] += 1
        out["rung1"][r1][1] += n
        fin = r.get("final") or "pending"
        out["final"].setdefault(fin, [0, 0])
        out["final"][fin][0] += 1
        out["final"][fin][1] += n
        if r1 == "unstable" and r.get("branch") in out["branch"]:
            out["branch"][r["branch"]] += 1
        flags = r.get("flags") or {}
        for f in ("no_gold", "no_class", "diverse"):
            out[f] += bool(flags.get(f))
        if fin in HALLUCINATION_FINALS:
            out["gated"][0] += 1
            out["gated"][1] += n
        for k, v in (r.get("sensitivity") or {}).items():
            if k in out["sensitivity"] and v in HALLUCINATION_FINALS:
                out["sensitivity"][k][0] += 1
                out["sensitivity"][k][1] += n
    return out
