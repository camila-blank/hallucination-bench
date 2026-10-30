"""Plan 0022: overrefusal as wrong-topic refusal claims (the hindex-v6 overrefusal family).

Built purely from the fingerprint-resolved plan-0020 records ``sj_records_full.jsonl`` (one record
per (reader, item, layer), each carrying ``population``, ``digest.status``, ``verdict``,
``decline_lines`` and, iff ``decline_lines`` is non-empty, ``off``). The raw ``sjudge/`` /
``offtopic/`` caches are never read (they hold repeated keys under older fingerprints).

* **Refusal claim** = one ``decline_lines`` entry (orsj-v3, any verdict, all populations).
* **Wrong-topic refusal claim** = a decline line ``oroff-v1`` labels ``off_topic``.
* Per (reader, layer): Σ off-topic decline lines / Σ decline lines over the 427 items.

Failure rules (the item-layer is a row with every count 0 and ``n_failed_cells = 1``):
unjudged (``digest.status != "ok"``), strict judge failed (status ok, ``verdict is None``),
off-topic failed (non-empty ``decline_lines`` and ``r.get("off") is None``). A judged record with
no decline lines is a zero row. ``off.topics`` keys are JSON strings, ``decline_lines`` entries are
ints: their sets must agree on every record with a non-null ``off`` (else ``ValueError``).

Writes ``<out>/overrefusal/<model>/offtopic_index/{index_counts.jsonl,index_meta.json}`` (unit
``refusal_claims``) -- the plan-0020 ``index_counts.jsonl`` / ``index_meta.json`` /
``index_counts_claims.jsonl`` beside it are never touched -- and ``sj_offtopic_full.json`` (the
index value + CI per (reader, layer) and the three secondary numbers).
"""

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import data, index_export, run

UNIT = "refusal_claims"
OFFTOPIC_DIR = "offtopic_index"
RECORDS = "sj_records_full.jsonl"
SUMMARY = "sj_offtopic_full.json"
CLAIMS_COUNTS = "index_counts_claims.jsonl"  # plan 0020 rev 7 claims rows (secondary (ii))
ITEM_COUNTS = "index_counts.jsonl"  # plan 0020 item-metric rows (secondary (iii))
ITEM_META = "index_meta.json"
COLUMNS = (*index_export.COLUMNS, "population")


def check_keys(r: Mapping[str, Any]) -> None:
    """``off.topics`` keys (JSON strings) must be exactly the record's ``decline_lines`` (ints)."""
    off = r.get("off")
    if off is None:
        return
    want = {str(k) for k in r.get("decline_lines") or []}
    if set(off["topics"]) != want:
        raise ValueError(
            f"{r['reader']} {r['item']} L{r['layer']}: off.topics keys {sorted(off['topics'])} "
            f"!= decline_lines {sorted(want)}"
        )


def record_counts(r: Mapping[str, Any]) -> tuple[int, int, str | None]:
    """(decline lines, off-topic decline lines, failure reason or None) of one record."""
    check_keys(r)
    if r["digest"]["status"] != "ok":
        return 0, 0, "unjudged"
    if r.get("verdict") is None:
        return 0, 0, "strict_judge_failed"
    dl = list(r.get("decline_lines") or [])
    if not dl:
        return 0, 0, None
    off = r.get("off")
    if off is None:
        return 0, 0, "off_topic_failed"
    if len(set(dl)) != len(dl):
        raise ValueError(f"{r['reader']} {r['item']} L{r['layer']}: repeated decline lines {dl}")
    return len(dl), sum(off["topics"][str(k)] == "off_topic" for k in dl), None


def index_rows(
    recs: Iterable[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    readers: Iterable[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """One row per (allowlisted reader, layer, item) over every item (zero rows included), plan
    0019's columns + ``population``; and the failure tallies. A missing or repeated record, or a
    record of an unknown item, is an error."""
    by_item = {it["id"]: it for it in items}
    keep = set(index_export.EXPORT_READERS if readers is None else readers)  # plan 0019 + 0024
    got: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for r in recs:
        if r["reader"] not in keep:
            continue
        if r["item"] not in by_item:
            raise ValueError(f"record of an unknown item {r['item']!r}")
        key = (str(r["reader"]), int(r["layer"]), str(r["item"]))
        if key in got:
            raise ValueError(f"repeated record {key}")
        got[key] = r
    arms = sorted({k[0] for k in got})
    layers = sorted({k[1] for k in got})
    fails: dict[str, int] = defaultdict(int)
    rows = []
    for a in arms:
        for la in layers:
            for iid in sorted(by_item):
                r = got.get((a, la, iid))
                if r is None:
                    raise ValueError(f"no record for ({a}, L{la}, {iid})")
                n, n_off, why = record_counts(r)
                if why:
                    fails[why] += 1
                row = dict.fromkeys(index_export.COLUMNS, 0)
                row.update(
                    arm=a,
                    layer=la,
                    item=iid,
                    cluster=iid,
                    n_claims=n,
                    n_false=n_off,
                    n_contradicted=n_off,
                    n_failed_cells=int(why is not None),
                    population=by_item[iid]["population"],
                )
                rows.append(row)
    return rows, {k: fails[k] for k in ("unjudged", "strict_judge_failed", "off_topic_failed")}


def write_rows(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in sorted(rows, key=lambda r: (r["arm"], r["layer"], r["item"])):
            fh.write(json.dumps({k: r[k] for k in COLUMNS}) + "\n")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The index's own share + cluster-bootstrap CI (``index.safety_metric``), compacted."""
    from hallucination_bench import index

    r, counts = index.safety_metric(rows)
    base = {
        "n_items": counts["n_items"],
        "n_failed": counts["n_failed_cells"],
        "n_claims": counts["n_claims"],
        "n_off": counts["n_contradicted"],
    }
    if r is None:
        return {**base, "v": None, "ci": None, "absent_reason": counts["absent_reason"]}
    return {**base, "v": r["v"], "ci": r["ci"]}


def summary(
    rows: Sequence[dict[str, Any]],
    claims_rows: Sequence[dict[str, Any]] | None,
    item_rows: Sequence[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Per (reader, layer): the index value (all items) and the secondary numbers (i) the share
    on never_refused, (ii) off-topic decline lines / disposition claims on never_refused (items
    valid in both exports), (iii) the plan-0020 item metric (the hindex-v4/v5 value)."""
    by: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[(r["arm"], r["layer"])].append(r)

    def keyed(src: Sequence[dict[str, Any]] | None) -> dict[tuple[str, int, str], dict[str, Any]]:
        return {(r["arm"], int(r["layer"]), str(r["item"])): r for r in src or ()}

    claims = keyed(claims_rows)
    item_by: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in item_rows or ():
        item_by[(r["arm"], int(r["layer"]))].append(r)
    out: dict[str, dict[str, Any]] = defaultdict(dict)
    for (a, la), rr in sorted(by.items()):
        nev = [r for r in rr if r["population"] == "never_refused"]
        cell: dict[str, Any] = {"index": _metric(rr), "never_refused": _metric(nev)}
        if claims_rows is not None:
            both = [
                (r, c)
                for r in nev
                if not r["n_failed_cells"]
                and (c := claims.get((a, la, r["item"]))) is not None
                and not c["n_failed_cells"]
            ]
            n_off = sum(r["n_contradicted"] for r, _ in both)
            n_cl = sum(int(c["n_claims"]) for _, c in both)
            cell["per_disposition_claim"] = {
                "n_items": len(both),
                "n_off": n_off,
                "n_disposition_claims": n_cl,
                "v": n_off / n_cl if n_cl else None,
            }
        if item_rows is not None:
            cell["item_metric"] = _metric(item_by.get((a, la), []))
        out[a][str(la)] = cell
    return dict(out)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export(out: Path, model: str) -> dict[str, Any]:
    """Write the ``offtopic_index/`` root and ``sj_offtopic_full.json`` from
    ``<out>/overrefusal/<model>/sj_records_full.jsonl``; returns the health block."""
    root = run.eval_root(out, model)
    src = root / RECORDS
    meta0 = json.loads((root / ITEM_META).read_text())  # plan 0020's meta: versions, models
    if meta0.get("unit") != "items":
        raise ValueError(f"{root / ITEM_META}: expected the plan-0020 export (unit items)")
    items = data.load_items()
    rows, fails = index_rows(data.read_jsonl(src), items)
    dest = root / OFFTOPIC_DIR
    write_rows(rows, dest / ITEM_COUNTS)
    n_dl = sum(r["n_claims"] for r in rows)
    health = {
        "n_records": len(rows),
        **fails,
        "n_failed_cells": sum(r["n_failed_cells"] for r in rows),
        "n_decline_lines": n_dl,
        "n_off_topic_lines": sum(r["n_contradicted"] for r in rows),
    }
    meta = {
        "eval": "overrefusal",
        "plan": "plans/0022-overrefusal-wrong-topic-refusals.md (index: hindex-v6)",
        "population": "all",
        "n_items": len(items),
        "unit": UNIT,
        "readers": sorted({r["arm"] for r in rows}),
        "layers": sorted({r["layer"] for r in rows}),
        "versions": meta0["versions"],
        "models": meta0.get("models"),
        "columns": (
            "n_claims = decline lines (orsj-v3), n_false = n_contradicted = off-topic decline "
            "lines (oroff-v1), n_unprobed = n_unresolved = 0, n_failed_cells = 1 for an unjudged "
            "/ strict-judge-failed / off-topic-failed item-layer (its counts 0), unit columns 0, "
            "cluster = item, population"
        ),
        "gates": {},  # no probe validation
        "source": {"records": str(src.resolve()), "sha256": _sha256(src)},
        "health": health,
    }
    (dest / ITEM_META).write_text(json.dumps(meta, indent=1) + "\n")
    claims = _read_rows(root / CLAIMS_COUNTS) if (root / CLAIMS_COUNTS).is_file() else None
    item_rows = _read_rows(root / ITEM_COUNTS)
    doc = {
        "plan": meta["plan"],
        "unit": UNIT,
        "versions": meta["versions"],
        "source": meta["source"],
        "health": health,
        "secondary": {
            "never_refused": "the same share on the 160 never_refused items only",
            "per_disposition_claim": "off-topic decline lines / orclaim-v1 disposition claims "
            f"(refusal + answer), never_refused, items valid in both ({CLAIMS_COUNTS})",
            "item_metric": f"the plan-0020 item metric (hindex-v4/v5 overrefusal; {ITEM_COUNTS})",
        },
        "metrics": summary(rows, claims, item_rows),
    }
    (root / SUMMARY).write_text(json.dumps(doc, indent=1) + "\n")
    return {"dest": str(dest), "summary": str(root / SUMMARY), **health}
