"""Plan 0029 Amendment 4: write the persistent-failures list `hbench final` excludes.

For every manifest entry, the failed cells (``final.scan``: the same recount `final` checks with)
are explained from the judge caches: each failed cell's judge calls (main / story: the J-lens
``summ:`` summary, ``:M`` map, ``:CV:<i>`` verifier verdicts; AM claims: ``:D`` / ``:Dsum`` /
``summ:`` detections, ``:M``, ``:CV:<i>``, and for a joint site its per-layer detections; AM
situation: ``sit:``; overrefusal: ``dg:`` digest, ``sj:`` strict judge, ``off:`` off-topic
judge). A call is failed when no row of its newest fingerprint has an answer; its attempts are
the rows of that fingerprint, its reason the newest row's:

* ``refused``  -- the call was refused (``usage.refusals`` > 0 with no answer: Opus, and the
  Sonnet fallback where the stage has one);
* ``invalid``  -- an invalid answer after the in-run retries (``n_invalid_attempts`` or a
  ``malformed`` / ``invalid`` reason);
* ``error``    -- a transport error / timeout with no model answer (never persistent: re-run);
* ``unknown``  -- anything else (never persistent).

A failed cell is explained by its **blocking stage**, read off its record: main / story the
first of summary (``stages.summary`` not ok) -> map (``stages.map`` not ok) -> cverify; AM claims
detection (``status`` pending / failed: the cell's, or for a joint site the 11 per-layer
``:D`` / ``:Dsum`` / ``summ:``) -> map (``map_status``) -> cverify; situation ``sit:``;
overrefusal the first unsettled of digest -> strict -> off-topic. A cell is listed (``cells``)
iff its blocking stage has at least one failed call, every failed call of that stage has reason
refused / invalid and at least ``final.PERSISTENT_MIN_ATTEMPTS`` attempts, and a single
required call (``:M``, ``sit:``, an overrefusal stage) has been asked at all (a never-asked call
is pending, not persistent). Each listed cell carries the sha256 of the entry file it was read
from (``file_sha256``; `final` refuses a list built from another file). Other failed cells go
to ``not_listed`` (with why) and count against `final`'s 1% allowance / the zero-unsettled-verdict
rule as before. An attempt is one cache row under the call's newest fingerprint: each run pass
writes one row per call after its in-run retries (and the Sonnet fallback), so 3 attempts = 3
passes.

usage (repo root): python scripts/v8/persistent_failures.py data/index/v8_manifest.json \\
    --or-joint-cache outputs/v8-full/safety/or/overrefusal/claude-opus-5-5 \\
    --dest data/index/v8_persistent_failures.json
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from hallucination_bench import final
from hallucination_bench import index as ix

KEY = re.compile(r'^\{"key": "((?:[^"\\]|\\.)*)"')
CELL = re.compile(r"^(?P<id>.+)__L(?P<la>\d{3}|all)__p(?P<pos>\d+)$")


def reason(row: dict[str, Any], batch: dict[str, Any] | None) -> str:
    meta = row.get("meta") or {}
    if meta.get("n_invalid_attempts") or meta.get("reason") in ("malformed", "invalid"):
        return "invalid"
    usage = meta.get("usage")
    if usage is None and batch is not None:  # AM batched detection rows: the batch row's usage
        bm = batch.get("meta") or {}
        if bm.get("n_invalid_attempts") or bm.get("reason") in ("malformed", "invalid"):
            return "invalid"
        usage = bm.get("usage")
    usage = usage or {}
    if int(usage.get("refusals") or 0) > 0:
        return "refused"
    if int(usage.get("errors") or 0) > 0 or int(usage.get("calls") or 0) == 0:
        return "error"
    return "unknown"


def scan_cache(path: Path, want: set[str], prefixes: tuple[str, ...]) -> dict[str, list[dict]]:
    """Rows of the wanted keys (exact, or starting with a prefix; plus the ``batchD:`` /
    ``batchH:`` rows they point at, in a second pass)."""
    rows: dict[str, list[dict]] = defaultdict(list)
    if not path.is_file() or not (want or prefixes):
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            m = KEY.match(line)
            if not m:
                continue
            k = m.group(1)
            if k in want or (prefixes and k.startswith(prefixes)):
                rows[k].append(json.loads(line))
    batches = {
        str((r.get("meta") or {}).get("batch"))
        for rs in rows.values()
        for r in rs
        if (r.get("meta") or {}).get("batch")
    }
    if batches:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                m = KEY.match(line)
                if m and m.group(1) in batches:
                    rows[m.group(1)].append(json.loads(line))
    return rows


def failed_call(key: str, rows: dict[str, list[dict]]) -> dict[str, Any] | None:
    """``None`` when the call is settled (an answer under its newest fingerprint) or absent."""
    rs = rows.get(key)
    if not rs:
        return None
    newest = max(rs, key=lambda r: float(r.get("ts") or 0))
    fp = newest.get("fingerprint")
    same = [r for r in rs if r.get("fingerprint") == fp]
    if any(r.get("result") is not None for r in same):
        return None
    batch_key = (newest.get("meta") or {}).get("batch")
    batch = None
    if batch_key and rows.get(batch_key):
        batch = max(rows[batch_key], key=lambda r: float(r.get("ts") or 0))
    meta = newest.get("meta") or {}
    return {
        "key": key,
        "attempts": len(same),
        "reason": reason(newest, batch),
        "stage": meta.get("stage") or key.split(":", 1)[0],
        "model": meta.get("model"),
        "answered_by": meta.get("answered_by"),
    }


def site_layers(cell: str) -> list[str]:
    m = CELL.match(cell)
    if not m:
        return []
    return [f"{m['id']}__L{la:03d}__p{m['pos']}" for la in final.EXPECTED_LAYERS]


def blocking(fam: str, cell: str, r: dict[str, Any]) -> tuple[str, set[str], tuple[str, ...], bool]:
    """The stage a failed claim-map / situation record is blocked at and its judge calls:
    ``(stage, keys, key prefixes, exact)``. ``exact`` = every key must have been asked (a
    single required call); otherwise at least one of the keys / prefixed keys must be a failed
    call (a set: the per-layer summaries or detections, the verifier verdicts)."""
    joint = "__Lall__" in cell
    per = site_layers(cell) if joint else [cell]
    if fam == ix.AM_SITUATION:
        return "situation", {f"sit:{cell}"}, (), True
    if fam in (ix.MAIN, ix.STORY):
        st = r.get("stages") or {}
        if st.get("summary") not in (None, "ok"):
            return "summary", {f"summ:{c}" for c in per}, (), False
        if st.get("map") != "ok":
            return "map", {f"{cell}:M"}, (), True
        return "cverify", set(), (f"{cell}:CV:",), False
    # AM claims
    if r.get("status") in ("pending", "failed"):
        det = {f"{k}{c}{t}" for c in per for k, t in (("", ":D"), ("", ":Dsum"), ("summ:", ""))}
        return "detect", det, (), False
    if r.get("map_status") not in (None, "ok"):
        return "map", {f"{cell}:M"}, (), True
    return "cverify", set(), (f"{cell}:CV:",), False


def cache_of(e: dict[str, Any], base: Path) -> Path:
    d = (base / e["root"] / e["file"]).parent
    return d / ("situation_cache.jsonl" if e["family"] == ix.AM_SITUATION else "cache.jsonl")


def or_calls(arm: str, item: str, unit: str) -> list[tuple[str, str]]:
    la = "all" if unit == ix.JOINT_LABEL else unit[1:]
    stages = (("digest", "dg"), ("sjudge", "sj"), ("offtopic", "off"))
    return [(d, f"{p}:{arm}:{item}:L{la}") for d, p in stages]


def failed_records(e: dict[str, Any], path: Path) -> dict[str, dict[str, Any]]:
    """The entry's failed records by cell id (``final.scan`` decides which are failed)."""
    _n, failed, _seen = final.scan(e, path)
    want = {c for c, _cv in failed}
    out: dict[str, dict[str, Any]] = {}
    unit, arm = final._layer(e["unit"]), final._arm(e)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if e["family"] in ix.SAFETY_FAMILIES_V8 and r.get("arm") != arm:
                continue
            if final._unit_layer(r["layer"]) != unit:
                continue
            c = final._cell_id(e, r)
            if c in want:
                out[c] = r
    return out


# AM detection can write two rows per pass (a batched call + its single re-ask): it needs twice
# the attempts so that a listed detection still means >= 3 passes
DETECT_ATTEMPT_FACTOR = 2


def decide_calls(
    rows: dict[str, list[dict]],
    keys: set[str],
    pre: tuple[str, ...],
    exact: bool,
    min_attempts: int | None = None,
) -> tuple[list[dict], str | None]:
    """The blocking stage's failed calls, or why the cell cannot be listed."""
    need = final.PERSISTENT_MIN_ATTEMPTS if min_attempts is None else min_attempts
    ks = sorted(keys | {k for k in rows if pre and k.startswith(pre)})
    if exact and any(not rows.get(k) for k in ks):
        return [], "the blocking call was never asked (no cache row)"
    calls = [x for x in (failed_call(k, rows) for k in ks) if x is not None]
    if not calls:
        return [], "no failed judge call of the blocking stage in the cache"
    if not all(c["reason"] in final.PERSISTENT_REASONS and c["attempts"] >= need for c in calls):
        return calls, "a failed call of the blocking stage is transient or under-retried"
    return calls, None


def build(manifest: Path, base: Path, or_joint_cache: Path | None) -> dict[str, Any]:
    m = final.load_manifest(manifest)
    groups: dict[Path, list[tuple[dict, dict[str, dict]]]] = defaultdict(list)
    or_groups: dict[Path, list[tuple[dict, dict[str, dict]]]] = defaultdict(list)
    for e in m["entries"]:
        path = (base / e["root"] / e["file"]).resolve()
        if e["family"] in (ix.SIBLING, ix.BENIGN_ROLEPLAY):
            if final.scan(e, path)[1]:
                print(f"note: {final._where(e)}: failed cells, no cache lookup", file=sys.stderr)
            continue
        recs = failed_records(e, path)
        if not recs:
            continue
        if e["family"] == ix.OVERREFUSAL:
            root = (base / e["root"]).resolve().parent
            if e["unit"] == ix.JOINT_LABEL and or_joint_cache is not None:
                root = or_joint_cache.resolve()
            or_groups[root].append((e, recs))
        else:
            groups[cache_of(e, base).resolve()].append((e, recs))
    listed: list[dict[str, Any]] = []
    not_listed: list[dict[str, Any]] = []

    def decide(e: dict, cell: str, stage: str, calls: list[dict], why: str | None) -> None:
        rec = {
            "family": e["family"],
            "method": e["method"],
            "unit": str(e["unit"]),
            "cell": cell,
            "stage": stage,
            "file_sha256": e["sha256"],
        }
        if calls:
            reasons = sorted({c["reason"] for c in calls})
            rec |= {
                "reason": "refused" if "refused" in reasons else reasons[0],
                "attempts": min(c["attempts"] for c in calls),
                "calls": calls,
            }
        if why is None:
            listed.append(rec)
        else:
            not_listed.append({**rec, "why": why})

    for cache, items in sorted(groups.items()):
        want: set[str] = set()
        pre: set[str] = set()
        plans = []
        for e, recs in items:
            for c, r in sorted(recs.items()):
                stage, keys, prefixes, exact = blocking(e["family"], c, r)
                want |= keys
                pre |= set(prefixes)
                plans.append((e, c, stage, keys, prefixes, exact))
        print(f"scanning {cache} for {len(plans)} failed cells", file=sys.stderr)
        rows = scan_cache(cache, want, tuple(sorted(pre)))
        for e, c, stage, keys, prefixes, exact in plans:
            need = final.PERSISTENT_MIN_ATTEMPTS
            if stage == "detect":
                need *= DETECT_ATTEMPT_FACTOR
            calls, why = decide_calls(rows, keys, prefixes, exact, need)
            decide(e, c, stage, calls, why)
    for root, items in sorted(or_groups.items()):
        by_file: dict[Path, set[str]] = defaultdict(set)
        for e, recs in items:
            arm = final._arm(e)
            for c in recs:
                for d, k in or_calls(arm, c, str(e["unit"])):
                    by_file[root / d / f"{arm}.jsonl"].add(k)
        rows: dict[str, list[dict]] = {}
        for f, ks in sorted(by_file.items()):
            print(f"scanning {f} for {len(ks)} calls", file=sys.stderr)
            rows.update(scan_cache(f, ks, ()))
        for e, recs in items:
            arm = final._arm(e)
            for c in sorted(recs):
                # the first unsettled stage in pipeline order (digest -> strict -> off-topic)
                # blocks the later ones; a never-asked stage is not persistent
                for d, k in or_calls(arm, c, str(e["unit"])):
                    if not rows.get(k) or failed_call(k, rows) is not None:
                        calls, why = decide_calls(rows, {k}, (), True)
                        decide(e, c, d, calls, why)
                        break
                else:
                    decide(e, c, "export", [], "every judge call is settled (re-export)")
    key = lambda r: (r["family"], r["method"], r["unit"], r["cell"])  # noqa: E731
    return {
        "version": final.PERSISTENT_VERSION,
        "min_attempts": final.PERSISTENT_MIN_ATTEMPTS,
        "n_listed": len(listed),
        "n_not_listed": len(not_listed),
        "cells": sorted(listed, key=key),
        "not_listed": sorted(not_listed, key=key),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--base", type=Path, default=Path.cwd())
    ap.add_argument("--or-joint-cache", type=Path, default=None)
    ap.add_argument("--dest", type=Path, required=True)
    a = ap.parse_args()
    doc = build(a.manifest, a.base, a.or_joint_cache)
    a.dest.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    counts: dict[tuple, int] = defaultdict(int)
    for c in doc["cells"]:
        counts[(c["family"], c["method"], c["reason"])] += 1
    for k, v in sorted(counts.items()):
        print(f"listed {k}: {v}")
    nl: dict[tuple, int] = defaultdict(int)
    for c in doc["not_listed"]:
        nl[(c["family"], c["method"], c.get("reason"), c["why"])] += 1
    for k, v in sorted(nl.items()):
        print(f"not listed {k}: {v}")
    print(f"wrote {a.dest}: {doc['n_listed']} listed, {doc['n_not_listed']} not listed")


if __name__ == "__main__":
    main()
