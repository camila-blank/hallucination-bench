"""Plan 0010 §10: import the plan-0009 state (``hbench gate import --from <worktree>``).

No API calls and no Modal launches: everything is read from the plan-0009 worktree (read-only)
and the local cells, and copied / derived into this worktree:

* ``data/sibling_gate/class_map.json`` (§10.2): rebuilt from the ``sibclass-v1`` cache (the raw
  class per (said, item), looked up by the exact fingerprint plan 0009 used) plus the latest
  non-null ``dedupe:all`` result through ``classes.apply_merges``; asserts that assembling over
  it reproduces ``classes.json`` row for row.
* ``data/sibling_gate/probe_bank/`` (§10.3): ``probe_texts.jsonl(.gz)`` split per class, each
  text with ``avoid: "legacy"`` and its capture record (label = ``probe_label(id)``, the shard
  ``acts-sibgate-probe-sNN`` whose ``cp_NN`` holds the label, the key from
  ``capture_prompts.json``); ``cells.json`` = every attempted cell (zero-text cells too).
* ``data/sibling_gate/distractors/<class>.json`` (§10.6): ``distractors.json`` split per class,
  with a sha per unit (generation 0).
* outputs: the fill / classes / texts caches, ``fill/rung1``, each pool's ``resample.jsonl`` as
  ``runs/run0.jsonl`` (+ ``state.json``: plan-0009 attempts), ``precheck.json``, the probe fits
  and scores, ``merge.json``, per-class ``inputs.sha256`` (§10.4, cross-checked against
  ``probe_inputs.json``) and score ``meta.json``, the plan-0009 labels (generation 0 stamps),
  ``calibration.json``, ``generation.json``; ``data/sibling_gate/unit_costs.json``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from hallucination_bench import arms as A
from hallucination_bench.cache import Cache, fingerprint

from . import classes as K
from . import fill_judge as F
from . import gate as G
from . import ladder as L
from . import resample as R
from . import texts as T
from . import units as U
from .stages import probe_label

# plan-0009 attempts per pool: s3d pools had the full run + the redo (redo_s3d.sh); a short key
# of these pools has exhausted its 2 attempts, so it stays resample_failed (no relaunch)
LEGACY_ATTEMPTS = {"s3d-rl600": 2, "s3d-sft251": 2}
BASELINE_PREFIX = "f8fd3f49969356ff"  # plan 0010 §10.1


def log(msg: str) -> None:
    print(f"[import {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def baseline_checksum(labels_dir: Path) -> str:
    """``sha256sum <dir>/*.jsonl | sort -k2 | sha256sum`` (absolute paths, as recorded)."""
    import hashlib

    lines = []
    for f in sorted(Path(labels_dir).glob("*.jsonl"), key=lambda p: str(p)):
        lines.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f}\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    elif src.exists():
        shutil.copy2(src, dst)


# ---------------------------------------------------------------- §10.2 class map


def build_class_map(src_out: Path, units, rung1, its, classes_json) -> dict[str, Any]:
    us = [u for u in units if F.needs_rung2(rung1.get(u["unit"]) or {})]
    reqs = K.class_requests(us)
    cache = Cache(src_out / "classes" / "cache.jsonl", readonly=True)
    raw: dict[tuple[str, str], dict[str, Any] | None] = {}
    for k, r in reqs.items():
        conv = F.conversation_for(its[k[1]], r["pos"])
        user = K.class_user(r["said"], r["segment"], F.window(conv, r.get("contradicted_by")))
        fp = fingerprint(K.CLASS_VERSION, F.MODEL, F.EFFORT, K.CLASS_SYSTEM, user)
        row = cache.get(f"class:{k[0]}:{k[1]}", fp)
        raw[k] = row["result"] if row is not None else None
    names = sorted({r["class"] for r in raw.values() if r} - {K.NO_CLASS, ""})
    dd_row = K.latest_dedupe_all(cache)
    if dd_row is None:
        raise SystemExit("no non-null dedupe:all row in the classes cache")
    user = "CLASS NAMES:\n" + "\n".join(f"- {n}" for n in names)
    fp = fingerprint(K.DEDUPE_VERSION, F.MODEL, F.EFFORT, K.DEDUPE_SYSTEM, user)
    frozen_names = names if dd_row["fingerprint"] == fp else []
    if not frozen_names:  # §10.2: an answer for another name list is not frozen
        log("WARNING: dedupe:all fingerprint differs from the current names: names not frozen")
    mp = K.apply_merges(names, dd_row["result"]["mapping"])
    classes = K.assemble(raw, mp)
    if classes != classes_json["classes"] or mp != classes_json["dedupe"]:
        bad = [k for k in classes_json["classes"] if classes.get(k) != classes_json["classes"][k]]
        raise SystemExit(f"class map does not reproduce classes.json: {len(bad)} rows, {bad[:3]}")
    log(
        f"class map: {len(raw)} pairs ({sum(v is None for v in raw.values())} failed, frozen as "
        f"failed), {len(frozen_names)} raw names; assemble == classes.json row for row"
    )
    return {
        "version": K.CLASS_MAP_VERSION,
        "class_version": K.CLASS_VERSION,
        "dedupe_version": K.DEDUPE_VERSION,
        "source": "plan 0009 sibclass-v1 cache + dedupe:all (plan 0010 §10.2)",
        "names": {n: mp[n] for n in frozen_names},
        "pairs": {f"{k[0]}|{k[1]}": v for k, v in raw.items()},
    }


# ---------------------------------------------------------------- §10.3 bank


def build_bank(src_out: Path, data_dir: Path, concepts: dict[str, Any]) -> dict[str, list]:
    src_probes = src_out / "probes"
    texts_path = data_dir / "probe_texts.jsonl.gz"
    with gzip.open(texts_path, "rt", encoding="utf-8") as fh:
        texts = [json.loads(x) for x in fh if x.strip()]
    cap = {r["label"]: r for r in json.loads((src_probes / "capture_prompts.json").read_text())}
    shard_of: dict[str, str] = {}
    dup = Counter()
    for f in sorted((src_probes / "shards").glob("cp_*.json")):
        nn = f.stem.split("_", 1)[1]
        for r in json.loads(f.read_text()):
            dup[r["label"]] += 1
            shard_of[r["label"]] = f"acts-sibgate-probe-s{nn}"
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    bad = []
    for t in texts:
        lab = probe_label(t["id"])
        if dup[lab] != 1 or lab not in cap:
            bad.append(lab)
            continue
        by[t["class"]].append(
            {
                **t,
                "avoid": T.LEGACY,
                "capture": {"label": lab, "shard": shard_of[lab], "key": cap[lab]["meta"]["key"]},
            }
        )
    if bad:
        raise SystemExit(f"{len(bad)} labels not in exactly one cp_NN / capture_prompts: {bad[:3]}")
    idx: dict[str, list[dict[str, Any]]] = {}
    for cls, row in concepts.items():
        n = Counter((t["concept"], t["kind"]) for t in by.get(cls, []))
        idx[cls] = [
            {
                "concept": c,
                "kind": k,
                "n": n.get((c, k), 0),
                "avoid": T.LEGACY,
                "version": T.GEN_VERSION,
            }
            for c, k in T.wanted_cells(row["concepts"])
        ]
        T.write_bank_class(data_dir, L.slug(cls), by.get(cls, []))
    T.save_cells_index(data_dir, idx)
    log(
        f"bank: {len(texts)} texts in {len(by)} classes; {sum(len(v) for v in idx.values())} "
        f"attempted cells ({sum(e['n'] == 0 for v in idx.values() for e in v)} kept no text)"
    )
    return by


# ---------------------------------------------------------------- §10.4 fit hashes


def fit_hashes(src_out: Path, out: Path, concepts, bank) -> dict[str, str]:
    pin = json.loads((src_out / "probes" / "probe_inputs.json").read_text())
    merges = json.loads((out / "probes" / "merge.json").read_text())
    res = {}
    for cls, row in sorted(concepts.items()):
        slug = L.slug(cls)
        rows = bank.get(cls, [])
        sl = G.fit_slice(cls, row, rows, merges.get(cls, list(range(T.N_CLASSES))))
        ref = pin["classes"].get(slug)
        if ref is None:
            raise SystemExit(f"{cls}: not in probe_inputs.json")
        if sl != ref:
            diff = [k for k in sl if sl[k] != ref.get(k)]
            raise SystemExit(f"{cls}: bank-derived fit inputs != probe_inputs.json ({diff})")
        shards = G.class_shards(rows)
        h = G.fit_input_hash(sl, rows, shards)
        hp = out / "probes" / "fit" / slug / "inputs.sha256"
        hp.parent.mkdir(parents=True, exist_ok=True)
        hp.write_text(h + "\n")
        res[cls] = h
    log(f"fit hashes: {len(res)} classes; texts / folds / merge / concepts == probe_inputs.json")
    return res


# ---------------------------------------------------------------- main


def cmd_import(a: argparse.Namespace) -> int:
    if not a.src:
        raise SystemExit("import needs --from <plan-0009 worktree>")
    src = Path(a.src)
    src_out = src / "outputs" / "sibling_gate"
    src_data = src / "data" / "sibling_gate"
    out, data = Path(a.out), Path(a.data_dir)
    reg = A.load()
    t0 = time.time()
    # the data files plan 0009 committed must be the ones in this worktree
    for f in ("classes.json", "concepts.json", "distractors.json", "probe_texts.jsonl.gz"):
        if G.sha256_file(src_data / f) != G.sha256_file(data / f):
            raise SystemExit(f"data/sibling_gate/{f} differs from {src_data / f}")
    base = baseline_checksum(src_out / "labels")
    log(f"baseline labels checksum {base[:16]} (plan §10.1: {BASELINE_PREFIX})")
    if not base.startswith(BASELINE_PREFIX):
        raise SystemExit("the plan-0009 labels changed since plan 0010 §10.1 fixed the baseline")

    log("copying caches, rung 1, probes (fit / score / merge), labels")
    for sub in ("fill/cache.jsonl", "fill/rung1", "classes/cache.jsonl", "texts/cache.jsonl"):
        copy(src_out / sub, out / sub)
    for sub in ("fit", "score", "merge.json"):
        copy(src_out / "probes" / sub, out / "probes" / sub)
    copy(src_out / "labels", out / "labels")

    # units: built from the cells (cells root), must equal the plan-0009 units
    gate = G.Gate(reg=reg, out=out, data=data, cells_root=Path(a.cells_root), quiet=True)
    units, uh = gate.load_units()
    for arm in reg["gated_arms"]:
        old = {r["unit"] for r in U.iter_jsonl(src_out / "units" / f"{arm}.jsonl")}
        new = {u["unit"] for u in units if u["arm"] == arm}
        if old != new:
            raise SystemExit(
                f"[{arm}] units from the cells != plan-0009 units ({len(new)} vs {len(old)})"
            )
    log(f"units: {len(units)} over {len(uh)} arms == plan-0009 units; unit hashes frozen")

    # §10.5 resample: run0 + state
    for pool in reg["pools"]:
        sp = src_out / "resample" / pool
        if not (sp / "resample.jsonl").exists():
            continue
        dp = R.pool_dir(pool, out)
        copy(sp / "resample.jsonl", R.runs_dir(pool, out) / f"{R.RUN0}.jsonl")
        req = json.loads((sp / "request.json").read_text())
        rows = {
            (r["id"], int(r["layer"]), int(r["pos"])) for r in U.iter_jsonl(sp / "resample.jsonl")
        }
        st = R.PoolState(pool)
        att = LEGACY_ATTEMPTS.get(pool, 1)
        for k in req["cells"]:
            st.attempts[st.kstr(tuple(k))] = att
        st.runs[R.RUN0] = {
            "status": "collected",
            "legacy": True,
            "n_requested": len(req["cells"]),
            "n_rows": len(rows),
            "legacy_runs": "gen-sibgate-<pool> (+ -s0..-s9 / -precheck / s3d redo)",
        }
        R.save_state(st, out)
        R.rebuild_resample(pool, out)
        if (dp / "resample.jsonl").read_bytes() != (sp / "resample.jsonl").read_bytes():
            raise SystemExit(f"{pool}: run0 merge does not reproduce resample.jsonl")
        pc = sp / "precheck" / "collect_stats.json"
        G.write_json(
            dp / "precheck.json",
            {
                "ok": True,
                "legacy": True,
                "stats": json.loads(pc.read_text()) if pc.exists() else None,
            },
        )
        log(
            f"resample {pool}: run0 {len(rows)} keys ({len(req['cells']) - len(rows)} short, "
            f"attempts {att})"
        )

    # §10.2 class map
    its = G.items()
    rung1 = gate.load_rung1()
    classes_json = json.loads((data / "classes.json").read_text())
    cm = build_class_map(src_out, units, rung1, its, classes_json)
    K.save_class_map(cm, data)

    # §10.3 bank + §10.4 fit hashes
    concepts = json.loads((data / "concepts.json").read_text())["classes"]
    bank = build_bank(src_out, data, concepts)
    fh = fit_hashes(src_out, out, concepts, bank)

    # §10.6 distractors per class + score metas
    dis = json.loads((data / "distractors.json").read_text())
    by: dict[str, dict[str, Any]] = defaultdict(dict)
    cls_of = {u["unit"]: classes_json["classes"].get(f"{u['said_norm']}|{u['id']}") for u in units}
    for unit, rec in dis.items():
        v = cls_of.get(unit)
        if v is None:
            raise SystemExit(f"distractor unit {unit} has no class")
        by[L.slug(v["class"])][unit] = {**rec, "sha": G.distractor_sha(rec), "generation": 0}
    ddir = data / "distractors"
    if ddir.exists():
        shutil.rmtree(ddir)
    for slug, recs in by.items():
        G.write_json(ddir / f"{slug}.json", dict(sorted(recs.items())), indent=0)
    sinp = json.loads((src_out / "probes" / "score_inputs.json").read_text())
    if sinp.get("distractors_sha256") != G.sha256_file(data / "distractors.json"):
        raise SystemExit("score_inputs.json was not scored after this distractors.json")
    es = G.eval_sites_hash(its)
    keys = {
        it: {str(s["pos"]): T.token_class(s.get("token") or "") for s in its[it]["sites"]}
        for it in its
    }
    if sinp["site_keys"] != keys or sinp["items"] != sorted(its):
        raise SystemExit("score_inputs.json eval sites differ from the bank's")
    n_meta = 0
    for cls in concepts:
        slug = L.slug(cls)
        fd = out / "probes" / "fit" / slug
        fits = [json.loads(f.read_text()) for f in sorted(fd.glob("fit_L*.json"))]
        scored = (out / "probes" / "score" / slug).exists()
        if scored != (bool(fits) and not all(f.get("error") for f in fits)):
            raise SystemExit(f"{cls}: score dir presence disagrees with its fit errors")
        G.write_json(
            out / "probes" / "score" / slug / "meta.json",
            {
                "class": cls,
                "input_hash": fh[cls],
                "eval_sites": es,
                "units": {k: v["sha"] for k, v in sorted(by.get(slug, {}).items())},
                "scored": scored,
                "generation": 0,
                "imported_from": str(src_out / "probes" / "score"),
                "ts": time.time(),
            },
        )
        n_meta += 1
    log(f"distractors: {len(dis)} units in {len(by)} class files; score meta for {n_meta} classes")

    # calibration (§4, not re-run by the gate) and unit costs (§6)
    calib = calibration(data, rung1)
    G.write_json(out / "calibration.json", calib)
    G.write_json(
        data / "unit_costs.json", unit_costs(src_out, units, rung1, classes_json, concepts, bank)
    )

    # generation 0 = plan 0009: stamps on the imported labels
    gate = G.Gate(reg=reg, out=out, data=data, cells_root=Path(a.cells_root), quiet=True)
    imported = [r for f in sorted((out / "labels").glob("*.jsonl")) for r in U.iter_jsonl(f)]
    st = gate.stamp(uh, concepts, fh, imported)
    for arm in reg["gated_arms"]:
        G.write_json(
            U.stamp_path(arm, out / "labels"),
            {"generation": 0, "unit_hash": uh[arm], "stamp_sha": G.sha256_json(st)},
        )
    classes = classes_json["classes"]
    gj = {
        "generation": 0,
        "imported_from": str(src),
        "baseline_labels_sha256": base,
        "stamp": st,
        "stamp_sha": G.sha256_json(st),
        "code_revision": G.code_revision(),
        "spend": {"llm_usd": 0.0, "llm_calls": 0, "modal_launches": 0},
        "pending": {},
        "cross_digest": gate.cross_digest(units, classes),
        "fit_hashes": fh,
        "ts": time.time(),
    }
    G.write_json(out / "generation.json", gj)
    log(f"done in {time.time() - t0:.0f}s: generation 0 (plan 0009) imported")
    return 0


def unit_costs(src_out: Path, units, rung1, classes_json, concepts, bank) -> dict[str, Any]:
    """Plan-0009 unit costs (from the cache usage) and the rates the --plan estimate uses."""

    def mean_usd(path: Path, prefix: str) -> tuple[float, int]:
        tot, n = 0.0, 0
        for r in U.iter_jsonl(path):
            if r["key"].startswith(prefix) and r.get("result") is not None:
                tot += float(((r.get("meta") or {}).get("usage") or {}).get("usd") or 0.0)
                n += 1
        return (tot / n if n else 0.0), n

    fill, n_fill = mean_usd(src_out / "fill" / "cache.jsonl", "fill:")
    cls, _ = mean_usd(src_out / "classes" / "cache.jsonl", "class:")
    con, _ = mean_usd(src_out / "classes" / "cache.jsonl", "concepts:")
    ded, _ = mean_usd(src_out / "classes" / "cache.jsonl", "dedupe:")
    gen_tot = 0.0
    for r in U.iter_jsonl(src_out / "texts" / "cache.jsonl"):
        gen_tot += float(((r.get("meta") or {}).get("usage") or {}).get("usd") or 0.0)
    n_cells = sum(len(T.wanted_cells(row["concepts"])) for row in concepts.values())
    n_texts = sum(len(v) for v in bank.values())
    n_r2 = sum(1 for u in units if F.needs_rung2(rung1.get(u["unit"]) or {}))
    n_pairs = len(classes_json["classes"])
    n_concepts = sum(len(r["concepts"]) for r in concepts.values())
    return {
        "_doc": "plan 0010 §6: plan-0009 unit costs (cache usage; Modal from "
        "outputs/sibling_gate/modal_estimate.json list prices) and rates for the --plan estimate",
        "fill_usd_per_call": round(fill, 5),
        "class_usd_per_call": round(cls, 5),
        "concepts_usd_per_call": round(con, 5),
        "dedupe_usd_per_call": round(ded, 5),
        "gen_usd_per_cell": round(gen_tot / max(1, n_cells), 5),
        "texts_per_cell": round(n_texts / max(1, n_cells), 2),
        "capture_usd_per_prompt": round(80.0 / 187978, 6),
        "fit_usd_per_class": round(308.0 / 223, 3),
        "score_usd_per_class": round(5.0 / 223, 4),
        "precheck_usd": 10.0,
        "rung2_share": round(n_r2 / max(1, len(units)), 4),
        "new_pairs_per_rung2_unit": round(n_pairs / max(1, n_r2), 4),
        "new_concepts_per_new_pair": round(n_concepts / max(1, n_pairs), 4),
        "crossing_classes_per_new_pair": round(len(concepts) / max(1, n_pairs), 4),
        "n_classes": len(concepts),
        "n_fill_calls": n_fill,
    }


def calibration(data: Path, rung1: dict[str, Any]) -> dict[str, Any]:
    """The fill judge's plan-0009 calibration (dev: ``fill_dev.jsonl`` blocks + the todo's
    said / actual; gold: ``fill_gold.jsonl`` + ``fill_gold_labels.jsonl``), with file hashes."""
    out: dict[str, Any] = {"prompt_version": F.PROMPT_VERSION}
    pairs = {
        "dev": ("fill_dev_todo.jsonl", "fill_dev.jsonl", "blocks"),
        "gold": ("fill_gold.jsonl", "fill_gold_labels.jsonl", "answers"),
    }
    for split, (meta_f, lab_f, field) in pairs.items():
        mp, lp = data / meta_f, data / lab_f
        if not (mp.exists() and lp.exists()):
            continue
        meta = {r["unit"]: r for r in U.iter_jsonl(mp)}
        gold = {}
        for r in U.iter_jsonl(lp):
            if r.get(field) and r["unit"] in meta:
                gold[r["unit"]] = {**meta[r["unit"]], "answers": r[field]}
        out[split] = {
            "sha256": {meta_f: G.sha256_file(mp), lab_f: G.sha256_file(lp)},
            "agreement": F.agreement(rung1, gold),
        }
    return out
