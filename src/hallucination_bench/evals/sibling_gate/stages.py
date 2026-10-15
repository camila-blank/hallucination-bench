"""``hbench sibling-gate <stage>`` (plan 0009 §5-7), in driver order:

units -> resample-manifests -> (GPU resample jobs) -> resample-collect -> fill -> fill-calib ->
classes -> concepts -> probe-texts -> capture-manifest -> (GPU capture) -> fit -> score ->
label -> report -> figures.

Local stages read and write ``outputs/sibling_gate/`` and ``data/sibling_gate/``. LLM stages
(fill, classes, concepts, probe-texts) go through ``llm.LLM`` with the append-only caches; with
``--no-calls`` they replay the cache only. GPU / Modal stages (the resample jobs, the capture,
fit and score) are never launched from here unless ``--launch`` is given: the stage writes its
inputs and prints the command ``scripts/sibling_gate/run_all.sh`` runs.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from hallucination_bench import DATA, REPO_ROOT

from . import units as U

DATA_DIR = DATA / "sibling_gate"
PROBES_DIR = U.OUT / "probes"
SCRIPTS = REPO_ROOT / "scripts" / "sibling_gate"
MODAL = "uvx --python 3.12 modal@latest"


def probe_label(text_id: str) -> str:
    """The capture label of a probe text (also its acts file / dir name): ``/`` and ``\\`` in a
    concept name (``Oracle Smart View/...``) made ``capture_acts`` write into a sub-directory
    and crash (rev 8.2); labels stay unique (checked on the full bank)."""
    return text_id.replace("|", "__").replace(" ", "_").replace("/", "_").replace("\\", "_")


def _configure(a: argparse.Namespace) -> None:
    """A non-default ``--out`` (the pilot) gets its own data / probes dirs and probe-acts subdir
    instead of ``data/sibling_gate`` and ``<default out>/probes``; the default is unchanged."""
    global DATA_DIR, PROBES_DIR
    out = Path(a.out)
    if out.resolve() != U.OUT.resolve():
        DATA_DIR, PROBES_DIR = out / "data", out / "probes"


def _arms(a: argparse.Namespace) -> list[str]:
    return [x for x in a.arms.split(",") if x] if a.arms else list(U.ARMS_IN_SCOPE)


def _items() -> dict[str, dict[str, Any]]:
    from hallucination_bench.evals.wsbench_chat import bank

    return {it["id"]: it for it in bank.load_items()}


def _llm(a: argparse.Namespace):
    if getattr(a, "no_calls", False):
        return None
    from hallucination_bench.llm import LLM

    return LLM(rpm=a.rpm, concurrency=a.concurrency)


def _all_units(a: argparse.Namespace) -> list[dict[str, Any]]:
    us = U.load_units(_arms(a), Path(a.out))
    if not us:
        raise SystemExit(f"no units under {Path(a.out) / 'units'}: run `sibling-gate units` first")
    return us


# ---------------------------------------------------------------- units


def cmd_units(a: argparse.Namespace) -> int:
    from . import resample as R

    out = Path(a.out)
    root = Path(a.cells_root)
    allu: list[dict[str, Any]] = []
    per_arm: dict[str, dict[str, int]] = {}
    for arm in _arms(a):
        p = root / arm / "cells.jsonl"
        if not p.exists():
            print(f"[{arm}] no cells.jsonl under {root / arm} (skipped)")
            continue
        us, c = U.build_units(arm, p)
        U.write_jsonl(U.units_path(arm, out), us)
        per_arm[arm] = {
            "n_units": len(us),
            "n_segments": c.get("n_segments", 0),
            "n_cells": c.get("n_cells", 0),
        }
        allu.extend(us)
    reqs = R.requests(allu)
    pools = {p: r.counts() for p, r in reqs.items()}
    det = [u for u in allu if u["arm"] in U.DETERMINISTIC_ARMS]
    names = U.item_sibling_names(allu)
    summary = {
        "cells_root": str(root),
        "per_arm": per_arm,
        "pools": pools,
        "deterministic": {
            "n_units": len(det),
            "n_cells": len({(u["arm"], u["cell"]) for u in det}),
        },
        "n_items_with_siblings": len(names),
        "total": {"n_units": len(allu), "n_segments": sum(len(u["segments"]) for u in allu)},
    }
    (out / "units").mkdir(parents=True, exist_ok=True)
    (out / "units" / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"{'arm':22s} {'units':>7s} {'segments':>9s} {'cells':>7s}  pool")
    for arm, c in per_arm.items():
        print(
            f"{arm:22s} {c['n_units']:7d} {c['n_segments']:9d} {c['n_cells']:7d}  "
            f"{R.pool_of(arm) or '(deterministic)'}"
        )
    print(f"\n{'pool':16s} {'units':>7s} {'req. cells':>11s} {'prompts':>8s}  layers")
    for p, c in pools.items():
        print(
            f"{p:16s} {c['n_units']:7d} {c['n_requested_cells']:11d} {c['n_prompts']:8d}  "
            f"{c['layers']}"
        )
    print(f"deterministic (token arms): {summary['deterministic']}")
    print(f"total: {summary['total']}; wrote {out / 'units'}")
    return 0


# ---------------------------------------------------------------- resample


def cmd_resample_manifests(a: argparse.Namespace) -> int:
    from . import resample as R

    reqs = R.requests(_all_units(a))
    if a.dry:
        tot_r = tot_g = 0
        print(f"{'pool':16s} {'requested':>10s} {'generated':>10s} {'gen. samples':>13s}  layers")
        for p, r in reqs.items():
            c = r.counts()
            tot_r += c["n_requested_cells"]
            tot_g += c["n_generated_cells"]
            print(
                f"{p:16s} {c['n_requested_cells']:10d} {c['n_generated_cells']:10d} "
                f"{c['n_generated_samples']:13d}  {c['layers']}"
            )
        print(f"{'total':16s} {tot_r:10d} {tot_g:10d} {tot_g * R.N_RESAMPLE:13d}")
        return 0
    mans = {
        R.ACTS_11: json.loads(Path(a.manifest_11).read_text()),
        R.ACTS_L42: json.loads(Path(a.manifest_l42).read_text()),
    }
    res = R.write_manifests(reqs, mans, Path(a.out), limit_cells=a.precheck or None)
    for p, r in res.items():
        print(
            f"{p}: {r['n_requested_cells']} requested / {r['n_generated_cells']} generated; "
            f"manifest errors {r['manifest_errors'] or 'none'}; {r['dir']}/jobs.sh"
        )
    return 1 if any(r["manifest_errors"] for r in res.values()) else 0


def cmd_resample_collect(a: argparse.Namespace) -> int:
    from . import resample as R

    pool = R.POOLS[a.pool]
    d = R.pool_dir(pool.name, Path(a.out)) / ("precheck" if a.precheck else "")
    req = json.loads((d / "request.json").read_text())
    want = [tuple(x) for x in req["cells"]]
    cfg = json.loads(Path(a.run_config).read_text()) if a.run_config else {"pool": pool.name}
    rows, stats = R.collect(R.iter_gen_rows(Path(a.src)), want, cfg)
    U.write_jsonl(d / "resample.jsonl", rows)
    if a.orig_readouts:  # step 0: block shape vs the originals on 20 cells
        orig = {
            (r["id"], int(r["layer"]), int(r["pos"])): r["samples"]
            for r in U.iter_jsonl(Path(a.orig_readouts))
        }
        keys = [k for k in sorted({(r["id"], r["layer"], r["pos"]) for r in rows}) if k in orig]
        keys = keys[: a.shape_cells]
        new = {(r["id"], r["layer"], r["pos"]): r["samples"] for r in rows}
        stats["block_shape"] = R.block_shape_check(
            [s for k in keys for s in orig[k]], [s for k in keys for s in new[k]]
        )
        stats["block_shape"]["n_cells"] = len(keys)
    (d / "collect_stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats, indent=1))
    ok = stats["keys_equal_request"] and stats.get("block_shape", {"ok": True})["ok"]
    return 0 if ok else 1


# ---------------------------------------------------------------- fill (rung 1)


def cmd_fill(a: argparse.Namespace) -> int:
    from hallucination_bench.cache import Cache

    from . import fill_judge as F
    from . import resample as R

    out = Path(a.out)
    units = _all_units(a)
    resampled = {p: R.load_resample(p, out) for p in R.POOLS}
    failed = frozenset(x for x in (a.failed_pools or "").split(",") if x)
    llm = _llm(a)

    async def go():
        with Cache(out / "fill" / "cache.jsonl", readonly=llm is None) as cache:
            try:
                return await F.run_fill(units, resampled, _items(), llm, cache, failed_pools=failed)
            finally:
                if llm is not None:
                    await llm.aclose()

    rows, spend = asyncio.run(go())
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[r["arm"]].append(r)
    for arm, rs in by.items():
        U.write_jsonl(F.rung1_path(arm, out), rs)
    c = Counter((r["arm"], r["rung1"]) for r in rows)
    for arm in sorted(by):
        print(arm, {k: v for (a2, k), v in c.items() if a2 == arm})
    print(
        f"failed quotes: {sum(r.get('n_failed_quotes') or 0 for r in rows)}; spend {spend.report()}"
    )
    return 0


def _rung1(out: Path, arms: list[str]) -> dict[str, dict[str, Any]]:
    from . import fill_judge as F

    rows: dict[str, dict[str, Any]] = {}
    for arm in arms:
        p = F.rung1_path(arm, out)
        if p.exists():
            rows.update({r["unit"]: r for r in U.iter_jsonl(p)})
    return rows


def cmd_fill_calib(a: argparse.Namespace) -> int:
    """``--make``: a seeded sample of units with their blocks (no judge output) for a blind
    labeller -> ``data/sibling_gate/fill_<split>.jsonl``; otherwise score the labelled file."""
    import random

    from . import fill_judge as F
    from . import resample as R

    out = Path(a.out)
    path = DATA_DIR / f"fill_{a.split}.jsonl"
    if a.make:
        units = [u for u in _all_units(a) if u["arm"] not in U.DETERMINISTIC_ARMS]
        resampled = {p: R.load_resample(p, out) for p in R.POOLS}
        units = [
            u
            for u in units
            if (resampled.get(R.pool_of(u["arm"])) or {}).get((u["id"], u["layer"], u["pos"]))
        ]
        rng = random.Random(a.seed)
        pick = rng.sample(units, min(a.n, len(units)))
        rows = []
        for u in pick:
            blocks = resampled[R.pool_of(u["arm"])][(u["id"], u["layer"], u["pos"])]
            rows.append(
                {
                    "unit": u["unit"],
                    "arm": u["arm"],
                    "said": u["said"],
                    "actual": u["actual"],
                    "segment": u["segments"][0]["text"],
                    "contradicted_by": u["segments"][0]["contradicted_by"],
                    "blocks": blocks,
                    "answers": None,
                }
            )
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        U.write_jsonl(path, rows)
        print(f"wrote {len(rows)} unlabelled units to {path}")
        return 0
    gold = {r["unit"]: r for r in U.iter_jsonl(path) if r.get("answers")}
    res = F.agreement(_rung1(out, _arms(a)), gold)
    print(json.dumps(res, indent=1))
    return 0


# ---------------------------------------------------------------- classes / concepts


def cmd_classes(a: argparse.Namespace) -> int:
    from hallucination_bench.cache import Cache

    from . import classes as K
    from . import fill_judge as F

    out = Path(a.out)
    rung1 = _rung1(out, _arms(a))
    units = [u for u in _all_units(a) if F.needs_rung2(rung1.get(u["unit"]) or {})]
    reqs = K.class_requests(units)
    llm = _llm(a)

    async def go():
        with Cache(out / "classes" / "cache.jsonl", readonly=llm is None) as cache:
            try:
                raw, spend = await K.run_classes(reqs, _items(), llm, cache)
                dd = await K.run_dedupe([r["class"] for r in raw.values() if r], llm, cache)
                return raw, dd, spend
            finally:
                if llm is not None:
                    await llm.aclose()

    raw, dd, spend = asyncio.run(go())
    classes = K.assemble(raw, dd)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "classes.json").write_text(
        json.dumps(
            {"version": K.CLASS_VERSION, "dedupe": dd, "classes": classes},
            indent=1,
            ensure_ascii=False,
        )
        + "\n"
    )
    c = Counter(v["class"] for v in classes.values())
    print(
        f"{len(classes)} (said, item) pairs, {len(c)} classes; no_class {c.get(K.NO_CLASS, 0)};"
        f" no_gold {sum(v['no_gold'] for v in classes.values())}; spend {spend.report()}"
    )
    return 0


def _classes() -> dict[str, dict[str, Any]]:
    return json.loads((DATA_DIR / "classes.json").read_text())["classes"]


def _concepts() -> dict[str, dict[str, Any]]:
    return json.loads((DATA_DIR / "concepts.json").read_text())["classes"]


def cmd_concepts(a: argparse.Namespace) -> int:
    from hallucination_bench.cache import Cache

    from . import classes as K

    out = Path(a.out)
    classes = _classes()
    npairs = Counter(v["class"] for v in classes.values())
    kept = {k: v for k, v in classes.items() if npairs[v["class"]] >= a.min_class_pairs}
    llm = _llm(a)

    async def go():
        with Cache(out / "classes" / "cache.jsonl", readonly=llm is None) as cache:
            try:
                return await K.run_concepts(K.class_members(kept), llm, cache)
            finally:
                if llm is not None:
                    await llm.aclose()

    concepts = asyncio.run(go())
    comp = K.compatible_of(classes)
    (DATA_DIR / "concepts.json").write_text(
        json.dumps(
            {
                "version": K.CONCEPTS_VERSION,
                "classes": concepts,
                "compatible": {g: sorted(v) for g, v in comp.items()},
            },
            indent=1,
            ensure_ascii=False,
        )
        + "\n"
    )
    small = [c for c, r in concepts.items() if len(r["concepts"]) < K.MIN_CLASS_SIZE]
    print(
        f"{len(concepts)} classes; {sum(len(r['concepts']) for r in concepts.values())} "
        f"concepts; below {K.MIN_CLASS_SIZE}: {len(small)}"
    )
    return 0


# ---------------------------------------------------------------- probe texts + capture


def _surface(units: list[dict[str, Any]], classes: dict[str, dict[str, Any]]):
    """class -> concept -> the raw said strings the lenses used for it (aliases)."""
    out: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for u in units:
        v = classes.get(f"{u['said_norm']}|{u['id']}")
        if v and v["class"] != "no_class":
            out[v["class"]][v["said_canonical"]].add(u["said"])
    return {k: {c: sorted(s) for c, s in v.items()} for k, v in out.items()}


def cmd_probe_texts(a: argparse.Namespace) -> int:
    from hallucination_bench.cache import Cache

    from . import texts as T

    out = Path(a.out)
    concepts = _concepts()
    surface = _surface(_all_units(a), _classes())
    index = T.item_ngram_index(_items().values())
    llm = _llm(a)

    async def go():
        with Cache(out / "texts" / "cache.jsonl", readonly=llm is None) as cache:
            try:
                return await T.run_texts(concepts, surface, index, llm, cache)
            finally:
                if llm is not None:
                    await llm.aclose()

    kept, causes, spend = asyncio.run(go())
    U.write_jsonl(DATA_DIR / "probe_texts.jsonl", kept)
    print(f"{len(kept)} probe texts; drops {dict(causes)}; spend {spend.report()}")
    return 0


def _site_share(path: Path) -> tuple[Counter, int]:
    """Raw token class -> number of cells in a readouts ``hallucination.jsonl`` (the eval sites
    over all cells, e.g. the 12,353 of s3d-rl600-all11)."""
    from . import texts as T

    n: Counter = Counter()
    tot = 0
    for r in U.iter_jsonl(path):
        n[T.token_class(r.get("token") or "")] += 1
        tot += 1
    return n, tot


def cmd_capture_manifest(a: argparse.Namespace) -> int:
    """Prompts JSON for ``wsbench_capture_modal.py --prompts-json`` (one bank,
    ``acts-sibgate-probe``): the assistant reply is the prefill, and ``eval_positions`` are the
    reply tokens of a key class after the first mention (<= 32, <= 8 per key, the caps applied
    after the per-class merge, :func:`texts.class_merge`). Writes ``merge.json`` (class -> key
    per raw token class, fixed across layers) and prints per key the training positions, the
    implied null positions and the eval-site share. Needs a tokenizer
    (``uv run --with transformers``)."""
    from . import texts as T

    try:
        from transformers import AutoTokenizer  # type: ignore[import-not-found]
    except ImportError:
        raise SystemExit(
            "capture-manifest needs transformers: uv run --with transformers --with jinja2 "
            "hbench sibling-gate capture-manifest"
        ) from None
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    enc_rows = []
    for t in U.iter_jsonl(DATA_DIR / "probe_texts.jsonl"):
        msgs = [
            {"role": "system", "content": t["system"]},
            {"role": "user", "content": t["user"]},
            {"role": "assistant", "content": t["assistant"]},
        ]
        text = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
        reply_at = text.rfind(t["assistant"])
        # the capture prefills the reply: no trailing `<|im_end|>\n` (its "\n" would be a read
        # position past the last captured token)
        text = text[: reply_at + len(t["assistant"])]
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        toks = [text[s:e] for s, e in enc["offset_mapping"]]
        starts = [s for s, _e in enc["offset_mapping"]]
        lo = reply_at + int(t["mention_char"] or 0)
        cand = T.candidates(toks, starts, reply_at, lo)
        enc_rows.append((t, toks, starts, reply_at, lo, [T.token_class(toks[i]) for i in cand]))
    by_cls: dict[str, list[list[int]]] = defaultdict(list)
    for t, *_r, raw in enc_rows:
        if t["concept"] and t["kind"] == "implied":
            by_cls[t["class"]].append(raw)
    merges = {
        c: T.class_merge(by_cls.get(c, [])) for c in sorted({r[0]["class"] for r in enc_rows})
    }
    rows = []
    n_short = 0
    n_train: Counter = Counter()
    n_null: Counter = Counter()
    n_merged: Counter = Counter()
    for t, toks, starts, reply_at, lo, _raw in enc_rows:
        merge = merges[t["class"]]
        pos = T.read_positions(toks, starts, reply_at, lo, merge)
        _e, key = T.class_key([toks[p] for p in pos], merge)
        n_short += not pos
        if t["concept"]:
            n_train.update(key)
            if t["kind"] == "implied":
                n_null.update(key)
        rows.append(
            {
                "label": probe_label(t["id"]),
                "family": "sibgate_probe",
                "system": t["system"],
                "user": t["user"],
                "prefill": t["assistant"],
                "look_for": "",
                "targets": [],
                "eval_positions": pos,
                "meta": {
                    "class": t["class"],
                    "concept": t["concept"],
                    "kind": t["kind"],
                    "key": key,
                },
            }
        )
    for m in merges.values():
        n_merged.update(c for c, k in m.items() if k != c)
    dest = PROBES_DIR / "capture_prompts.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rows, ensure_ascii=False))
    (PROBES_DIR / "merge.json").write_text(
        json.dumps({c: [m[k] for k in range(T.N_CLASSES)] for c, m in merges.items()}, indent=0)
        + "\n"
    )
    cells = Path(a.site_cells)
    share, n_cells = _site_share(cells) if cells.exists() else (Counter(), 0)
    print(
        f"per key ({len(merges)} classes; merge < {T.MIN_NULL_POSITIONS} implied null positions "
        f"per class): training positions / implied null positions / classes merged / eval-site "
        f"share over {n_cells} cells of {cells}"
    )
    for k, name in enumerate(T.TOKEN_CLASSES):
        sh = f"{100 * share[k] / n_cells:5.1f}%" if n_cells else "  n/a"
        print(f"  {name:>14}: {n_train[k]:7d} {n_null[k]:7d} {n_merged[k]:4d}  {sh}")
    print(
        f"wrote {len(rows)} capture prompts ({n_short} with no read position) to {dest}\n"
        f"GPU capture (not launched): (cd $SNAP_WB && PYTHONPATH=src {MODAL} run "
        "scripts/olens_suite/workspace_bench/wsbench_capture_modal.py --prompts-json "
        f"{dest} --acts-subdir {a.probe_acts} --layers 20,24,28,32,36,40,42,44,48,52,56,60)"
    )
    return 0


# ---------------------------------------------------------------- fit / score (Modal CPU)


def _launch(cmd: str, launch: bool) -> int:
    print(("launching: " if launch else "command (not launched): ") + cmd)
    return subprocess.call(cmd, shell=True, cwd=REPO_ROOT) if launch else 0


def cmd_fit(a: argparse.Namespace) -> int:
    from . import ladder as L
    from . import probes as P
    from . import texts as T

    concepts = _concepts()
    texts = list(U.iter_jsonl(DATA_DIR / "probe_texts.jsonl"))
    cap = {r["label"]: r for r in json.loads((PROBES_DIR / "capture_prompts.json").read_text())}
    merges = json.loads((PROBES_DIR / "merge.json").read_text())  # capture-manifest
    per_class: dict[str, Any] = {}
    for cls, row in sorted(concepts.items()):
        ci = {c: i for i, c in enumerate(row["concepts"])}
        ts = [t for t in texts if t["class"] == cls]
        folds = P.assign_folds(ts, salt=cls)
        per_class[L.slug(cls)] = {
            "class": cls,
            "concepts": row["concepts"],
            # raw token class -> key (fixed across layers); a class without texts: identity
            "merge": merges.get(cls, list(range(T.N_CLASSES))),
            "texts": [
                {
                    "id": probe_label(t["id"]),
                    "c": ci.get(t["concept"], P.NEUTRAL) if t["concept"] else P.NEUTRAL,
                    "kind": t["kind"],
                    "request_idx": 0,
                    "fold": folds[t["id"]],
                    "key": cap.get(probe_label(t["id"]), {}).get("meta", {}).get("key"),
                }
                for t in ts
            ],
        }
    inp = {
        "run": a.run,
        "probe_acts": a.probe_acts,
        "layers": list(L.PROBE_LAYERS),
        "classes": per_class,
    }
    p = PROBES_DIR / "probe_inputs.json"
    p.write_text(json.dumps(inp, ensure_ascii=False))
    print(
        f"wrote {p} ({len(per_class)} classes, {sum(len(c['texts']) for c in per_class.values())}"
        " texts)"
    )
    return _launch(
        f"{MODAL} run {SCRIPTS / 'probes_modal.py'} --phase fit --inputs {p} "
        f"--out-dir {PROBES_DIR}",
        a.launch,
    )


def _distractors(units, rung1, classes, concepts, cps) -> dict[str, Any]:
    from . import fill_judge as F
    from . import ladder as L
    from .classes import canonical_map, compatible_of

    canon = canonical_map(classes)
    names = U.item_sibling_names(units, canon)
    comp = compatible_of(classes)
    out = {}
    for u in units:
        if not F.needs_rung2(rung1.get(u["unit"]) or {}):
            continue
        v = classes.get(f"{u['said_norm']}|{u['id']}")
        if not v or v["class"] == "no_class" or v["class"] not in cps:
            continue
        cp = cps[v["class"]]
        anchor = v["said_canonical"] if v["no_gold"] else v["gold"]
        excl = {
            v["said_canonical"],
            *([] if v["no_gold"] else [v["gold"]]),
            *comp.get(U.norm(v["gold"]), ()),
            *names.get(u["id"], {}),
        }
        out[u["unit"]] = L.select_distractors(
            u["unit"], anchor, cp.concepts, cp.centroids, excl, said=v["said_canonical"]
        )
    return out


def cmd_score(a: argparse.Namespace) -> int:
    """Freeze the distractors (sha256 recorded; ``label`` checks it), write the eval inputs and
    print / launch the Modal score phase."""
    from . import ladder as L
    from . import texts as T

    out = Path(a.out)
    units = _all_units(a)
    classes, concepts = _classes(), _concepts()
    cps = L.load_class_probes(PROBES_DIR, concepts)
    dis = _distractors(units, _rung1(out, _arms(a)), classes, concepts, cps)
    dp = DATA_DIR / "distractors.json"
    dp.write_text(json.dumps(dis, indent=0, ensure_ascii=False))
    sha = hashlib.sha256(dp.read_bytes()).hexdigest()
    items = _items()
    keys = {
        it: {str(s["pos"]): T.token_class(s.get("token") or "") for s in items[it]["sites"]}
        for it in items
    }
    inp = {
        "run": a.run,
        "items": sorted(items),
        "site_keys": keys,
        "eval_acts": {
            str(la): ("acts-hallucination-chat-L42" if la == 42 else "acts-hallucination-chat")
            for la in L.PROBE_LAYERS
        },
        "layers": list(L.PROBE_LAYERS),
        "classes": sorted(L.slug(c) for c in concepts),
        "distractors_sha256": sha,
    }
    p = PROBES_DIR / "score_inputs.json"
    p.write_text(json.dumps(inp))
    print(f"froze {len(dis)} distractor sets ({dp}, sha256 {sha[:12]})")
    return _launch(
        f"{MODAL} run {SCRIPTS / 'probes_modal.py'} --phase score --inputs {p} "
        f"--out-dir {PROBES_DIR} --distractors {dp}",
        a.launch,
    )


# ---------------------------------------------------------------- label


def label_rows(
    units: list[dict[str, Any]],
    rung1: dict[str, dict[str, Any]],
    classes: dict[str, dict[str, Any]],
    dis: dict[str, Any],
    cps: dict[str, Any],
    site_tok: dict[tuple[str, int], str],
    *,
    ready: bool,
    l42_tokens_ok: bool = True,
    stale: set[str] | frozenset[str] = frozenset(),
) -> tuple[list[dict[str, Any]], Counter]:
    """Rung 2 (strict precedence), the sensitivity variants and the matched null for every
    unit; ``cmd_label`` and ``hbench gate`` (plan 0010) share it. ``units`` = every gated arm's
    units (the cross index, the shared prior and the null are computed over them). ``stale``
    (plan 0010 §10.6): units whose frozen distractor sha differs from their class's score
    ``meta.json`` -> ``probe_na`` (``stale_score``)."""
    from . import fill_judge as F
    from . import ladder as L
    from . import texts as T
    from .classes import NO_CLASS, canonical_map

    canon = canonical_map(classes)
    idx = L.cross_index(units, canon)
    prior = L.shared_prior_arms(units, classes, canon)
    items_by_gold: dict[str, set[str]] = defaultdict(set)
    for k, v in classes.items():
        if not v["no_gold"]:
            items_by_gold[U.norm(v["gold"])].add(k.split("|", 1)[1])
    named: dict[tuple[str, str], set[tuple[int, int]]] = defaultdict(set)
    for (item, c), sites in idx.items():
        named[(item, c)] |= {(la, p) for _arm, la, p in sites}
    kw = {"l42_tokens_ok": l42_tokens_ok}
    rows, nulls = [], Counter()
    for u in units:
        r1 = rung1.get(u["unit"])
        v = classes.get(f"{u['said_norm']}|{u['id']}")
        r2 = variants = None
        tk = site_tok.get((u["id"], int(u["pos"])))
        raw_key = None if tk is None else T.token_class(tk)
        if ready and r1 is not None and F.needs_rung2(r1):
            cls = v["class"] if v else NO_CLASS
            if u["unit"] in stale:
                st = {
                    "na_reason": "stale_score",
                    "fire_L": False,
                    "fire_nearby": [],
                    "cross_arms": [],
                    "no_gold": not v or bool(v["no_gold"]),
                    "status": "probe_na",
                }
                key = None if raw_key is None else T.TOKEN_CLASSES[raw_key]
                var = {k: st for k in ("cross_same_pos", "cross_same_family", "cross_shared_prior")}
                rows.append(L.label_row(u, r1, st, v, var, key=key))
                print(f"[gate] {u['unit']}: distractor sha != score meta -> probe_na stale_score")
                continue
            up = L.UnitProbe(
                u["unit"],
                u["arm"],
                u["id"],
                u["layer"],
                u["pos"],
                cls,
                v["said_canonical"] if v else u["said"],
                None if (not v or v["no_gold"]) else v["gold"],
                (dis.get(u["unit"]) or {}).get("distractors", []),
                raw_key,
            )
            cp = cps.get(cls)
            r2 = L.rung2(up, cp, idx, **kw)
            disc = prior.get((cls, U.norm(up.said)), set())
            variants = {
                "cross_same_pos": L.rung2(up, cp, idx, **kw, cross_kwargs={"same_pos": True}),
                "cross_same_family": L.rung2(up, cp, idx, **kw, cross_kwargs={"same_family": True}),
                "cross_shared_prior": L.rung2(up, cp, idx, **kw, cross_kwargs={"discounted": disc}),
            }
            mn = L.matched_null(up, cp, items_by_gold, named)
            nulls["n_sites"] += mn["n_sites"]
            nulls["n_fire"] += mn["n_fire"]
        key = None if raw_key is None else T.TOKEN_CLASSES[raw_key]
        rows.append(L.label_row(u, r1, r2, v, variants, key=key))
    return rows, nulls


def cmd_label(a: argparse.Namespace) -> int:
    """Rung 2 (strict precedence) for every rung-2 unit, the sensitivity variants, the matched
    null and the labels files. Before classes / probe scores exist, rung-2 units get
    ``final: null`` (scoring keeps their raw leaf)."""
    from . import ladder as L

    out = Path(a.out)
    units = _all_units(a)
    site_tok = {
        (it, int(s["pos"])): s.get("token") or ""
        for it, row in _items().items()
        for s in row["sites"]
    }
    rung1 = _rung1(out, _arms(a))
    have_classes = (DATA_DIR / "classes.json").exists() and (DATA_DIR / "concepts.json").exists()
    classes = _classes() if have_classes else {}
    concepts = _concepts() if have_classes else {}
    dp = DATA_DIR / "distractors.json"
    dis = json.loads(dp.read_text()) if dp.exists() else {}
    ss = PROBES_DIR / "score" / "summary.json"
    if dis and ss.exists():
        want = json.loads(ss.read_text()).get("distractors_sha256")
        if want != hashlib.sha256(dp.read_bytes()).hexdigest():
            raise SystemExit("distractors.json changed after scoring (sha256 mismatch)")
    cps = L.load_class_probes(PROBES_DIR, concepts) if concepts else {}
    ready = have_classes and ss.exists() and any(cp.z is not None for cp in cps.values())
    rows, nulls = label_rows(
        units,
        rung1,
        classes,
        dis,
        cps,
        site_tok,
        ready=ready,
        l42_tokens_ok=not a.l42_token_mismatch,
    )
    ld = Path(a.labels_dir) if a.labels_dir else out / "labels"
    n = L.write_labels(rows, ld)
    (ld / "matched_null.json").write_text(json.dumps(dict(nulls)) + "\n")
    print(
        f"labels ({'rung 2 scored' if ready else 'rung 2 pending'}): {n}; matched null "
        f"{dict(nulls)}"
    )
    return 0


# ---------------------------------------------------------------- report / figures


def cmd_report(a: argparse.Namespace) -> int:
    from hallucination_bench import report_v2

    ld = Path(a.labels_dir) if a.labels_dir else Path(a.out) / "labels"
    from . import ladder as L

    md = report_v2.sibling_gate_section(ld, _arms(a))
    rows = [
        r
        for arm in _arms(a)
        if (ld / f"{arm}.jsonl").exists()
        for r in U.iter_jsonl(ld / f"{arm}.jsonl")
    ]
    if md:
        md += L.per_key_table(rows)
    sums = {
        arm: U.summarize_label_rows(U.iter_jsonl(ld / f"{arm}.jsonl"))
        for arm in _arms(a)
        if (ld / f"{arm}.jsonl").exists()
    }
    (Path(a.out) / "report.md").write_text("\n".join(md) + "\n")
    (Path(a.out) / "report.json").write_text(json.dumps(sums, indent=1) + "\n")
    (Path(a.out) / "report_per_key.json").write_text(
        json.dumps(L.per_key_summary(rows), indent=1) + "\n"
    )
    print("\n".join(md) if md else f"no labels under {ld}")
    return 0


def cmd_figures(a: argparse.Namespace) -> int:
    return subprocess.call(
        [sys.executable, str(SCRIPTS / "plot_sibling_gate.py"), "--out", str(Path(a.out))],
        cwd=REPO_ROOT,
    )


STAGES = {
    "units": cmd_units,
    "resample-manifests": cmd_resample_manifests,
    "resample-collect": cmd_resample_collect,
    "fill": cmd_fill,
    "fill-calib": cmd_fill_calib,
    "classes": cmd_classes,
    "concepts": cmd_concepts,
    "probe-texts": cmd_probe_texts,
    "capture-manifest": cmd_capture_manifest,
    "fit": cmd_fit,
    "score": cmd_score,
    "label": cmd_label,
    "report": cmd_report,
    "figures": cmd_figures,
}


def add_parser(sub: Any) -> None:
    sg = sub.add_parser("sibling-gate", help="plan 0009: the sibling resample + probe gate")
    sg.add_argument("stage", choices=list(STAGES))
    sg.add_argument("--out", default=str(U.OUT), help="sibling-gate output root")
    sg.add_argument("--arms", default=None, help="comma list (default: the 14 in-scope arms)")
    sg.add_argument(
        "--cells-root",
        default=str(U.CELLS_ROOT_DEFAULT),
        help="units: DIR with <arm>/cells.jsonl (the v2.7 judge outputs)",
    )
    sg.add_argument("--labels-dir", default=None, help="labels dir (default <out>/labels)")
    sg.add_argument("--no-calls", action="store_true", help="LLM stages: replay the cache only")
    sg.add_argument("--rpm", type=float, default=1500.0)
    sg.add_argument("--concurrency", type=int, default=64)
    sg.add_argument(
        "--dry",
        action="store_true",
        help="resample-manifests: only count requested / over-generated cells",
    )
    sg.add_argument(
        "--precheck",
        type=int,
        default=0,
        help="resample-manifests / -collect: the step-0 N-cell pre-check (5)",
    )
    sg.add_argument("--manifest-11", default=None, help="source acts-hallucination-chat manifest")
    sg.add_argument("--manifest-l42", default=None, help="source ...-L42 manifest")
    sg.add_argument("--pool", default=None, help="resample-collect: the pool")
    sg.add_argument("--src", default=None, help="resample-collect: gen dir or readouts jsonl")
    sg.add_argument("--run-config", default=None, help="resample-collect: run_config.json")
    sg.add_argument(
        "--orig-readouts",
        default=None,
        help="resample-collect: the arm's original hallucination.jsonl (shape check)",
    )
    sg.add_argument("--shape-cells", type=int, default=20)
    sg.add_argument("--failed-pools", default=None, help="fill: pools whose pre-check failed")
    sg.add_argument("--split", default="dev", choices=("dev", "gold"))
    sg.add_argument("--make", action="store_true", help="fill-calib: write the unlabelled sample")
    sg.add_argument("--n", type=int, default=40)
    sg.add_argument("--seed", type=int, default=0)
    sg.add_argument("--tokenizer", default="Qwen/Qwen3.6-27B")
    sg.add_argument("--run", default="sibgate-v1", help="fit / score: the sibling-probes run dir")
    sg.add_argument(
        "--min-class-pairs",
        type=int,
        default=0,
        help="concepts: only classes with >= N (said, item) pairs (pilot cost cap; default all)",
    )
    sg.add_argument("--probe-acts", default="acts-sibgate-probe", help="probe-text acts subdir")
    sg.add_argument(
        "--site-cells",
        default="/workspace/hb-readouts/s3d-rl600-all11/hallucination.jsonl",
        help="capture-manifest: readouts whose cells give the eval-site share per key",
    )
    sg.add_argument("--launch", action="store_true", help="fit / score: launch the Modal job")
    sg.add_argument(
        "--l42-token-mismatch",
        action="store_true",
        help="label: the step-0 L42-vs-11-layer token check failed (nearby off)",
    )
    sg.set_defaults(fn=lambda a: (_configure(a), STAGES[a.stage](a))[1])
