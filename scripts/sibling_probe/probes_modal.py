"""Plan 0003 §Phase E on Modal: stream the captured residuals, fit the cross-fit DiffMean probes,
score the held-out texts and every eval cell, write small outputs (CPU container, no GPU).

Inputs:
  * ``wsbench-acts:/<probe_acts>/`` (probe + neutral texts) and ``wsbench-acts:/<eval_acts>/``
    (eval items): capture ``manifest.json`` (``ActsManifest``: ``layers``, ``prompts[].{label,
    file, n_pos, eval_positions}``) + one ``<label>.safetensors`` per row (``acts`` [layers,
    n_pos, d] bf16). Mounted read-only; each file is opened once and only its eval positions
    at the requested layers are read (``get_slice("acts")[l0:l1, p0:p1]``, kept as fp16, ~5.5 GB
    for the probe capture: the volume is latency-bound per open); fitting runs layer by layer.
  * ``--inputs`` (``hbench sibling probes --prepare``): the concept order per category, the
    probe-text and item metadata, and the Qwen token ids of every token alias (for the
    unembedding check; W_U rows come from the ``jlens-hf-cache`` HF volume).

Outputs: one CPU container per category (``probe_category.map``), TWO recipes each (plan 0003
rev 11): ``v2`` = shrinkage-whitened DiffMean (primary), ``v0`` = raw text-mean DiffMean (rev 10).
Pieces land on the ``sibling-probes`` volume (``/<run>/<recipe>/<category>/``) and, via the local
entrypoint, in ``--out-dir/pieces/``; ``hbench sibling probes`` merges them
(``probe_io.merge_pieces``) into ``--out-dir/{v2,v0}/``:
  directions_<cat>.safetensors  ``L{L}/fold{k}`` -> raw (V2: whitened) directions [C, d] fp32
  oof_<cat>.npz                 out-of-fold scores per layer (``s_L{L}`` [P, C]) + row/text index
  thresholds.json               per (category, layer): every fold's thresholds, pooled
                                thresholds, text-bootstrap CIs of the pooled tau_cat / tau_neutral
                                / tau(c|g); V2 also the chosen lambda per outer fold
  validity.jsonl                {category, concept, layer, auroc, pass, fpr_neutral}
  unembed.jsonl                 {category, concept, layer, cos_max, cos_mean, per_alias}
  eval_scores.jsonl.gz          {id, category, layer, pos, offset, concept, gold, s, z, det_cat,
                                det_cg, det_cg_all, det_neu, det_ceil} (fold-averaged)
and ``--out-dir/summary.json`` (config, counts, timings).

``--synthetic-probes`` (smoke): the probe texts' residuals are synthesised inside the container
from the eval capture's windows of the same (category, ending) plus noise and a planted
per-concept direction, so the job runs end to end on ``acts-sibling-smoke`` (2 eval items).

    MODAL_PROFILE=safety-sahan-2 uvx --python 3.12 modal@latest run \\
        scripts/sibling_probe/probes_modal.py --eval-acts acts-sibling-eval \\
        --probe-acts acts-sibling-probe --run sibling-v1
"""

import gzip
import io
import json
import os
import time
import zlib
from pathlib import Path, PurePosixPath

import modal

MODEL_ID = "Qwen/Qwen3.6-27B"
HF_MOUNT = Path("/hf")
ACTS_MOUNT = Path("/acts")
OUT_MOUNT = Path("/out")
_HERE = Path(__file__).resolve()
_PROBES_PY = (
    _HERE.parents[2] / "src" / "hallucination_bench" / "evals" / "sibling_probe" / "probes.py"
    if len(_HERE.parents) > 2
    else _HERE  # inside the container (/root/probes_modal.py): never added
)


def _hf_token() -> str:
    for p in (
        os.environ.get("HF_TOKEN"),
        Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "token",
        Path("~/.cache/huggingface/token").expanduser(),
    ):
        if isinstance(p, str) and p:
            return p
        if isinstance(p, Path) and p.is_file():
            return p.read_text().strip()
    return ""


image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch>=2.6", index_url="https://download.pytorch.org/whl/cpu")
    .uv_pip_install("numpy>=2.0", "safetensors>=0.5", "huggingface_hub>=0.35")
    .env({"HF_HOME": str(HF_MOUNT), "PYTHONUNBUFFERED": "1"})
)
if (
    _PROBES_PY.name == "probes.py" and _PROBES_PY.is_file()
):  # local launch; inside the container the file is already at /root
    image = image.add_local_file(str(_PROBES_PY), "/root/sibprobes.py")

app = modal.App("sibling-probes")
acts_vol = modal.Volume.from_name("wsbench-acts", create_if_missing=True)
hf_cache = modal.Volume.from_name("jlens-hf-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("sibling-probes", create_if_missing=True)
volumes: dict[str | PurePosixPath, modal.Volume] = {
    PurePosixPath(ACTS_MOUNT): acts_vol.read_only(),
    PurePosixPath(HF_MOUNT): hf_cache,
    PurePosixPath(OUT_MOUNT): out_vol,
}


def _unembed_rows(ids: list[int]):
    """W_U rows [n, d] fp32 for the given token ids (only the lm_head shard is touched)."""
    import numpy as np
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    idx = json.loads(Path(hf_hub_download(MODEL_ID, "model.safetensors.index.json")).read_text())
    keys = [k for k in idx["weight_map"] if k.endswith("lm_head.weight")]
    if len(keys) != 1:
        raise RuntimeError(f"lm_head.weight not unique in the index: {keys}")
    shard = hf_hub_download(MODEL_ID, idx["weight_map"][keys[0]])
    with safe_open(shard, framework="pt", device="cpu") as f:
        sl = f.get_slice(keys[0])
        rows = [sl[i : i + 1].float().numpy()[0] for i in ids]
    return np.stack(rows) if rows else np.zeros((0, 0), np.float32)


RECIPES = ("v2", "v0")  # primary first


@app.function(
    image=image,
    volumes=volumes,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": _hf_token()})],
    cpu=16,
    memory=49152,
    timeout=6 * 3600,
)
def probe_category(cat: str, inputs: dict, cfg: dict) -> dict:
    """Both recipes for one category at every layer; returns the category's pieces."""
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import sibprobes as P  # type: ignore
    from safetensors import safe_open
    from safetensors.numpy import save as st_save

    if not os.environ.get("HF_TOKEN"):
        os.environ.pop("HF_TOKEN", None)
    t0 = time.time()
    layers = [int(x) for x in cfg["layers"]]
    order: list[str] = inputs["concepts"][cat]
    cidx = {c: i for i, c in enumerate(order)}
    C = len(order)

    def manifest(sub: str) -> dict:
        return json.loads((ACTS_MOUNT / sub / "manifest.json").read_text())

    ev_man = manifest(cfg["eval_acts"])
    ev_entries = {p["label"]: p for p in ev_man["prompts"]}
    items = [it for it in inputs["items"] if it["category"] == cat and it["id"] in ev_entries]
    missing_items = sorted(
        {it["id"] for it in inputs["items"] if it["category"] == cat} - set(ev_entries)
    )
    for L in layers:
        if L not in ev_man["layers"]:
            raise RuntimeError(f"eval capture lacks layer {L} ({ev_man['layers']})")

    # One read per file: the eval positions of every requested layer, kept as fp16 (the volume
    # is latency-bound per file open, so re-reading each file once per layer is ~5x slower).
    cache: dict[tuple[str, str], tuple] = {}

    def fetch(sub: str, entry: dict, lis: list[int]) -> None:
        pos = list(entry.get("eval_positions") or range(entry["n_pos"]))
        p0, p1 = min(pos), max(pos) + 1
        a, b = min(lis), max(lis) + 1
        with safe_open(str(ACTS_MOUNT / sub / entry["file"]), framework="pt", device="cpu") as f:
            x = f.get_slice("acts")[a:b, p0:p1].float().numpy()
        x = x[[li - a for li in lis]][:, [p - p0 for p in pos]].astype(np.float16)
        off = [entry["n_pos"] - 1 - p for p in pos]
        cache[(sub, entry["label"])] = (dict(zip(lis, x, strict=True)), pos, off)

    def prefetch(sub: str, entries: list[dict], lis: list[int]) -> None:
        tp = time.time()
        with ThreadPoolExecutor(int(cfg.get("threads", 64))) as ex:
            for i, _ in enumerate(ex.map(lambda e: fetch(sub, e, lis), entries)):
                if (i + 1) % 500 == 0:
                    print(
                        f"[probes {cat}] {sub}: read {i + 1}/{len(entries)} "
                        f"({time.time() - tp:.0f}s)",
                        flush=True,
                    )
        print(f"[probes {cat}] {sub}: {len(entries)} files read in {time.time() - tp:.0f}s")

    def load_many(sub: str, entries: list[dict], li: int):
        out = []
        for e in entries:
            xs, pos, off = cache[(sub, e["label"])]
            out.append((xs[li].astype(np.float32), pos, off))
        return out

    synthetic = bool(cfg.get("synthetic_probes"))
    texts = [t for t in inputs["probe_texts"] if t["category"] == cat]
    if synthetic:
        have = {it["request_idx"] for it in items}
        texts = [t for t in texts if t["request_idx"] in have]
        pr_entries = {}
        pr_layers = ev_man["layers"]
    else:
        pr_man = manifest(cfg["probe_acts"])
        pr_layers = pr_man["layers"]
        pr_entries = {p["label"]: p for p in pr_man["prompts"]}
        missing_texts = [t["id"] for t in texts if t["id"] not in pr_entries]
        if missing_texts:
            print(f"[probes {cat}] {len(missing_texts)} probe texts not captured", flush=True)
        texts = [t for t in texts if t["id"] in pr_entries]
    if not texts:
        return {"category": cat, "empty": True}
    folds = P.assign_folds(texts, salt=cat)

    # W_U rows for the unembedding check (this category's token aliases only)
    tok_ids = (inputs.get("token_ids") or {}).get(cat) or {}
    flat = sorted({i for v in tok_ids.values() for i in v})
    W = None
    if flat and not cfg.get("skip_unembed"):
        try:
            W = _unembed_rows(flat)
            print(f"[probes {cat}] W_U rows {W.shape}", flush=True)
        except Exception as e:  # the check is reported, never fatal
            print(f"[probes {cat}] unembedding check skipped: {type(e).__name__}: {e}", flush=True)
    wrow = {t: i for i, t in enumerate(flat)}

    prefetch(
        cfg["eval_acts"],
        [ev_entries[it["id"]] for it in items],
        [ev_man["layers"].index(L) for L in layers],
    )
    if not synthetic:
        prefetch(
            cfg["probe_acts"],
            [pr_entries[t["id"]] for t in texts],
            [pr_layers.index(L) for L in layers],
        )

    tensors: dict[str, dict[str, np.ndarray]] = {r: {} for r in RECIPES}
    oof_store: dict[str, dict[str, np.ndarray]] = {r: {} for r in RECIPES}
    thr_out: dict[str, dict] = {r: {} for r in RECIPES}
    validity_rows: dict[str, list] = {r: [] for r in RECIPES}
    unembed_rows: dict[str, list] = {r: [] for r in RECIPES}
    eval_buf = {r: io.StringIO() for r in RECIPES}
    n_eval_rows = 0
    timings = {}
    for L in layers:
        tl = time.time()
        li_ev = ev_man["layers"].index(L)
        ev_loaded = load_many(cfg["eval_acts"], [ev_entries[it["id"]] for it in items], li_ev)
        ev_w = {it["id"]: w for it, w in zip(items, ev_loaded, strict=True)}
        if synthetic:
            rng = np.random.default_rng([L, 7])
            base = {}
            for it in items:
                base.setdefault(it["request_idx"], ev_w[it["id"]])
            dirs = {}
            pr_w = []
            for t in texts:
                x, pos, off = base[t["request_idx"]]
                rms = float(x.std())
                if t["concept"] is not None and t["concept"] not in dirs:
                    seed = zlib.crc32(f"{cat}/{t['concept']}".encode())
                    u = np.random.default_rng(seed).normal(size=x.shape[1])
                    dirs[t["concept"]] = u / np.linalg.norm(u)
                y = x + 0.5 * rms * rng.normal(size=x.shape).astype(np.float32)
                if t["concept"] is not None:
                    a = 1.0 if t["kind"] == "explicit" else 0.5
                    y = y + (a * rms * np.sqrt(x.shape[1]) * 0.05) * dirs[t["concept"]][None, :]
                pr_w.append((y.astype(np.float32), pos, off))
        else:
            li_pr = pr_layers.index(L)
            pr_w = load_many(cfg["probe_acts"], [pr_entries[t["id"]] for t in texts], li_pr)
        tread = time.time() - tl
        ts = P.TextSet.build(
            [w[0] for w in pr_w],
            [t["id"] for t in texts],
            [cidx[t["concept"]] if t["concept"] is not None else P.NEUTRAL for t in texts],
            [t["kind"] for t in texts],
            [t["request_idx"] for t in texts],
            [folds[t["id"]] for t in texts],
            offsets=[w[2] for w in pr_w],
        )
        its = [it for it in items if it["gold"] in cidx]
        ws = [ev_w[it["id"]] for it in its]
        wh = P.Whitener(ts, C)  # eigendecompositions are memoised across the fold sets
        for recipe in RECIPES:
            trc = time.time()
            res = P.run_category_layer(ts, C, recipe=recipe, whitener=wh)
            print(
                f"[probes {cat}] L{L} {recipe} fitted in {time.time() - trc:.0f}s lam={res.hyper}",
                flush=True,
            )
            for k, mdl in enumerate(res.cf.models):
                tensors[recipe][f"L{L}/fold{k}"] = mdl.D.astype(np.float32)
            st = oof_store[recipe]
            st[f"s_L{L}"] = res.cf.oof.astype(np.float32)
            if "row_text" not in st:
                st["row_text"] = ts.text.astype(np.int32)
                st["row_offset"] = ts.offset.astype(np.int16)
                st["row_ending"] = ts.ending.astype(np.int8)
                st["text_id"] = np.array(ts.ids)
                st["text_concept"] = ts.concept.astype(np.int16)
                st["text_kind"] = np.array(ts.kind)
                st["text_fold"] = ts.fold.astype(np.int8)
                st["concepts"] = np.array(order)
            # pooled-threshold CIs (text-level bootstrap)
            rc, ri, oof, rt, rf = ts.row_concept, ts.row_implied, res.cf.oof, ts.text, ts.row_fold
            ci = {"tau_cat": [], "tau_neu": [], "tau_cg": []}
            nb = int(cfg.get("n_boot", 500))
            for c in range(C):
                m = (rc != c) & (rc >= 0)
                ci["tau_cat"].append(P.threshold_ci(oof[m, c], rt[m], n_boot=nb, seed=c))
                m = rc == P.NEUTRAL
                ci["tau_neu"].append(P.threshold_ci(oof[m, c], rt[m], n_boot=nb, seed=c))
                row = []
                for g in range(C):
                    m = (rc == g) & ri
                    row.append(
                        P.threshold_ci(oof[m, c], rt[m], n_boot=max(nb // 5, 50), seed=1000 * c + g)
                    )
                ci["tau_cg"].append(row)
            thr_out[recipe][str(L)] = {
                "folds": [th.to_dict() for th in res.ths],
                "pooled": res.pooled.to_dict(),
                "ci": ci,
                "lam": res.hyper,
            }
            for c, cid in enumerate(order):
                # neutral FPR: share of fold k's neutral OOF rows above fold k's tau_cat, mean
                # over folds (diagnostic)
                fn = []
                for k, th in enumerate(res.ths):
                    xn = oof[(rf == k) & (rc == P.NEUTRAL), c]
                    xn = xn[np.isfinite(xn)]
                    if xn.size and np.isfinite(th.tau_cat[c]):
                        fn.append(float(np.mean(xn > th.tau_cat[c])))
                validity_rows[recipe].append(
                    {
                        "category": cat,
                        "concept": cid,
                        "layer": L,
                        "auroc": round(float(res.valid_auroc[c]), 4),
                        "pass": bool(res.valid_auroc[c] >= P.VALIDITY_BY_RECIPE[recipe]),
                        "n_pos": int((rc == c).sum()),
                        "fpr_neutral": round(float(np.mean(fn)), 4) if fn else None,
                    }
                )
                ids = tok_ids.get(cid) or []
                if W is not None and ids:
                    Wc = W[[wrow[i] for i in ids]]
                    cos = np.mean(
                        [P.unembed_cosines(m_.D[c : c + 1], Wc)[0] for m_ in res.cf.models], axis=0
                    )
                    unembed_rows[recipe].append(
                        {
                            "category": cat,
                            "concept": cid,
                            "layer": L,
                            "cos_max": round(float(cos.max()), 4),
                            "cos_mean": round(float(cos.mean()), 4),
                            "per_alias": {
                                str(i): round(float(v), 4) for i, v in zip(ids, cos, strict=True)
                            },
                        }
                    )
            if not its:
                continue
            H = np.concatenate([w[0] for w in ws])
            ending = np.concatenate(
                [[it["request_idx"]] * len(w[1]) for it, w in zip(its, ws, strict=True)]
            )
            offset = np.concatenate([w[2] for w in ws])
            gold = np.concatenate(
                [[cidx[it["gold"]]] * len(w[1]) for it, w in zip(its, ws, strict=True)]
            )
            es = P.score_eval(res.cf.models, res.ths, H, ending, offset, gold)
            r = 0
            for it, w in zip(its, ws, strict=True):
                for pos, off in zip(w[1], w[2], strict=True):
                    for c, cid in enumerate(order):

                        def f(a, _r=r, _c=c):
                            v = float(a[_r, _c])
                            return round(v, 4) if np.isfinite(v) else None

                        eval_buf[recipe].write(
                            json.dumps(
                                {
                                    "id": it["id"],
                                    "category": cat,
                                    "layer": L,
                                    "pos": int(pos),
                                    "offset": int(off),
                                    "concept": cid,
                                    "gold": it["gold"],
                                    "s": f(es.s),
                                    "z": f(es.z),
                                    "det_cat": f(es.det_cat),
                                    "det_cg": f(es.det_cg),
                                    "det_cg_all": f(es.det_cg_all),
                                    "det_neu": f(es.det_neu),
                                    "det_ceil": f(es.det_ceil),
                                }
                            )
                            + "\n"
                        )
                        if recipe == RECIPES[0]:
                            n_eval_rows += 1
                    r += 1
        wh.drop()
        timings[str(L)] = {"read_s": round(tread, 1), "total_s": round(time.time() - tl, 1)}
        print(
            f"[probes {cat}] L{L}: {len(texts)} texts, {len(items)} eval items, "
            f"read {tread:.0f}s, total {time.time() - tl:.0f}s",
            flush=True,
        )

    run_dir = OUT_MOUNT / cfg["run"]
    pieces: dict[str, dict[str, bytes]] = {}
    for recipe in RECIPES:
        f: dict[str, bytes] = {}
        f["directions.safetensors"] = st_save(tensors[recipe])
        b = io.BytesIO()
        np.savez_compressed(b, **oof_store[recipe])
        f["oof.npz"] = b.getvalue()
        f["thresholds.json"] = json.dumps(
            {"concepts": {cat: order}, "thresholds": {cat: thr_out[recipe]}}
        ).encode()
        f["validity.jsonl"] = "".join(json.dumps(r) + "\n" for r in validity_rows[recipe]).encode()
        f["unembed.jsonl"] = "".join(json.dumps(r) + "\n" for r in unembed_rows[recipe]).encode()
        f["eval_scores.jsonl.gz"] = gzip.compress(eval_buf[recipe].getvalue().encode())
        pieces[recipe] = f
        d = run_dir / recipe / cat
        d.mkdir(parents=True, exist_ok=True)
        for name, data in f.items():
            (d / name).write_bytes(data)
    out_vol.commit()
    return {
        "category": cat,
        "pieces": pieces,
        "n_probe_texts": len(texts),
        "n_eval_items": len(items),
        "missing_eval_items": missing_items,
        "n_eval_rows": n_eval_rows,
        "validity_pass": {
            r: f"{sum(x['pass'] for x in validity_rows[r])}/{len(validity_rows[r])}"
            for r in RECIPES
        },
        "unembed_rows": {r: len(unembed_rows[r]) for r in RECIPES},
        "timings": timings,
        "runtime_s": round(time.time() - t0, 1),
    }


@app.local_entrypoint()
def main(
    inputs: str = "outputs/sibling/probes/inputs.json",
    eval_acts: str = "acts-sibling-eval",
    probe_acts: str = "acts-sibling-probe",
    run: str = "sibling-v3",
    out_dir: str = "outputs/sibling/probes",
    layers: str = "28,36,42,44,52",
    n_boot: int = 500,
    synthetic_probes: bool = False,
    skip_unembed: bool = False,
    categories: str = "",
) -> None:
    inp = json.loads(Path(inputs).read_text())
    cfg = {
        "eval_acts": eval_acts,
        "probe_acts": probe_acts,
        "run": run,
        "layers": [int(x) for x in layers.split(",")],
        "n_boot": n_boot,
        "synthetic_probes": synthetic_probes,
        "skip_unembed": skip_unembed,
        "recipes": list(RECIPES),
    }
    cats = categories.split(",") if categories else list(inp["concepts"])
    out = Path(out_dir)
    t0 = time.time()
    metas = []
    for res in probe_category.map(cats, kwargs={"inputs": inp, "cfg": cfg}):
        if res.get("empty"):
            print(f"[probes] {res['category']}: no probe texts")
            continue
        for recipe, files in res["pieces"].items():
            d = out / "pieces" / recipe / res["category"]
            d.mkdir(parents=True, exist_ok=True)
            for name, data in files.items():
                (d / name).write_bytes(data)
        meta = {k: v for k, v in res.items() if k != "pieces"}
        metas.append(meta)
        print(f"[probes] {res['category']} done ({res['runtime_s']}s)", flush=True)
    summary = {
        "config": cfg,
        "git_sha": inp.get("git_sha"),
        "layers": cfg["layers"],
        "recipes": list(RECIPES),
        "categories": [m["category"] for m in metas],
        "n_probe_texts": sum(m["n_probe_texts"] for m in metas),
        "n_eval_items": sum(m["n_eval_items"] for m in metas),
        "missing_eval_items": sorted({i for m in metas for i in m["missing_eval_items"]}),
        "n_eval_rows": sum(m["n_eval_rows"] for m in metas),
        "per_category": metas,
        "synthetic_probes": synthetic_probes,
        "wall_s": round(time.time() - t0, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[probes] pieces -> {out / 'pieces'} (also sibling-probes:/{run}/); merge with hbench")
