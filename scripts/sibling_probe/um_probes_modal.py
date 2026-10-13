"""Plan 0006 §4 on Modal (CPU): the two probe types per layer for the country concept set C.

Two phases (the leakage guard of §4a):

* ``--phase fit``: per layer, read the probe texts' window residuals from ``wsbench-acts:/<probe
  acts>/`` (one container per layer), cross-fit type (a) whitened DiffMean (plan 0003 V2,
  ``probes.py``) and type (b) multinomial logistic regression (``um_probe.py``), the per-fold
  null statistics, the implied validity of both and the implied centroids. Fold models are saved
  on the ``sibling-probes`` volume (``/<run>/fit/L<L>.npz``); validity / lambda / C / centroids
  come back to ``--out-dir/fit/``. **No eval residual is read.**
* ``--phase score``: per layer, load the frozen fold models and score every country eval cell
  (``acts-sibling-eval-all11`` for 20..60, ``acts-sibling-eval`` for 42) -> fold-averaged null z
  of every concept in C, both types -> ``--out-dir/score/L<L>.npz``. The run records the sha256
  of the distractor file it was launched after (``--distractors``); ``label`` refuses a mismatch.

    MODAL_PROFILE=safety-sahan-2 uvx --python 3.12 modal@latest run \\
        scripts/sibling_probe/um_probes_modal.py --phase fit
"""

import hashlib
import io
import json
import os
import time
from pathlib import Path, PurePosixPath

import modal

ACTS_MOUNT = Path("/acts")
OUT_MOUNT = Path("/out")
_HERE = Path(__file__).resolve()
_SRC = (
    _HERE.parents[2] / "src" / "hallucination_bench" / "evals" / "sibling_probe"
    if len(_HERE.parents) > 2
    else _HERE
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch>=2.6", index_url="https://download.pytorch.org/whl/cpu")
    .uv_pip_install("numpy>=2.0", "safetensors>=0.5", "scikit-learn>=1.5")
    .env({"PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "16", "OPENBLAS_NUM_THREADS": "16"})
)
if (_SRC / "probes.py").is_file():  # local launch; inside the container the files are at /root
    image = image.add_local_file(str(_SRC / "probes.py"), "/root/sibprobes.py")
    image = image.add_local_file(str(_SRC / "um_probe.py"), "/root/um_probe.py")

app = modal.App("um-country-probes")
acts_vol = modal.Volume.from_name("wsbench-acts", create_if_missing=True)
out_vol = modal.Volume.from_name("sibling-probes", create_if_missing=True)
volumes: dict[str | PurePosixPath, modal.Volume] = {
    PurePosixPath(ACTS_MOUNT): acts_vol.read_only(),
    PurePosixPath(OUT_MOUNT): out_vol,
}


def _load_windows(sub: str, labels: list[str], layer: int, threads: int = 64):
    """{label: (x [n_win, d] float32, pos list, offset list)} at one layer."""
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    from safetensors import safe_open

    man = json.loads((ACTS_MOUNT / sub / "manifest.json").read_text())
    if layer not in man["layers"]:
        raise RuntimeError(f"{sub} lacks layer {layer} ({man['layers']})")
    li = man["layers"].index(layer)
    ent = {p["label"]: p for p in man["prompts"]}
    missing = [x for x in labels if x not in ent]
    if missing:
        raise RuntimeError(f"{sub}: {len(missing)} labels not captured, e.g. {missing[:3]}")

    def one(label: str):
        e = ent[label]
        pos = list(e.get("eval_positions") or range(e["n_pos"]))
        p0, p1 = min(pos), max(pos) + 1
        with safe_open(str(ACTS_MOUNT / sub / e["file"]), framework="pt", device="cpu") as f:
            x = f.get_slice("acts")[li : li + 1, p0:p1].float().numpy()[0]
        x = x[[p - p0 for p in pos]].astype(np.float32)
        return label, (x, pos, [e["n_pos"] - 1 - p for p in pos])

    t = time.time()
    with ThreadPoolExecutor(threads) as ex:
        out = dict(ex.map(one, labels))
    print(f"[um-probes L{layer}] {sub}: {len(labels)} files in {time.time() - t:.0f}s", flush=True)
    return out


N_PACK_SHARDS = 16


@app.function(
    image=image, volumes=volumes, cpu=8, memory=32768, timeout=4 * 3600, nonpreemptible=True
)
def pack_shard(shard: int, inputs: dict) -> dict:
    """Read every probe-text file of this shard ONCE at all layers (the volume fetches whole
    files, so one layer costs as much as twelve) and write per-layer fp16 window arrays."""
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    from safetensors import safe_open

    sub = inputs["probe_acts"]
    man = json.loads((ACTS_MOUNT / sub / "manifest.json").read_text())
    ent = {p["label"]: p for p in man["prompts"]}
    labels = [t["id"] for t in inputs["texts"]][shard::N_PACK_SHARDS]
    layers = [int(x) for x in inputs["layers"]]
    lis = [man["layers"].index(L) for L in layers]

    def one(label: str):
        e = ent[label]
        pos = list(e["eval_positions"])
        p0, p1 = min(pos), max(pos) + 1
        with safe_open(str(ACTS_MOUNT / sub / e["file"]), framework="pt", device="cpu") as f:
            x = f.get_slice("acts")[min(lis) : max(lis) + 1, p0:p1].float().numpy()
        x = x[[li - min(lis) for li in lis]][:, [p - p0 for p in pos]].astype(np.float16)
        return x, [e["n_pos"] - 1 - p for p in pos]

    t = time.time()
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(one, labels))
    d = OUT_MOUNT / inputs["run"] / "pack"
    d.mkdir(parents=True, exist_ok=True)
    for k, L in enumerate(layers):
        np.save(d / f"L{L:02d}_s{shard:02d}.npy", np.stack([r[0][k] for r in res]))
    (d / f"s{shard:02d}.json").write_text(
        json.dumps({"labels": labels, "offsets": [r[1] for r in res]})
    )
    out_vol.commit()
    return {"shard": shard, "n": len(labels), "read_s": round(time.time() - t, 1)}


def _textset(inputs: dict, layer: int):
    import numpy as np
    import sibprobes as P  # type: ignore

    texts = inputs["texts"]
    d = OUT_MOUNT / inputs.get("pack_run", inputs["run"]) / "pack"
    win: dict[str, tuple] = {}
    for sh in range(N_PACK_SHARDS):
        meta = json.loads((d / f"s{sh:02d}.json").read_text())
        arr = np.load(d / f"L{layer:02d}_s{sh:02d}.npy")
        for i, (lab, off) in enumerate(zip(meta["labels"], meta["offsets"], strict=True)):
            win[lab] = (arr[i].astype(np.float32), off)
    missing = [t["id"] for t in texts if t["id"] not in win]
    if missing:
        raise RuntimeError(f"pack lacks {len(missing)} texts, e.g. {missing[:3]}")
    W = [win[t["id"]][0] for t in texts]
    offs = [win[t["id"]][1] for t in texts]
    return P.TextSet.build(
        W,
        [t["id"] for t in texts],
        [t["c"] for t in texts],
        [t["kind"] for t in texts],
        [t["request_idx"] for t in texts],
        [t["fold"] for t in texts],
        offsets=offs,
    )


@app.function(
    image=image, volumes=volumes, cpu=16, memory=98304, timeout=6 * 3600, nonpreemptible=True
)
def fit_layer(layer: int, inputs: dict) -> dict:
    import numpy as np
    import sibprobes as P  # type: ignore
    import um_probe as Q  # type: ignore

    t0 = time.time()
    out_vol.reload()
    C = len(inputs["concepts"])
    ts = _textset(inputs, layer)
    print(f"[um-probes L{layer}] textset {ts.H.shape} in {time.time() - t0:.0f}s", flush=True)
    cfa, lams, _tab = P.cross_fit_whitened(ts, C)
    ta = time.time() - t0
    cfb = Q.cross_fit_logreg(ts, C)
    tb = time.time() - t0 - ta
    nsa, nsb = Q.null_stats(ts, cfa.oof, C), Q.null_stats(ts, cfb.oof, C)
    za, zb = Q.oof_z(ts, cfa.oof, nsa), Q.oof_z(ts, cfb.oof, nsb)
    va, vb = Q.validity_implied(ts, za, C), Q.validity_implied(ts, zb, C)
    # neutral FPR at the 95th pct of the other concepts' implied z (diagnostic)
    rc, imp = ts.row_concept, ts.row_implied
    fpr = {}
    for name, Z in (("a", za), ("b", zb)):
        thr = [np.nanquantile(Z[imp & (rc != c) & (rc >= 0), c], 0.95) for c in range(C)]
        fpr[name] = [float(np.nanmean(Z[rc == P.NEUTRAL, c] > thr[c])) for c in range(C)]
    cent = Q.implied_centroids(ts, C)
    arrs: dict[str, np.ndarray] = {}
    for k, m in enumerate(cfa.models):
        keys = np.array(sorted(m.keys, key=lambda kk: m.keys[kk]), dtype=np.int64)
        arrs |= {f"a{k}_U": m.U, f"a{k}_muU": m.muU, f"a{k}_sigma": m.sigma, f"a{k}_keys": keys}
    for k, m in enumerate(cfb.models):
        arrs |= {
            f"b{k}_{f}": getattr(m, f)
            for f in ("offsets", "mu", "mean", "scale", "pca_mean", "comp", "coef", "intercept")
        }
        arrs[f"b{k}_C"] = np.array(m.C_reg)
    for name, ns in (("na", nsa), ("nb", nsb)):
        arrs[f"{name}_offsets"] = ns.offsets
        for k in range(len(ns.m)):
            arrs[f"{name}{k}_m"], arrs[f"{name}{k}_sd"] = ns.m[k], ns.sd[k]
    run_dir = OUT_MOUNT / inputs["run"] / "fit"
    run_dir.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez(buf, **arrs)
    (run_dir / f"L{layer:02d}.npz").write_bytes(buf.getvalue())
    out_vol.commit()
    cbuf = io.BytesIO()
    np.save(cbuf, cent.astype(np.float32))
    (run_dir / f"centroids_L{layer:02d}.npy").write_bytes(cbuf.getvalue())
    res = {
        "layer": layer,
        "validity": {"a": [float(x) for x in va], "b": [float(x) for x in vb]},
        "neutral_fpr": fpr,
        "lambda_per_fold": lams,
        "C_per_fold": cfb.C_per_fold,
        "n_texts": len(ts.ids),
        "n_rows": int(ts.H.shape[0]),
        "timings": {
            "a_s": round(ta, 1),
            "b_s": round(tb, 1),
            "total_s": round(time.time() - t0, 1),
        },
    }
    (run_dir / f"fit_L{layer:02d}.json").write_text(json.dumps(res))
    out_vol.commit()
    return {**res, "centroids_npy": cbuf.getvalue()}


@app.function(image=image, volumes=volumes, cpu=8, memory=32768, timeout=3600, nonpreemptible=True)
def score_layer(layer: int, inputs: dict) -> bytes:
    import numpy as np
    import sibprobes as P  # type: ignore
    import um_probe as Q  # type: ignore

    out_vol.reload()
    z = np.load(io.BytesIO((OUT_MOUNT / inputs["run"] / "fit" / f"L{layer:02d}.npz").read_bytes()))
    nf = P.N_FOLDS
    ma = [
        P.FoldModel(
            D=z[f"a{k}_U"],
            U=z[f"a{k}_U"],
            keys={(int(e), int(o)): i for i, (e, o) in enumerate(z[f"a{k}_keys"])},
            muU=z[f"a{k}_muU"],
            sigma=z[f"a{k}_sigma"],
        )
        for k in range(nf)
    ]
    mb = [
        Q.LogregModel(
            *(
                z[f"b{k}_{f}"]
                for f in ("offsets", "mu", "mean", "scale", "pca_mean", "comp", "coef", "intercept")
            ),
            C_reg=float(z[f"b{k}_C"]),
        )
        for k in range(nf)
    ]
    ns = {
        name: Q.NullStats(
            z[f"{name}_offsets"],
            [z[f"{name}{k}_m"] for k in range(nf)],
            [z[f"{name}{k}_sd"] for k in range(nf)],
        )
        for name in ("na", "nb")
    }
    sub = inputs["eval_acts"][str(layer)]
    items = inputs["items"]
    win = _load_windows(sub, items, layer)
    rows_item, rows_pos, ZA, ZB = [], [], [], []
    for it in items:
        x, pos, off = win[it]
        s = Q.score_eval(
            ma,
            mb,
            ns["na"],
            ns["nb"],
            x,
            np.full(len(pos), int(inputs.get("item_endings", {}).get(it, 0)), np.int64),
            np.asarray(off),
        )
        ZA.append(s["a"])
        ZB.append(s["b"])
        rows_item += [it] * len(pos)
        rows_pos += pos
    buf = io.BytesIO()
    np.savez(
        buf,
        item=np.array(rows_item),
        pos=np.array(rows_pos, np.int64),
        za=np.concatenate(ZA).astype(np.float32),
        zb=np.concatenate(ZB).astype(np.float32),
    )
    sd = OUT_MOUNT / inputs["run"] / "score"
    sd.mkdir(parents=True, exist_ok=True)
    (sd / f"L{layer:02d}.npz").write_bytes(buf.getvalue())
    out_vol.commit()
    return buf.getvalue()


@app.local_entrypoint()
def main(
    phase: str,
    inputs: str = "outputs/sibling/um/probe_inputs.json",
    out_dir: str = "outputs/sibling/um/probes",
    layers: str = "",
    distractors: str = "data/sibling_probe/um_country_distractors.json",
) -> None:
    inp = json.loads(Path(inputs).read_text())
    Ls = [int(x) for x in layers.split(",")] if layers else list(inp["layers"])
    out = Path(out_dir)
    t0 = time.time()
    if phase == "pack":
        for res in pack_shard.map(range(N_PACK_SHARDS), kwargs={"inputs": inp}):
            print(f"[um-probes] pack {res}", flush=True)
    elif phase == "fit":
        d = out / "fit"
        d.mkdir(parents=True, exist_ok=True)
        meta = []
        for res in fit_layer.map(Ls, kwargs={"inputs": inp}):
            (d / f"centroids_L{res['layer']:02d}.npy").write_bytes(res.pop("centroids_npy"))
            (d / f"fit_L{res['layer']:02d}.json").write_text(json.dumps(res))
            meta.append({"layer": res["layer"], **res["timings"]})
            print(f"[um-probes] fit L{res['layer']} {res['timings']}", flush=True)
        (d / "summary.json").write_text(
            json.dumps(
                {
                    "run": inp["run"],
                    "inputs_sha256": hashlib.sha256(Path(inputs).read_bytes()).hexdigest(),
                    "concepts": inp["concepts"],
                    "layers": Ls,
                    "meta": meta,
                    "wall_s": round(time.time() - t0, 1),
                },
                indent=1,
            )
        )
    elif phase == "score":
        dist = Path(distractors)
        if not dist.exists():
            raise SystemExit(f"{dist} missing: run `um_ladder.py distractors` after the fit phase")
        d = out / "score"
        d.mkdir(parents=True, exist_ok=True)
        # recorded BEFORE scoring (survives a client drop): the distractor file D_i was frozen at
        summ = {
            "run": inp["run"],
            "distractors_sha256": hashlib.sha256(dist.read_bytes()).hexdigest(),
            "layers": Ls,
            "started": time.time(),
        }
        (d / "summary.json").write_text(json.dumps(summ, indent=1))
        for L, data in zip(Ls, score_layer.map(Ls, kwargs={"inputs": inp}), strict=True):
            (d / f"L{L:02d}.npz").write_bytes(data)
            print(f"[um-probes] score L{L}", flush=True)
        (d / "summary.json").write_text(
            json.dumps(
                {
                    "run": inp["run"],
                    "distractors_sha256": hashlib.sha256(dist.read_bytes()).hexdigest(),
                    "layers": Ls,
                    "wall_s": round(time.time() - t0, 1),
                },
                indent=1,
            )
        )
    else:
        raise SystemExit("--phase pack | fit | score")
    os.sync()
