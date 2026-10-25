"""Plan 0006 §4 probe job for the V4 port: type (a) whitened DiffMean (``probes.py``, unchanged),
per-fold null z, implied validity and centroids (``um_probe.py``, unchanged) on DeepSeek-V4-Flash
residuals (block-L output, 4 mHC streams flattened = 16,384-d).

Differences from ``scripts/sibling_probe/um_probes_modal.py`` (whose phase logic this copies):

* inputs: probe texts from the ``um-v4`` volume's packs (``/pack/<domain>/<tag><shard>.json`` +
  ``<tag><shard>_L<L>.safetensors``, written by ``gw-v4-flash-ar/scripts/v4_flash/um_v4_modal.py
  ::probe``), eval cells from ``/eval_acts/`` (that file's ``items``; the same states the V4 lens
  readouts were made from);
* S_w and its eigendecomposition run on one GPU in float64 (``Whitener.eig`` replaced by an
  identical-math torch version: 16,384-d makes the CPU eigh the bottleneck);
* **type (a) only** (the headline type since plan 0006 rev 8): type (b) logistic regression is not
  fitted; its validity is written as 0 and its z as NaN, so every type-(b) sensitivity row is n/a.

    MODAL_PROFILE=safety-sahan-2 uvx --python 3.12 modal@latest run \\
        scripts/v4_um/um_probes_v4_modal.py --phase fit \\
        --inputs outputs/v4/um/holiday/probe_inputs.json \\
        --out-dir outputs/v4/um/holiday/probes --domain holiday
"""

import hashlib
import io
import json
import os
import time
from pathlib import Path, PurePosixPath

import modal

UM = Path("/um")
_HERE = Path(__file__).resolve()
_SRC = (
    _HERE.parents[2] / "src" / "hallucination_bench" / "evals" / "sibling_probe"
    if len(_HERE.parents) > 2
    else _HERE
)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch>=2.6", "numpy>=2.0", "safetensors>=0.5", "scikit-learn>=1.5")
    .env({"PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "16", "OPENBLAS_NUM_THREADS": "16"})
)
if (_SRC / "probes.py").is_file():
    image = image.add_local_file(str(_SRC / "probes.py"), "/root/sibprobes.py")
    image = image.add_local_file(str(_SRC / "um_probe.py"), "/root/um_probe.py")

app = modal.App("um-v4-probes")
um_vol = modal.Volume.from_name("um-v4", create_if_missing=True)
volumes: dict[str | PurePosixPath, modal.Volume] = {PurePosixPath(UM): um_vol}


def _gpu_eig(self, folds):  # = probes.Whitener.eig, on the GPU in float64
    import numpy as np
    import torch

    folds = frozenset(folds)
    if folds not in self._eig:
        ts, C, K = self.ts, self.C, self.K
        T = self.train_texts(folds)
        intr = np.zeros(len(ts.ids), bool)
        intr[T] = True
        rows = np.flatnonzero(intr[ts.text])
        dev = torch.device("cuda")
        Hr = torch.from_numpy(ts.H[rows]).to(dev, torch.float64)
        gid = torch.from_numpy(self.rkey[rows] * C + ts.row_concept[rows]).to(dev)
        m = torch.zeros((K * C, Hr.shape[1]), dtype=torch.float64, device=dev)
        m.index_add_(0, gid, Hr)
        cnt = torch.bincount(gid, minlength=K * C).to(torch.float64)
        m /= cnt.clamp_min(1)[:, None]
        Hr -= m[gid]
        del m
        S = (Hr.T @ Hr) / max(len(rows) - K * C, 1)
        del Hr
        E, V = torch.linalg.eigh(S)
        del S
        self._eig[folds] = (np.maximum(E.cpu().numpy(), 0), V.cpu().numpy())
        torch.cuda.empty_cache()
    return self._eig[folds]


def _pack_windows(domain: str, layer: int, want: set[str]):
    """{label: (x [n, d] float32, offsets)} from every pack shard of the domain."""
    from safetensors import safe_open

    d = UM / "pack" / domain
    out = {}
    for mp in sorted(d.glob("*.json")):
        meta = json.loads(mp.read_text())
        if not set(meta["labels"]) & want:
            continue
        with safe_open(str(d / f"{mp.stem}_L{layer:02d}.safetensors"), framework="pt") as f:
            x = f.get_tensor("x").float().numpy()
        s = 0
        for lab, off, n in zip(meta["labels"], meta["offsets"], meta["lengths"], strict=True):
            if lab in want:
                out[lab] = (x[s : s + n], off)
            s += n
    return out


def _textset(inputs: dict, layer: int):
    import sibprobes as P  # type: ignore

    texts = inputs["texts"]
    win = _pack_windows(inputs["domain"], layer, {t["id"] for t in texts})
    missing = [t["id"] for t in texts if t["id"] not in win]
    if missing:
        raise RuntimeError(f"pack lacks {len(missing)} texts, e.g. {missing[:3]}")
    return P.TextSet.build(
        [win[t["id"]][0] for t in texts],
        [t["id"] for t in texts],
        [t["c"] for t in texts],
        [t["kind"] for t in texts],
        [t["request_idx"] for t in texts],
        [t["fold"] for t in texts],
        offsets=[win[t["id"]][1] for t in texts],
    )


@app.function(
    image=image,
    volumes=volumes,
    gpu=["H100", "A100-80GB"],
    cpu=16,
    memory=160 * 1024,
    timeout=8 * 3600,
)
def fit_layer(layer: int, inputs: dict) -> dict:
    import numpy as np
    import sibprobes as P  # type: ignore
    import um_probe as Q  # type: ignore

    P.Whitener.eig = _gpu_eig
    t0 = time.time()
    um_vol.reload()
    C = len(inputs["concepts"])
    ts = _textset(inputs, layer)
    print(
        f"[v4-probes {inputs['domain']} L{layer}] textset {ts.H.shape} in {time.time() - t0:.0f}s",
        flush=True,
    )
    cfa, lams, _tab = P.cross_fit_whitened(ts, C)
    ta = time.time() - t0
    nsa = Q.null_stats(ts, cfa.oof, C)
    za = Q.oof_z(ts, cfa.oof, nsa)
    va = Q.validity_implied(ts, za, C)
    rc, imp = ts.row_concept, ts.row_implied
    thr = [np.nanquantile(za[imp & (rc != c) & (rc >= 0), c], 0.95) for c in range(C)]
    fpr = {"a": [float(np.nanmean(za[rc == P.NEUTRAL, c] > thr[c])) for c in range(C)]}
    cent = Q.implied_centroids(ts, C)
    arrs: dict[str, np.ndarray] = {}
    for k, m in enumerate(cfa.models):
        keys = np.array(sorted(m.keys, key=lambda kk: m.keys[kk]), dtype=np.int64)
        arrs |= {f"a{k}_U": m.U, f"a{k}_muU": m.muU, f"a{k}_sigma": m.sigma, f"a{k}_keys": keys}
    arrs["na_offsets"] = nsa.offsets
    for k in range(len(nsa.m)):
        arrs[f"na{k}_m"], arrs[f"na{k}_sd"] = nsa.m[k], nsa.sd[k]
    run_dir = UM / "probes" / inputs["run"] / "fit"
    run_dir.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez(buf, **arrs)
    (run_dir / f"L{layer:02d}.npz").write_bytes(buf.getvalue())
    cbuf = io.BytesIO()
    np.save(cbuf, cent.astype(np.float32))
    (run_dir / f"centroids_L{layer:02d}.npy").write_bytes(cbuf.getvalue())
    res = {
        "layer": layer,
        "validity": {"a": [float(x) for x in va], "b": [0.0] * C},
        "types_fitted": ["a"],
        "neutral_fpr": fpr,
        "lambda_per_fold": lams,
        "n_texts": len(ts.ids),
        "n_rows": int(ts.H.shape[0]),
        "d": int(ts.H.shape[1]),
        "timings": {"a_s": round(ta, 1), "total_s": round(time.time() - t0, 1)},
    }
    (run_dir / f"fit_L{layer:02d}.json").write_text(json.dumps(res))
    um_vol.commit()
    return {**res, "centroids_npy": cbuf.getvalue()}


@app.function(image=image, volumes=volumes, cpu=8, memory=64 * 1024, timeout=3600)
def score_layer(layer: int, inputs: dict) -> bytes:
    import numpy as np
    import sibprobes as P  # type: ignore
    import um_probe as Q  # type: ignore
    from safetensors import safe_open

    um_vol.reload()
    z = np.load(
        io.BytesIO((UM / "probes" / inputs["run"] / "fit" / f"L{layer:02d}.npz").read_bytes())
    )
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
    ns = Q.NullStats(
        z["na_offsets"], [z[f"na{k}_m"] for k in range(nf)], [z[f"na{k}_sd"] for k in range(nf)]
    )
    man = json.loads((UM / "eval_acts" / "manifest.json").read_text())
    li = man["layers"].index(layer)
    ent = {p["label"]: p for p in man["prompts"]}
    rows_item, rows_pos, ZA = [], [], []
    for it in inputs["items"]:
        e = ent[it]
        with safe_open(str(UM / e["file"]), framework="pt") as f:
            x = f.get_tensor("acts")[li].float().numpy()
        pos = e["eval_positions"]
        off = np.asarray([e["n_pos"] - 1 - p for p in pos])
        ending = np.full(len(pos), int(inputs["item_endings"][it]), np.int64)
        key = Q.eval_key(ending, off)
        za = np.nanmean([ns.z(k, m.score(x, ending, off), key) for k, m in enumerate(ma)], axis=0)
        ZA.append(za)
        rows_item += [it] * len(pos)
        rows_pos += list(pos)
    za = np.concatenate(ZA).astype(np.float32)
    buf = io.BytesIO()
    np.savez(
        buf,
        item=np.array(rows_item),
        pos=np.array(rows_pos, np.int64),
        za=za,
        zb=np.full_like(za, np.nan),
    )
    sd = UM / "probes" / inputs["run"] / "score"
    sd.mkdir(parents=True, exist_ok=True)
    (sd / f"L{layer:02d}.npz").write_bytes(buf.getvalue())
    um_vol.commit()
    return buf.getvalue()


@app.local_entrypoint()
def main(
    phase: str, inputs: str, out_dir: str, domain: str, layers: str = "", distractors: str = ""
) -> None:
    inp = json.loads(Path(inputs).read_text())
    inp["domain"] = domain
    Ls = [int(x) for x in layers.split(",")] if layers else list(inp["layers"])
    out = Path(out_dir)
    t0 = time.time()
    if phase == "fit":
        d = out / "fit"
        d.mkdir(parents=True, exist_ok=True)
        meta = []
        for res in fit_layer.map(Ls, kwargs={"inputs": inp}, return_exceptions=True):
            if isinstance(res, Exception):
                print(f"[v4-probes] fit FAILED: {res!r}", flush=True)
                continue
            (d / f"centroids_L{res['layer']:02d}.npy").write_bytes(res.pop("centroids_npy"))
            (d / f"fit_L{res['layer']:02d}.json").write_text(json.dumps(res))
            meta.append({"layer": res["layer"], **res["timings"]})
            print(f"[v4-probes] fit L{res['layer']} {res['timings']}", flush=True)
        (d / "summary.json").write_text(
            json.dumps(
                {
                    "run": inp["run"],
                    "inputs_sha256": hashlib.sha256(Path(inputs).read_bytes()).hexdigest(),
                    "concepts": inp["concepts"],
                    "layers": Ls,
                    "meta": meta,
                    "types": ["a"],
                    "wall_s": round(time.time() - t0, 1),
                },
                indent=1,
            )
        )
    elif phase == "score":
        dist = Path(distractors)
        if not dist.exists():
            raise SystemExit(f"{dist} missing: run `distractors` after the fit phase")
        d = out / "score"
        d.mkdir(parents=True, exist_ok=True)
        summ = {
            "run": inp["run"],
            "distractors_sha256": hashlib.sha256(dist.read_bytes()).hexdigest(),
            "layers": Ls,
            "started": time.time(),
        }
        (d / "summary.json").write_text(json.dumps(summ, indent=1))
        for L, data in zip(Ls, score_layer.map(Ls, kwargs={"inputs": inp}), strict=True):
            (d / f"L{L:02d}.npz").write_bytes(data)
            print(f"[v4-probes] score L{L}", flush=True)
        summ["wall_s"] = round(time.time() - t0, 1)
        (d / "summary.json").write_text(json.dumps(summ, indent=1))
    else:
        raise SystemExit("--phase fit | score")
    os.sync()
