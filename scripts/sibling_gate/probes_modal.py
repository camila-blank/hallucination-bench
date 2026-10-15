"""Plan 0009 §3c-d on Modal (CPU), adapted from hb-um-ladder ``um_probes_modal.py`` (fit/score).

Type (a) whitened DiffMean only (``probes.cross_fit_whitened``, lambda by inner CV; 4 folds by
text id from ``probes.assign_folds``), per class of ``data/sibling_gate/concepts.json``. **New
recipe:** centring and the null are keyed by the read position's token class (``texts.token_class``;
``ending = 0``, ``offset = key``), not by plan 0006's (ending, offset). Rev 8.1: the raw class ->
key merge is per class and fixed across layers (``capture-manifest`` writes it; a key class with
< 400 implied-text null positions merges into ``other``); fit uses the inputs' ``merge`` as is.

* ``--phase pack``: read every probe-text file of ``acts-sibgate-probe`` once at all 12 layers
  and write per-layer fp16 window arrays to ``sibling-probes:/<run>/pack/``.
* ``--phase fit``: one container per layer; per class: cross-fit, per-fold null stats, OOF
  implied validity per concept, the null check per key (split-half on held-out folds: the null
  z of non-target concepts' implied positions has >= 400 values, |mean| <= 0.1 and
  0.8 <= sd <= 1.25). Rev 8: a failing key does not take down the class; ``ladder`` makes only
  the units whose eval-site key fails ``probe_na`` (``null_fail``) at that layer. At L44,
  the grand-mean-centred implied centroids. -> ``--out-dir/fit/<class>/fit_L<L>.json``.
  **No eval residual is read.**
* ``--phase score``: per layer, the frozen fold models score every wsbench_chat eval site
  (``acts-hallucination-chat`` for 20..60, ``acts-hallucination-chat-L42`` for 42) -> per class
  ``--out-dir/score/<class>/L<L>.npz`` (``item``, ``pos``, ``za`` [P, C], ``key`` [P] the
  merged site key). The run records the
  sha256 of the distractor file it was launched after; ``hbench sibling-gate label`` refuses a
  mismatch.

    uvx --python 3.12 modal@latest run scripts/sibling_gate/probes_modal.py --phase fit \\
        --inputs outputs/sibling_gate/probes/probe_inputs.json --out-dir outputs/sibling_gate/probes
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
RUN_ROOT = "sibgate-v1"
_HERE = Path(__file__).resolve()
_SRC = (
    _HERE.parents[2] / "src" / "hallucination_bench" / "evals" / "sibling_gate"
    if len(_HERE.parents) > 2
    else _HERE
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch>=2.6", index_url="https://download.pytorch.org/whl/cpu")
    .uv_pip_install("numpy>=2.0", "safetensors>=0.5")
    .env({"PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "16", "OPENBLAS_NUM_THREADS": "16"})
)
if (_SRC / "probes.py").is_file():  # local launch; inside the container the files are at /root
    image = image.add_local_file(str(_SRC / "probes.py"), "/root/sibprobes.py")
    image = image.add_local_file(str(_SRC / "um_probe.py"), "/root/um_probe.py")

app = modal.App("sibling-gate-probes")
acts_vol = modal.Volume.from_name("wsbench-acts", create_if_missing=True)
out_vol = modal.Volume.from_name("sibling-probes", create_if_missing=True)
volumes: dict[str | PurePosixPath, modal.Volume] = {
    PurePosixPath(ACTS_MOUNT): acts_vol.read_only(),
    PurePosixPath(OUT_MOUNT): out_vol,
}

N_PACK_SHARDS = 16
OTHER = 13  # texts.TOKEN_CLASSES.index("other")
N_CLASSES = 14  # len(texts.TOKEN_CLASSES)
MIN_NULL_Z = 400  # a key with fewer null-z values fails the null check
NULL_MEAN_MAX = 0.1
NULL_SD = (0.8, 1.25)
DISTRACTOR_LAYER = 44


def _run_dir(inputs: dict) -> Path:
    return OUT_MOUNT / RUN_ROOT / inputs["run"]


def _all_texts(inputs: dict) -> list[dict]:
    return [t for c in inputs["classes"].values() for t in c["texts"]]


@app.function(
    image=image, volumes=volumes, cpu=8, memory=32768, timeout=4 * 3600, nonpreemptible=True
)
def pack_shard(shard: int, inputs: dict) -> dict:
    """Read every probe-text file of this shard ONCE at all layers; per-layer fp16 windows."""
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    from safetensors import safe_open

    sub = inputs["probe_acts"]
    man = json.loads((ACTS_MOUNT / sub / "manifest.json").read_text())
    ent = {p["label"]: p for p in man["prompts"]}
    labels = sorted({t["id"] for t in _all_texts(inputs)})[shard::N_PACK_SHARDS]
    layers = [int(x) for x in inputs["layers"]]
    lis = [man["layers"].index(L) for L in layers]

    def one(label: str):
        e = ent[label]
        pos = list(e["eval_positions"])
        if not pos:
            return np.zeros((len(lis), 0, 1), np.float16), []
        p0, p1 = min(pos), max(pos) + 1
        with safe_open(str(ACTS_MOUNT / sub / e["file"]), framework="pt", device="cpu") as f:
            x = f.get_slice("acts")[min(lis) : max(lis) + 1, p0:p1].float().numpy()
        x = x[[li - min(lis) for li in lis]][:, [p - p0 for p in pos]].astype(np.float16)
        return x, pos

    t = time.time()
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(one, labels))
    d = _run_dir(inputs) / "pack"
    d.mkdir(parents=True, exist_ok=True)
    for k, L in enumerate(layers):
        parts = [r[0][k] for r in res if r[1]]
        np.save(d / f"L{L:02d}_s{shard:02d}.npy", np.concatenate(parts) if parts else np.zeros(0))
    (d / f"s{shard:02d}.json").write_text(
        json.dumps({"labels": labels, "lengths": [len(r[1]) for r in res]})
    )
    out_vol.commit()
    return {"shard": shard, "n": len(labels), "read_s": round(time.time() - t, 1)}


def _windows(inputs: dict, layer: int) -> dict:
    import numpy as np

    d = _run_dir(inputs) / "pack"
    win: dict = {}
    for sh in range(N_PACK_SHARDS):
        meta = json.loads((d / f"s{sh:02d}.json").read_text())
        arr = np.load(d / f"L{layer:02d}_s{sh:02d}.npy")
        start = 0
        for lab, n in zip(meta["labels"], meta["lengths"], strict=True):
            win[lab] = arr[start : start + n].astype(np.float32)
            start += n
    return win


def _merge(cls: dict) -> dict[int, int]:
    """The class's raw token class -> key map (``capture-manifest``, fixed across layers). The
    texts' ``key`` arrays are already in this key space; old inputs without ``merge``: identity."""
    m = cls.get("merge") or list(range(N_CLASSES))
    return {c: int(m[c]) for c in range(N_CLASSES)}


def _textset(texts: list[dict], win: dict, merge: dict[int, int]):
    import sibprobes as P  # type: ignore

    texts = [t for t in texts if t["id"] in win and len(win[t["id"]])]
    return texts, P.TextSet.build(
        [win[t["id"]] for t in texts],
        [t["id"] for t in texts],
        [t["c"] for t in texts],
        [t["kind"] for t in texts],
        [0] * len(texts),  # ending 0: the key is the token class alone
        [t["fold"] for t in texts],
        offsets=[[merge[int(k)] for k in t["key"]] for t in texts],
    )


def null_check(ts, oof, n_concepts: int, n_folds: int = 4) -> dict:
    """Split-half on held-out folds, per key: a null fitted on half of fold k's non-target
    implied texts standardises the other half (and vice versa); the pooled z must have at least
    400 values, |mean| <= 0.1 and 0.8 <= sd <= 1.25. ``other_ok`` is informational (rev 8: the
    state that counts is per key)."""
    import numpy as np
    import um_probe as Q  # type: ignore

    key = Q._key(ts)
    rf, rc, imp = ts.row_fold, ts.row_concept, ts.row_implied
    half = ts.text % 2
    res: dict[str, dict] = {}
    for o in np.unique(key[rc >= 0]):
        zs = []
        for k in range(n_folds):
            for h in (0, 1):
                fit = (rf == k) & imp & (rc >= 0) & (key == o) & (half == h)
                ev = (rf == k) & imp & (rc >= 0) & (key == o) & (half != h)
                for c in range(n_concepts):
                    x = oof[fit & (rc != c), c]
                    x = x[np.isfinite(x)]
                    y = oof[ev & (rc != c), c]
                    y = y[np.isfinite(y)]
                    if x.size > 1 and y.size and x.std(ddof=1) > 0:
                        zs.append((y - x.mean()) / x.std(ddof=1))
        z = np.concatenate(zs) if zs else np.zeros(0)
        m, sd = (float(z.mean()), float(z.std(ddof=1))) if z.size > 1 else (np.nan, np.nan)
        ok = bool(
            z.size >= MIN_NULL_Z and abs(m) <= NULL_MEAN_MAX and NULL_SD[0] <= sd <= NULL_SD[1]
        )
        res[str(int(o))] = {"n": int(z.size), "mean": m, "sd": sd, "ok": ok}
    other = res.get(str(OTHER))
    failing = sorted(k for k, v in res.items() if not v["ok"])
    # no `other` positions: the class passes iff no key fails
    other_ok = bool(other["ok"]) if other else not failing
    return {"keys": res, "other_ok": other_ok, "failing_keys": failing}


@app.function(
    image=image, volumes=volumes, cpu=16, memory=98304, timeout=6 * 3600, nonpreemptible=True
)
def fit_layer(layer: int, inputs: dict) -> list[dict]:
    import numpy as np
    import sibprobes as P  # type: ignore
    import um_probe as Q  # type: ignore

    out_vol.reload()
    win = _windows(inputs, layer)
    results = []
    for slug, cls in sorted(inputs["classes"].items()):
        t0 = time.time()
        C = len(cls["concepts"])
        merge = _merge(cls)
        texts, ts = _textset(cls["texts"], win, merge)
        res: dict = {"layer": layer, "class": cls["class"], "n_texts": len(texts)}
        try:
            cfa, lams, _tab = P.cross_fit_whitened(ts, C)
        except ValueError as e:  # a concept without training texts
            res |= {
                "error": str(e),
                "validity": {"a": [None] * C},
                "null_check": {"other_ok": False},
            }
            results.append({**res, "slug": slug})
            continue
        nsa = Q.null_stats(ts, cfa.oof, C)
        za = Q.oof_z(ts, cfa.oof, nsa)
        va = Q.validity_implied(ts, za, C)
        nc = null_check(ts, cfa.oof, C)
        # rev 8: no refit / merge of failing keys; only units on a failing key are n/a there
        arrs: dict[str, np.ndarray] = {}
        for k, m in enumerate(cfa.models):
            keys = np.array(sorted(m.keys, key=lambda kk: m.keys[kk]), dtype=np.int64)
            arrs |= {f"a{k}_U": m.U, f"a{k}_muU": m.muU, f"a{k}_sigma": m.sigma, f"a{k}_keys": keys}
        arrs["na_offsets"] = nsa.offsets
        for k in range(len(nsa.m)):
            arrs[f"na{k}_m"], arrs[f"na{k}_sd"] = nsa.m[k], nsa.sd[k]
        arrs["merge"] = np.array([merge[c] for c in range(N_CLASSES)], dtype=np.int64)
        d = _run_dir(inputs) / "fit" / slug
        d.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        np.savez(buf, **arrs)
        (d / f"L{layer:02d}.npz").write_bytes(buf.getvalue())
        res |= {
            "validity": {"a": [None if np.isnan(x) else float(x) for x in va]},
            "lambda_per_fold": lams,
            "null_check": nc,
            "merge": merge,
            "n_rows": int(ts.H.shape[0]),
            "seconds": round(time.time() - t0, 1),
        }
        if layer == DISTRACTOR_LAYER:
            cent = Q.implied_centroids(ts, C)
            res["centroids"] = {
                c: [float(x) for x in cent[i]] for i, c in enumerate(cls["concepts"])
            }
        (d / f"fit_L{layer:02d}.json").write_text(json.dumps(res))
        results.append({**res, "slug": slug})
        print(f"[sibgate-probes L{layer}] {cls['class']}: {res['seconds']}s", flush=True)
    out_vol.commit()
    return results


def _load_models(inputs: dict, slug: str, layer: int):
    import numpy as np
    import sibprobes as P  # type: ignore
    import um_probe as Q  # type: ignore

    p = _run_dir(inputs) / "fit" / slug / f"L{layer:02d}.npz"
    if not p.exists():
        return None
    z = np.load(io.BytesIO(p.read_bytes()))
    ma = [
        P.FoldModel(
            D=z[f"a{k}_U"],
            U=z[f"a{k}_U"],
            keys={(int(e), int(o)): i for i, (e, o) in enumerate(z[f"a{k}_keys"])},
            muU=z[f"a{k}_muU"],
            sigma=z[f"a{k}_sigma"],
        )
        for k in range(P.N_FOLDS)
    ]
    ns = Q.NullStats(
        z["na_offsets"],
        [z[f"na{k}_m"] for k in range(P.N_FOLDS)],
        [z[f"na{k}_sd"] for k in range(P.N_FOLDS)],
    )
    return ma, ns, {c: int(m) for c, m in enumerate(z["merge"])}


def _eval_windows(sub: str, layer: int, items: list[str]):
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    from safetensors import safe_open

    man = json.loads((ACTS_MOUNT / sub / "manifest.json").read_text())
    li = man["layers"].index(layer)
    ent = {p["label"]: p for p in man["prompts"]}

    def one(label: str):
        e = ent[label]
        pos = list(e["eval_positions"])
        p0, p1 = min(pos), max(pos) + 1
        # the L42 manifest points at acts-jlens-pr-L42 (gone from the volume); same acts live in
        # acts-hallucination-chat-L42cap (resample.DIR_REMAP)
        f_rel = e["file"].replace("/acts-jlens-pr-L42/", "/acts-hallucination-chat-L42cap/")
        path = Path(f_rel) if f_rel.startswith("/") else ACTS_MOUNT / sub / f_rel
        with safe_open(str(path), framework="pt", device="cpu") as f:
            x = f.get_slice("acts")[li : li + 1, p0:p1].float().numpy()[0]
        return label, (x[[p - p0 for p in pos]].astype(np.float32), pos)

    with ThreadPoolExecutor(64) as ex:
        return dict(ex.map(one, [i for i in items if i in ent]))


@app.function(image=image, volumes=volumes, cpu=8, memory=32768, timeout=3600, nonpreemptible=True)
def score_layer(layer: int, inputs: dict) -> dict[str, bytes]:
    import numpy as np

    out_vol.reload()
    win = _eval_windows(inputs["eval_acts"][str(layer)], layer, inputs["items"])
    out: dict[str, bytes] = {}
    for slug in inputs["classes"]:
        loaded = _load_models(inputs, slug, layer)
        if loaded is None:
            continue
        ma, ns, merge = loaded
        rows_item, rows_pos, rows_key, Z = [], [], [], []
        for it, (x, pos) in sorted(win.items()):
            keys = np.array(
                [merge.get(int(inputs["site_keys"][it].get(str(p), OTHER)), OTHER) for p in pos],
                dtype=np.int64,
            )
            ending = np.zeros(len(pos), np.int64)
            za = np.nanmean(
                [ns.z(k, m.score(x, ending, keys), keys) for k, m in enumerate(ma)], axis=0
            )
            Z.append(za)
            rows_item += [it] * len(pos)
            rows_pos += pos
            rows_key += keys.tolist()
        buf = io.BytesIO()
        np.savez(
            buf,
            item=np.array(rows_item),
            pos=np.array(rows_pos, np.int64),
            za=np.concatenate(Z).astype(np.float32),
            key=np.array(rows_key, np.int64),
        )
        d = _run_dir(inputs) / "score" / slug
        d.mkdir(parents=True, exist_ok=True)
        (d / f"L{layer:02d}.npz").write_bytes(buf.getvalue())
        out[slug] = buf.getvalue()
    out_vol.commit()
    return out


@app.local_entrypoint()
def main(
    phase: str,
    inputs: str = "outputs/sibling_gate/probes/probe_inputs.json",
    out_dir: str = "outputs/sibling_gate/probes",
    layers: str = "",
    distractors: str = "data/sibling_gate/distractors.json",
) -> None:
    inp = json.loads(Path(inputs).read_text())
    Ls = [int(x) for x in layers.split(",")] if layers else list(inp["layers"])
    out = Path(out_dir)
    t0 = time.time()
    if phase == "pack":
        for res in pack_shard.map(range(N_PACK_SHARDS), kwargs={"inputs": inp}):
            print(f"[sibgate-probes] pack {res}", flush=True)
    elif phase == "fit":
        for results in fit_layer.map(Ls, kwargs={"inputs": inp}):
            for res in results:
                d = out / "fit" / res.pop("slug")
                d.mkdir(parents=True, exist_ok=True)
                cent = res.pop("centroids", None)
                if cent is not None:
                    (d / f"centroids_L{DISTRACTOR_LAYER}.json").write_text(json.dumps(cent))
                (d / f"fit_L{res['layer']:02d}.json").write_text(json.dumps(res))
        (out / "fit" / "summary.json").write_text(
            json.dumps(
                {
                    "run": inp["run"],
                    "inputs_sha256": hashlib.sha256(Path(inputs).read_bytes()).hexdigest(),
                    "layers": Ls,
                    "wall_s": round(time.time() - t0, 1),
                },
                indent=1,
            )
        )
    elif phase == "score":
        dist = Path(distractors)
        if not dist.exists():
            raise SystemExit(f"{dist} missing: run `hbench sibling-gate score` (it freezes them)")
        sha = hashlib.sha256(dist.read_bytes()).hexdigest()
        (out / "score").mkdir(parents=True, exist_ok=True)
        summ = {"run": inp["run"], "distractors_sha256": sha, "layers": Ls, "started": time.time()}
        (out / "score" / "summary.json").write_text(json.dumps(summ, indent=1))
        for L, data in zip(Ls, score_layer.map(Ls, kwargs={"inputs": inp}), strict=True):
            for slug, b in data.items():
                d = out / "score" / slug
                d.mkdir(parents=True, exist_ok=True)
                (d / f"L{L:02d}.npz").write_bytes(b)
            print(f"[sibgate-probes] score L{L}", flush=True)
        (out / "score" / "summary.json").write_text(
            json.dumps({**summ, "wall_s": round(time.time() - t0, 1)}, indent=1)
        )
    else:
        raise SystemExit("--phase pack | fit | score")
    os.sync()
