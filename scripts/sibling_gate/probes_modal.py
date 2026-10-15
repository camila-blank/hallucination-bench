"""Plan 0009 §3c-d on Modal (CPU), adapted from hb-um-ladder ``um_probes_modal.py`` (fit/score).

Type (a) whitened DiffMean only (``probes.cross_fit_whitened``, lambda by inner CV; 4 folds by
text id from ``probes.assign_folds``), per class of ``data/sibling_gate/concepts.json``. **New
recipe:** centring and the null are keyed by the read position's token class (``texts.token_class``;
``ending = 0``, ``offset = key``), not by plan 0006's (ending, offset). Rev 8.1: the raw class ->
key merge is per class and fixed across layers (``capture-manifest`` writes it; a key class with
< 400 implied-text null positions merges into ``other``); fit uses the inputs' ``merge`` as is.

* ``--phase pack``: no-op since rev 8.2 (kept so older drivers still run). Every class
  container now reads its own texts' windows straight from ``acts-sibgate-probe``.
* ``--phase fit``: one container per CLASS (rev 8.2: streams per class, ~220 classes do not fit a
  per-layer container; core in ``probe_fit.py``), looping over the 12 layers; per class:
  cross-fit, per-fold null stats, OOF
  implied validity per concept, the null check per key (split-half on held-out folds: the null
  z of non-target concepts' implied positions has >= 400 values, |mean| <= 0.1 and
  0.8 <= sd <= 1.25). Rev 8: a failing key does not take down the class; ``ladder`` makes only
  the units whose eval-site key fails ``probe_na`` (``null_fail``) at that layer. At L44,
  the grand-mean-centred implied centroids. -> ``--out-dir/fit/<class>/fit_L<L>.json``.
  **No eval residual is read.**
* ``--phase score``: one container per (layer, chunk of classes); the frozen fold models score every
  wsbench_chat eval site
  (``acts-hallucination-chat`` for 20..60, ``acts-hallucination-chat-L42`` for 42) -> per class
  ``--out-dir/score/<class>/L<L>.npz`` (``item``, ``pos``, ``za`` [P, C], ``key`` [P] the
  merged site key). The run records the
  sha256 of the distractor file it was launched after; ``hbench sibling-gate label`` refuses a
  mismatch.

    uvx --python 3.12 modal@latest run scripts/sibling_gate/probes_modal.py --phase fit \\
        --inputs outputs/sibling_gate/probes/probe_inputs.json --out-dir outputs/sibling_gate/probes
"""

import hashlib
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
    image = image.add_local_file(str(_SRC / "probe_fit.py"), "/root/sibfit.py")

app = modal.App("sibling-gate-probes")
acts_vol = modal.Volume.from_name("wsbench-acts", create_if_missing=True)
out_vol = modal.Volume.from_name("sibling-probes", create_if_missing=True)
volumes: dict[str | PurePosixPath, modal.Volume] = {
    PurePosixPath(ACTS_MOUNT): acts_vol.read_only(),
    PurePosixPath(OUT_MOUNT): out_vol,
}

N_PACK_SHARDS = 16  # unused since rev 8.2 (pack removed)
OTHER = 13  # texts.TOKEN_CLASSES.index("other")
SCORE_CHUNK = 24  # classes per score container
DISTRACTOR_LAYER = 44


def _run_dir(inputs: dict) -> Path:
    return OUT_MOUNT / RUN_ROOT / inputs["run"]


@app.function(
    image=image, volumes=volumes, cpu=16, memory=65536, timeout=6 * 3600, nonpreemptible=True
)
def fit_class(inputs: dict) -> list[dict]:
    """``inputs`` = the run inputs with ``classes`` restricted to ONE class. Reads that class's
    texts' windows (all layers, fp16) once, then fits each layer in turn."""
    import sibfit as F  # type: ignore

    out_vol.reload()
    ((slug, cls),) = inputs["classes"].items()
    man = json.loads(
        (ACTS_MOUNT / inputs["probe_acts"].split(",")[0] / "manifest.json").read_text()
    )
    layers = [int(x) for x in inputs["layers"]]
    t0 = time.time()
    win16 = F.read_class_windows(
        ACTS_MOUNT, inputs["probe_acts"], cls["texts"], man["layers"], layers
    )
    read_s = round(time.time() - t0, 1)
    d = _run_dir(inputs) / "fit" / slug
    d.mkdir(parents=True, exist_ok=True)
    results = []
    for k, layer in enumerate(layers):
        res, blob = F.fit_class_layer(slug, cls, F.layer_windows(win16, k), layer)
        if blob is not None:
            (d / f"L{layer:02d}.npz").write_bytes(blob)
            (d / f"fit_L{layer:02d}.json").write_text(json.dumps(res))
        results.append({**res, "slug": slug, "read_s": read_s})
        print(f"[sibgate-probes] {cls['class']} L{layer}: {res.get('seconds')}s", flush=True)
    out_vol.commit()
    return results


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
def score_chunk(layer: int, slugs: list[str], inputs: dict) -> dict[str, bytes]:
    import sibfit as F  # type: ignore

    out_vol.reload()
    win = _eval_windows(inputs["eval_acts"][str(layer)], layer, inputs["items"])
    out: dict[str, bytes] = {}
    for slug in slugs:
        p = _run_dir(inputs) / "fit" / slug / f"L{layer:02d}.npz"
        if not p.exists():
            continue
        out[slug] = F.score_class_layer(p.read_bytes(), win, inputs["site_keys"])
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
        print("[sibgate-probes] pack is a no-op since rev 8.2 (per-class streaming in fit)")
    elif phase == "fit":
        per = [{**inp, "layers": Ls, "classes": {s: c}} for s, c in sorted(inp["classes"].items())]
        n = 0
        for results in fit_class.map(per):
            for res in results:
                d = out / "fit" / res["slug"]
                d.mkdir(parents=True, exist_ok=True)
                cent = res.pop("centroids", None)
                if cent is not None:
                    (d / f"centroids_L{DISTRACTOR_LAYER}.json").write_text(json.dumps(cent))
                (d / f"fit_L{res['layer']:02d}.json").write_text(json.dumps(res))
            n += 1
            if n % 10 == 0:
                print(f"[sibgate-probes] fit {n}/{len(per)} classes", flush=True)
        (out / "fit").mkdir(parents=True, exist_ok=True)
        (out / "fit" / "summary.json").write_text(
            json.dumps(
                {
                    "run": inp["run"],
                    "inputs_sha256": hashlib.sha256(Path(inputs).read_bytes()).hexdigest(),
                    "layers": Ls,
                    "n_classes": len(per),
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
        slugs = sorted(inp["classes"])
        jobs = [
            (L, slugs[i : i + SCORE_CHUNK]) for L in Ls for i in range(0, len(slugs), SCORE_CHUNK)
        ]
        ls, ch = zip(*jobs, strict=True)
        for L, data in zip(ls, score_chunk.map(ls, ch, kwargs={"inputs": inp}), strict=True):
            for slug, b in data.items():
                d = out / "score" / slug
                d.mkdir(parents=True, exist_ok=True)
                (d / f"L{L:02d}.npz").write_bytes(b)
            print(f"[sibgate-probes] score L{L} ({len(data)} classes)", flush=True)
        (out / "score" / "summary.json").write_text(
            json.dumps({**summ, "wall_s": round(time.time() - t0, 1)}, indent=1)
        )
    else:
        raise SystemExit("--phase pack | fit | score")
    os.sync()
