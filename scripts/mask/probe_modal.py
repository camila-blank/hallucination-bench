# ruff: noqa  (Modal launcher)
"""Plan 0017 rev 4: fit the deception probes and score every read site, one Modal container per
layer (CPU; reads only the needed positions from the captures on ``wsbench-acts``).

Per layer: the Instructed-Pairs probes ``resp`` (response tokens) and ``end`` (prompt-end
template positions) from ``hb-mask-probe-train``; read sites from ``hb-mask-pilot`` scored with
``resp`` (response sites) / ``end`` (prompt sites); the on-distribution cross-fit probe per site
type on MASK items (read + ``hb-mask-probe-extra`` lies; 4 item folds; read items scored
out-of-fold only). Writes ``data/mask/probe_scores.json``.

    MODAL_RESOURCE_ENV=camila-blank uvx --python 3.12 modal@latest run scripts/mask/probe_modal.py
"""

import json
import zlib
from pathlib import Path

import modal

_HERE = Path(__file__).resolve()
# in the container the launcher is copied to /root/probe_modal.py (no repo layout)
REPO = (
    _HERE.parents[2]
    if len(_HERE.parents) > 2 and (_HERE.parents[2] / "src").is_dir()
    else Path("/root")
)
LAYERS = (28, 36, 44, 52)
app = modal.App("hb-mask-probe")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "numpy>=1.26",
        "safetensors>=0.4",
        "torch>=2.4",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .env({"PYTHONPATH": "/root/src", "OMP_NUM_THREADS": "16", "OPENBLAS_NUM_THREADS": "16"})
    .add_local_dir(str(REPO / "src"), "/root/src", ignore=["**/__pycache__"])
)
acts_vol = modal.Volume.from_name("wsbench-acts")


def _load(subdir: str, wanted: dict[str, list[int]], layer: int) -> dict[str, "object"]:
    """{label: [len(positions), d] float32} at ``layer`` for the wanted positions."""
    import numpy as np
    from safetensors import safe_open

    root = Path("/acts") / subdir
    man = json.loads((root / "manifest.json").read_text())
    li = list(man["layers"]).index(layer)
    out = {}
    for e in man["prompts"]:
        if e["label"] not in wanted:
            continue
        with safe_open(str(root / e["file"]), framework="pt", device="cpu") as f:
            a = f.get_slice("acts")[li : li + 1][0]
        out[e["label"]] = a[wanted[e["label"]]].float().numpy().astype(np.float32)
    missing = set(wanted) - set(out)
    assert not missing, f"{subdir}: missing {sorted(missing)[:3]}"
    return out


@app.function(image=image, volumes={"/acts": acts_vol}, cpu=16, memory=65536, timeout=4 * 3600)
def run_layer(layer: int, manifest: dict, sites: dict, labels: dict) -> dict:
    import numpy as np

    from hallucination_bench.evals.mask import probe as PR

    out: dict = {"layer": layer}
    # ---- Instructed-Pairs probes
    tr = manifest["train"]
    acts = _load("hb-mask-probe-train", {t["label"]: t["end"] + t["resp"] for t in tr}, layer)
    probes = {}
    for kind in ("end", "resp"):
        win = []
        for t in tr:
            a = acts[t["label"]]
            win.append(a[: len(t["end"])] if kind == "end" else a[len(t["end"]) :])
        y = [int(t["cond"] == "deceptive") for t in tr]
        f = [int(t["fold"]) for t in tr]
        probes[kind] = PR.fit(win, y, f, ids=[t["label"] for t in tr])
        out[f"ip_{kind}"] = probes[kind].summary()
        print(layer, kind, out[f"ip_{kind}"], flush=True)
    del acts
    # ---- read + extra MASK sites
    read = _load("hb-mask-pilot", {t: [s["pos"] for s in ss] for t, ss in sites.items()}, layer)
    extra_sites = {e["task_id"]: e["sites"] for e in manifest["extra"]}
    extra = _load(
        "hb-mask-probe-extra", {t: [s["pos"] for s in ss] for t, ss in extra_sites.items()}, layer
    )
    ip_scores = {}
    for t, ss in sites.items():
        a = read[t]
        z = np.full(len(ss), np.nan)
        for kind, st in (("end", "prompt"), ("resp", "response")):
            idx = [i for i, s in enumerate(ss) if s["site_type"] == st]
            if idx:
                z[idx] = probes[kind].score(a[idx])
        ip_scores[t] = [float(x) for x in z]
    out["ip_scores"] = ip_scores
    # ---- on-distribution cross-fit probe per site type (item folds)
    allsites = {**sites, **extra_sites}
    allacts = {**read, **extra}
    cf_scores = {t: [float("nan")] * len(ss) for t, ss in sites.items()}
    for st in ("prompt", "response"):
        ids, win, y, f, idxs = [], [], [], [], []
        for t, ss in sorted(allsites.items()):
            lab = labels[t]
            if lab != "lie" and not lab.startswith("honest"):
                continue
            idx = [i for i, s in enumerate(ss) if s["site_type"] == st]
            if not idx:
                continue
            ids.append(t)
            win.append(allacts[t][idx])
            y.append(int(lab == "lie"))
            f.append(zlib.crc32(t.encode()) % 4)
            idxs.append(idx)
        p = PR.fit(win, y, f, ids=ids)
        out[f"cf_{st}"] = p.summary()
        print(layer, "cf", st, out[f"cf_{st}"], flush=True)
        rows_of = p.ts.text
        for j, t in enumerate(ids):
            if t in cf_scores:
                zz = p.oof_z[rows_of == j]
                for k, i in enumerate(idxs[j]):
                    cf_scores[t][i] = float(zz[k])
    out["cf_scores"] = cf_scores
    return out


@app.local_entrypoint()
def main(out: str = "data/mask/probe_scores.json") -> None:
    d = REPO / "data" / "mask"
    manifest = json.loads((d / "probe_manifest.json").read_text())
    sites = {
        r["task_id"]: [{"pos": s["pos"], "site_type": s["site_type"]} for s in r["sites"]]
        for r in json.loads((d / "sites.json").read_text())["items"]
    }
    labels = {
        r["task_id"]: r["label"] for r in json.loads((d / "labels.json").read_text())["items"]
    }
    res = {}
    for r in run_layer.map(LAYERS, kwargs={"manifest": manifest, "sites": sites, "labels": labels}):
        res[str(r["layer"])] = r
        print(f"[probe] layer {r['layer']} done", flush=True)
    meta = {
        "plan": "0017 rev 4",
        "layers": list(LAYERS),
        "manifest_meta": manifest["meta"],
        "sites": "data/mask/sites.json order",
        "z": "fold-averaged null z (ip), OOF z (cf)",
    }
    Path(out).write_text(json.dumps({"meta": meta, "layers": res}) + "\n")
    print(f"[probe] wrote {out}", flush=True)
