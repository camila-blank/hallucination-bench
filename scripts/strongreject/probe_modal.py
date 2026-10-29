"""Score every strongreject read site with the overrefusal eval's refusal probe (plan 0016 rev 4,
the plan-0015 probe protocol; Modal, CPU, one container per layer).

The probe artifact is plan 0015's ``refusal_probe_v1.npz`` (hb-overrefusal,
``data/overrefusal/probes/``; keys ``{variant}/L{layer}/{U,muU,sigma,null_mean,null_sd,tau,
c_mean,c_sd,c_tau,validity}``, arrays [fold, tail offset]). Scoring is that plan's rule, re-
implemented here (numpy only): tail sites (the 4 template-tail positions ``<|im_end|> \\n
<|im_start|> assistant``, offsets 3..0) ``s = (h.U[f] - muU[f,o]) / sigma[f,o]``, ``z = (s -
null_mean) / null_sd``, present iff ``s > tau``, averaged over the 4 folds (present iff mean
presence >= 0.5); every other read site uses the content rule ``p = h.U[f]`` vs
``c_mean / c_sd / c_tau``. Our acts (``wsbench-acts:/hb-strongreject-acts``) hold the whole
template render, think block included, so the tail is located from ``items.json``
(``n_positions``), not from the end of the acts.

    MODAL_PROFILE=safety-sahan-2 ~/.local/bin/modal run scripts/strongreject/probe_modal.py \\
        --npz /workspace/hb-overrefusal/data/overrefusal/probes/refusal_probe_v1.npz
        [--calib-npz .../refusal_probe_v1_calib.npz]
With a v2 artifact (``tau_v2`` keys: plan 0015's on-distribution answered-prompt threshold)
``present`` / ``z`` use v2 and ``present_alt`` is the v1 Alpaca threshold (sensitivity); with v1
(+ ``--calib-npz``) ``present`` is Alpaca and ``present_alt`` the exploratory calib threshold.
Writes outputs/strongreject/probe_sites.jsonl ({variant, layer, id, pos, cls, z, present,
present_alt}).
"""

import io
import json
from pathlib import Path, PurePosixPath

import modal

ACTS = Path("/acts")
READ_ACTS = "hb-strongreject-acts"
LAYERS = (28, 36, 44, 52)
N_TAIL = 4
_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2] if len(_HERE.parents) > 2 else Path("/root")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch>=2.6", index_url="https://download.pytorch.org/whl/cpu")
    .uv_pip_install("numpy>=2.0", "safetensors>=0.5")
)
app = modal.App("hb-strongreject-probe")
acts_vol = modal.Volume.from_name("wsbench-acts")
volumes: dict[str | PurePosixPath, modal.Volume] = {PurePosixPath(ACTS): acts_vol.read_only()}


@app.function(image=image, volumes=volumes, cpu=8, memory=32768, timeout=3600)
def score_layer(layer: int, npz: bytes, items: list[dict], acts: str = READ_ACTS) -> list[list]:
    import numpy as np
    from safetensors import safe_open

    arr = dict(np.load(io.BytesIO(npz)))
    variants = sorted({k.split("/")[0] for k in arr if k.split("/")[1] == f"L{layer}"})
    man = json.loads((ACTS / acts / "manifest.json").read_text())
    li = man["layers"].index(layer)
    files = {e["label"]: e for e in man["prompts"]}
    out = []
    for it in items:
        e = files[it["id"]]
        with safe_open(str(ACTS / acts / e["file"]), framework="pt", device="cpu") as fh:
            H = fh.get_slice("acts")[li].float().numpy().astype(np.float64)
        n = it["n_positions"]
        toks = [t.replace("Ċ", "\n") for t in e["tokens"][n - N_TAIL : n]]
        assert toks == ["<|im_end|>", "\n", "<|im_start|>", "assistant"], (it["id"], toks)
        for v in variants:
            g = {
                k.split("/")[2]: arr[k].astype(np.float64)
                for k in arr
                if k.startswith(f"{v}/L{layer}/")
            }
            F = g["U"].shape[0]
            for p in it["sites"]:
                h = H[p]
                if p >= n - N_TAIL:
                    o = n - 1 - p
                    s = (g["U"] @ h - g["muU"][:, o]) / g["sigma"][:, o]
                    if "tau_v2" in g:  # plan 0015's v2 threshold; v1 (Alpaca) kept as alt
                        z = (s - g["mu_v2"][:, o]) / g["sd_v2"][:, o]
                        pr, prc = s > g["tau_v2"][:, o], s > g["tau"][:, o]
                    else:
                        z = (s - g["null_mean"][:, o]) / g["null_sd"][:, o]
                        pr = s > g["tau"][:, o]
                        prc = s > g["tau_calib"][:, o] if "tau_calib" in g else None
                    cls = f"tail{o}"
                else:
                    pp = g["U"] @ h
                    if "c_tau_v2" in g:
                        z = (pp - g["c_mu_v2"]) / g["c_sd_v2"]
                        pr, prc = pp > g["c_tau_v2"], pp > g["c_tau"]
                    else:
                        z = (pp - g["c_mean"]) / g["c_sd"]
                        pr = pp > g["c_tau"]
                        prc = pp > g["c_tau_calib"] if "c_tau_calib" in g else None
                    cls = "content"
                assert len(z) == F
                pc = None if prc is None else float(prc.mean())
                out.append([v, layer, it["id"], p, cls, float(z.mean()), float(pr.mean()), pc])
    return out


@app.local_entrypoint()
def main(
    npz: str,
    calib_npz: str = "",
    out: str = "outputs/strongreject/probe_sites.jsonl",
    layers: str = ",".join(map(str, LAYERS)),
    acts: str = READ_ACTS,
) -> None:
    """Plan 0019: ``--layers 20,24,32,40,48,56,60 --acts hb-strongreject-L11-acts --npz
    .../refusal_probe_v2_l11.npz --out outputs/strongreject/probe_sites_l11.jsonl``."""
    import numpy as np

    arrs = dict(np.load(npz))
    if calib_npz:  # plan 0015's exploratory answered-prompt thresholds (tau_calib, c_tau_calib)
        arrs.update(dict(np.load(calib_npz)))
    buf = io.BytesIO()
    np.savez(buf, **arrs)
    blob = buf.getvalue()
    items = [
        {"id": it["id"], "n_positions": it["n_positions"], "sites": [s["pos"] for s in it["sites"]]}
        for it in json.loads((_REPO / "data/strongreject/items.json").read_text())["items"]
    ]
    dst = _REPO / out
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with dst.open("w") as fh:
        lays = [int(x) for x in layers.split(",")]
        for rows in score_layer.starmap([(la, blob, items, acts) for la in lays]):
            for v, la, i, p, cls, z, pr, pc in rows:
                rec = {"variant": v, "layer": la, "id": i, "pos": p, "cls": cls, "z": z}
                rec |= {"present": pr, "present_alt": pc}
                fh.write(json.dumps(rec) + "\n")
                n += 1
    print(f"wrote {dst}: {n} site scores")
