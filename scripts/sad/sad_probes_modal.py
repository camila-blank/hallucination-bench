# ruff: noqa: E501
"""Plan 0018 rev 6: eval-awareness probes on Modal (one CPU container per layer, fanned out).

Inputs ``outputs/sad/probe_inputs.json`` (``hbench sad probe-inputs``: positions, labels, folds,
centring keys; no text). Reads the residuals at the listed sites from ``wsbench-acts``
(``hb-sad-probe-train`` / ``hb-sad-probe-contrast`` / the pilot's ``hb-sad-acts{,2}``), fits the
shrinkage-whitened DiffMean (``evals/sibling_probe/probes.py`` recipe V2, shipped into the
container as a file and imported -- the same module the UM ladder uses) and writes per-layer JSON
to ``--out``: OOF / transfer AUROCs, the chosen lambdas, the opposite-class q95 thresholds and every
pilot site's scores + present flags.

    MODAL_PROFILE=safety-sahan-2 uvx --python 3.12 modal@latest run --env camila-blank \\
        scripts/sad/sad_probes_modal.py --inputs outputs/sad/probe_inputs.json --out outputs/sad/probes
"""

import json
import time
from pathlib import Path, PurePosixPath

import modal

ACTS_MOUNT = Path("/acts")
_HERE = Path(__file__).resolve()
_SRC = (
    _HERE.parents[2] / "src" / "hallucination_bench" / "evals" / "sibling_probe"
    if len(_HERE.parents) > 2
    else _HERE
)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch>=2.6", index_url="https://download.pytorch.org/whl/cpu")
    .uv_pip_install("numpy>=2.0", "safetensors>=0.5")
    .env({"PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "16", "OPENBLAS_NUM_THREADS": "16"})
)
if (_SRC / "probes.py").is_file():
    image = image.add_local_file(str(_SRC / "probes.py"), "/root/sibprobes.py")

app = modal.App("hb-sad-probes")
acts_vol = modal.Volume.from_name("wsbench-acts", environment_name="camila-blank")
volumes: dict[str | PurePosixPath, modal.Volume] = {PurePosixPath(ACTS_MOUNT): acts_vol.read_only()}


def _load(sub: str, rows: list[dict], layer: int):
    """[sum sites, d] float32 for ``rows`` (each ``{id, pos}``), in order."""
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    from safetensors import safe_open

    man = json.loads((ACTS_MOUNT / sub / "manifest.json").read_text())
    li = man["layers"].index(layer)
    ent = {p["label"]: p for p in man["prompts"]}

    def one(r: dict):
        e = ent[r["id"]]
        pos = r["pos"]
        p0, p1 = min(pos), max(pos) + 1
        with safe_open(str(ACTS_MOUNT / sub / e["file"]), framework="pt", device="cpu") as f:
            x = f.get_slice("acts")[li : li + 1, p0:p1].float().numpy()[0]
        return x[[p - p0 for p in pos]].astype(np.float32)

    t = time.time()
    with ThreadPoolExecutor(32) as ex:
        out = list(ex.map(one, rows))
    print(f"[L{layer}] {sub}: {len(rows)} files in {time.time() - t:.0f}s", flush=True)
    return out


def _textset(P, X: list, items: list[dict], fold_key: str | None):
    import numpy as np

    text = np.concatenate([np.full(len(x), i) for i, x in enumerate(X)]).astype(np.int64)
    keys = [k for it in items for k in it["keys"]]
    return P.TextSet(
        H=np.concatenate(X).astype(np.float32),
        text=text,
        ending=np.array([k[0] for k in keys], dtype=np.int64),
        offset=np.array([k[1] for k in keys], dtype=np.int64),
        ids=[it["id"] for it in items],
        concept=np.array([it["label"] for it in items], dtype=np.int64),
        kind=["implied"] * len(items),
        t_ending=np.zeros(len(items), dtype=np.int64),
        fold=np.array([it[fold_key] if fold_key else -1 for it in items], dtype=np.int64),
    )


def _item_auroc(P, ts, S, rows=None):
    """AUROC of the per-item mean eval score (column 0), eval (label 0) vs deploy (label 1)."""
    import numpy as np

    sel = np.ones(len(ts.text), bool) if rows is None else rows
    T = len(ts.ids)
    s = np.full(T, np.nan)
    for i in range(T):
        m = sel & (ts.text == i)
        if m.any():
            s[i] = np.nanmean(S[m, 0])
    lab = ts.concept
    return float(P.auroc(s[lab == 0], s[lab == 1]))


def _fit(P, ts, tag: str, layer: int):
    """Cross-fit V2; OOF item AUROCs; per-key opposite-class q95 thresholds."""

    t = time.time()
    cf, lams, _ = P.cross_fit_whitened(ts, 2, P.N_FOLDS)
    oof = cf.oof
    rc = ts.row_concept
    res = {
        "lams": lams,
        "auroc_item": _item_auroc(P, ts, oof),
        "auroc_assistant": _item_auroc(P, ts, oof, (ts.ending == 1) & (ts.offset == 0))
        if (ts.ending == 1).any()
        else None,
        "auroc_body": _item_auroc(P, ts, oof, ts.ending == 0) if (ts.ending == 0).any() else None,
        "n_items": len(ts.ids),
        "n_by_label": [int((ts.concept == 0).sum()), int((ts.concept == 1).sum())],
    }
    tau = {}
    for e, o in sorted(set(zip(ts.ending.tolist(), ts.offset.tolist(), strict=True))):
        at = (ts.ending == e) & (ts.offset == o)
        tau[f"{e},{o}"] = {
            "eval": P.q95(oof[at & (rc == 1), 0]),  # eval probe vs deploy-class sites
            "deploy": P.q95(oof[at & (rc == 0), 1]),  # deploy probe vs eval-class sites
        }
    res["tau"] = tau
    print(
        f"[L{layer}] {tag}: item AUROC {res['auroc_item']:.3f} lams {lams} ({time.time() - t:.0f}s)",
        flush=True,
    )
    return cf, res


def _score_pilot(P, cf, tau, pts, single_key: bool):
    import numpy as np

    S = np.nanmean(np.stack([m.score(pts.H, pts.ending, pts.offset) for m in cf.models]), axis=0)
    out = []
    i = 0
    for t, _iid in enumerate(pts.ids):
        n = int((pts.text == t).sum())
        for j in range(n):
            e, o = int(pts.ending[i]), int(pts.offset[i])
            k = "0,0" if single_key else f"{e},{o}"
            out.append(
                {
                    "i": t,
                    "j": j,
                    "s_eval": float(S[i, 0]),
                    "s_deploy": float(S[i, 1]),
                    "eval": bool(S[i, 0] > tau[k]["eval"]),
                    "deploy": bool(S[i, 1] > tau[k]["deploy"]),
                }
            )
            i += 1
    return S, out


@app.function(image=image, volumes=volumes, cpu=16, memory=65536, timeout=3 * 3600)
def fit_layer(layer: int, inputs: dict) -> dict:
    import sys

    sys.path.insert(0, "/root")
    import sibprobes as P

    train, ctr, pil = inputs["train"], inputs["contrast"], inputs["pilot"]
    Xtr = _load("hb-sad-probe-train", train, layer)
    Xc = _load("hb-sad-probe-contrast", ctr, layer)
    Xp = [None] * len(pil)
    for sub in sorted({p["acts"] for p in pil}):
        idx = [i for i, p in enumerate(pil) if p["acts"] == sub]
        for i, x in zip(idx, _load(sub, [pil[i] for i in idx], layer), strict=True):
            Xp[i] = x
    pts = _textset(P, Xp, pil, None)
    out: dict = {"layer": layer, "pilot_ids": [p["id"] for p in pil]}
    for variant in ("wildchat", "all"):
        items = [t for t in train if variant == "all" or t["wildchat"]]
        X = [x for x, t in zip(Xtr, train, strict=True) if variant == "all" or t["wildchat"]]
        ts = _textset(P, X, items, f"fold_{variant}")
        cf, res = _fit(P, ts, f"primary/{variant}", layer)
        S, sites = _score_pilot(P, cf, res["tau"], pts, False)
        res["transfer_auroc_item"] = _item_auroc(P, pts, S)
        res["transfer_auroc_assistant"] = _item_auroc(
            P, pts, S, (pts.ending == 1) & (pts.offset == 0)
        )
        res["pilot_sites"] = sites
        out[f"primary_{variant}"] = res
    tsc = _textset(P, Xc, ctr, "fold")
    cf, res = _fit(P, tsc, "secondary/contrast", layer)
    tau1 = {"0,0": res["tau"]["0,0"]}
    pts1 = _textset(P, Xp, [{**p, "keys": [(0, 0)] * len(p["keys"])} for p in pil], None)
    S, sites = _score_pilot(P, cf, tau1, pts1, True)
    res["transfer_auroc_item"] = _item_auroc(P, pts1, S)
    res["transfer_auroc_assistant"] = _item_auroc(P, pts1, S, (pts.ending == 1) & (pts.offset == 0))
    res["pilot_sites"] = sites
    out["secondary"] = res
    return out


@app.local_entrypoint()
def main(inputs: str, out: str, layers: str = "28,36,44,52") -> None:
    inp = json.loads(Path(inputs).read_text())
    Ls = [int(x) for x in layers.split(",")]
    d = Path(out)
    d.mkdir(parents=True, exist_ok=True)
    for res in fit_layer.starmap([(L, inp) for L in Ls]):
        (d / f"L{res['layer']:02d}.json").write_text(json.dumps(res))
        print(f"[probes] wrote L{res['layer']}", flush=True)
