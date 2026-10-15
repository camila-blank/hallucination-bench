"""Plan 0009 §3c-d, per-class core of the Modal CPU job (rev 8.2: stream per class).

``scripts/sibling_gate/probes_modal.py`` copies this file to ``/root/sibfit.py``. Everything
here is plain numpy and takes its inputs as arguments, so the per-class path is unit-testable
without Modal or safetensors (tests/test_sibling_gate_probe_fit.py).

Why per class: the uncapped / >=4-pair run has ~220 classes x ~800 texts x ~70 positions x 12
layers x 5,120 dims; one per-layer container holding every class's windows does not fit in
memory. A class's probes only need that class's texts, so a container reads exactly one class's
read-position windows (all layers at once, fp16) and fits its 12 layers in sequence.
"""

from __future__ import annotations

import io
import json
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

try:  # inside the Modal job the modules are copied as top-level files
    from hallucination_bench.evals.sibling_gate import probes as P
    from hallucination_bench.evals.sibling_gate import um_probe as Q
except ImportError:  # pragma: no cover
    import sibprobes as P  # type: ignore[no-redef]
    import um_probe as Q  # type: ignore[no-redef]

OTHER = 13  # texts.TOKEN_CLASSES.index("other")
N_CLASSES = 14  # len(texts.TOKEN_CLASSES)
MIN_NULL_Z = 400  # a key with fewer null-z values fails the null check
NULL_MEAN_MAX = 0.1
NULL_SD = (0.8, 1.25)
DISTRACTOR_LAYER = 44

# (path, lo, hi) -> [n_layers_in_file, hi - lo, d] float array: the window [lo, hi) of every
# stored layer of one acts file (``safetensors`` ``acts`` tensor, layer-major).
SliceReader = Callable[[str, int, int], np.ndarray]


def safetensors_reader(path: str, lo: int, hi: int) -> np.ndarray:  # pragma: no cover - modal
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as f:
        return f.get_slice("acts")[:, lo:hi].float().numpy()


def read_class_windows(
    acts_root: Path,
    sub: str,
    texts: Sequence[Mapping[str, Any]],
    file_layers: Sequence[int],
    layers: Sequence[int],
    reader: SliceReader = safetensors_reader,
    workers: int = 16,
) -> dict[str, np.ndarray]:
    """One class's texts: label -> fp16 ``[len(layers), n_pos, d]`` read at the manifest's
    ``eval_positions`` (every layer of the file read once; positions outside [min, max] never
    touched). ``file_layers`` = the manifest's layer list, ``layers`` those to keep."""
    # ``sub`` may be a comma list of capture shards (rev 8.2: the probe bank is captured in
    # parallel shards ``acts-sibgate-probe-sNN``); labels are unique across shards
    ent: dict[str, dict] = {}
    for sd in sub.split(","):
        man = json.loads((acts_root / sd / "manifest.json").read_text())
        for p in man["prompts"]:
            ent[p["label"]] = {**p, "file": f"{sd}/{p['file']}"}
    lis = [list(file_layers).index(L) for L in layers]

    def one(label: str):
        e = ent[label]
        pos = list(e["eval_positions"])
        if not pos:
            return label, np.zeros((len(lis), 0, 1), np.float16), []
        p0, p1 = min(pos), max(pos) + 1
        x = reader(str(acts_root / e["file"]), p0, p1)
        return label, x[lis][:, [p - p0 for p in pos]].astype(np.float16), pos

    labels = sorted({t["id"] for t in texts})
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, labels))
    return {lab: x for lab, x, pos in res if pos}


def layer_windows(win16: Mapping[str, np.ndarray], k: int) -> dict[str, np.ndarray]:
    """The k-th layer of every text as float32 (one layer at a time keeps memory bounded)."""
    return {lab: x[k].astype(np.float32) for lab, x in win16.items()}


def _merge(cls: Mapping[str, Any]) -> dict[int, int]:
    """The class's raw token class -> key map (``capture-manifest``, fixed across layers). The
    texts' ``key`` arrays are already in this key space; old inputs without ``merge``: identity."""
    m = cls.get("merge") or list(range(N_CLASSES))
    return {c: int(m[c]) for c in range(N_CLASSES)}


def textset(texts: Sequence[Mapping[str, Any]], win: Mapping[str, np.ndarray], merge):
    texts = [t for t in texts if t["id"] in win and len(win[t["id"]])]
    return list(texts), P.TextSet.build(
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


def fit_class_layer(
    slug: str, cls: Mapping[str, Any], win: Mapping[str, np.ndarray], layer: int
) -> tuple[dict, bytes | None]:
    """Fit one (class, layer): cross-fit, per-fold null stats, OOF implied validity per concept,
    the null check per key (rev 8: a failing key does not take down the class). Returns the
    result row (``slug``; at L44 + the grand-mean-centred implied ``centroids``) and the frozen
    fold models as ``np.savez`` bytes (None when a concept has no training texts)."""
    t0 = time.time()
    C = len(cls["concepts"])
    merge = _merge(cls)
    texts, ts = textset(cls["texts"], win, merge)
    res: dict = {"layer": layer, "class": cls["class"], "n_texts": len(texts)}
    try:
        cfa, lams, _tab = P.cross_fit_whitened(ts, C)
    except ValueError as e:  # a concept without training texts
        res |= {
            "error": str(e),
            "validity": {"a": [None] * C},
            "null_check": {"other_ok": False},
        }
        return {**res, "slug": slug}, None
    nsa = Q.null_stats(ts, cfa.oof, C)
    za = Q.oof_z(ts, cfa.oof, nsa)
    va = Q.validity_implied(ts, za, C)
    nc = null_check(ts, cfa.oof, C)
    arrs: dict[str, np.ndarray] = {}
    for k, m in enumerate(cfa.models):
        keys = np.array(sorted(m.keys, key=lambda kk: m.keys[kk]), dtype=np.int64)
        arrs |= {f"a{k}_U": m.U, f"a{k}_muU": m.muU, f"a{k}_sigma": m.sigma, f"a{k}_keys": keys}
    arrs["na_offsets"] = nsa.offsets
    for k in range(len(nsa.m)):
        arrs[f"na{k}_m"], arrs[f"na{k}_sd"] = nsa.m[k], nsa.sd[k]
    arrs["merge"] = np.array([merge[c] for c in range(N_CLASSES)], dtype=np.int64)
    buf = io.BytesIO()
    np.savez(buf, **arrs)
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
        res["centroids"] = {c: [float(x) for x in cent[i]] for i, c in enumerate(cls["concepts"])}
    return {**res, "slug": slug}, buf.getvalue()


def load_models(blob: bytes):
    """The frozen fold models + null stats + key merge of one (class, layer) ``npz``."""
    z = np.load(io.BytesIO(blob))
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


def score_class_layer(
    blob: bytes,
    eval_win: Mapping[str, tuple[np.ndarray, list[int]]],
    site_keys: Mapping[str, Mapping[str, int]],
) -> bytes:
    """Score every eval site of every item with the frozen fold models: ``item``, ``pos``,
    ``za`` [P, C], ``key`` [P] (the merged site key) as ``np.savez`` bytes."""
    ma, ns, merge = load_models(blob)
    rows_item, rows_pos, rows_key, Z = [], [], [], []
    for it, (x, pos) in sorted(eval_win.items()):
        keys = np.array(
            [merge.get(int(site_keys[it].get(str(p), OTHER)), OTHER) for p in pos], dtype=np.int64
        )
        ending = np.zeros(len(pos), np.int64)
        za = np.nanmean([ns.z(k, m.score(x, ending, keys), keys) for k, m in enumerate(ma)], axis=0)
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
    return buf.getvalue()
