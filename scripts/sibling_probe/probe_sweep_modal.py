"""Diagnostic sweep of concept-probe recipes (plan 0003 §Phase E follow-up; diagnostics only, the
probe pipeline in ``probes_modal.py`` / ``probes.py`` is untouched).

One CPU container per category (``sweep_category.map``) reads that category's probe texts and
eval items from ``wsbench-acts`` (same capture and reading code as ``probes_modal.py``) and
evaluates, at every layer, with the SAME category-level folds (``probes.assign_folds``,
salt = category):

  V0  the current recipe (``probes.cross_fit``: text-mean DiffMean, raw, one-vs-rest).
  V1  per-position DiffMean: one direction per (concept, position group) from the training rows
      at that position, centred by the per-(ending, offset) equal-weighted concept mean.
      Position group = (ending, offset) for request-sentence offsets (offset >= 9) and
      offset alone for the 9 template-tail offsets 0..8 (token-identical across endings, so
      endings are pooled there).
  V2  shrinkage-whitened text-mean DiffMean: d = (S_w + a I)^-1 d_V0, a = lam * tr(S_w) / D,
      S_w = pooled within-(concept, ending, offset) covariance of the training POSITIONS
      (rows minus their (concept, ending, offset) mean; ~11k rows x 5120, exact eigh).
  V3  V1 directions whitened with the same S_w / lam.
  V4  L2 logistic regression one-vs-rest (class-balanced) on per-position activations,
      centred per (ending, offset) and standardised per dim; ceiling reference only.

lam (V2/V3) and the L2 strength (V4) are chosen per outer fold by inner CV on the outer
training folds only (leave-one-training-fold-out, criterion = mean over concepts of the
position-level AUROC on the inner held-out fold). Every variant then uses the identical
per-(ending, offset) centring and scaling of ``probes.fit_fold`` (mu = equal-weighted concept
mean, sigma = sd of the projected training rows at that position).

Metrics per (variant, category, layer, concept): position-level OOF AUROC (all, request
offsets, tail offsets, per offset); eval-site transfer AUROC (fold-averaged score, c-gold items'
cells vs other-gold items' cells; items whose prompt names the gold are not positives), pooled
over cells and averaged over offsets; calibration at the fold's tau_cat (95th pct of s_c over
fold k's other-concept OOF positions): FPR on held-out other-concept positions (text split-half
inside fold k), on fold k's neutral texts, and at eval cells of other-gold items; gold
sensitivity = detection rate of the gold probe at its own items' cells (fold-averaged).

    MODAL_PROFILE=safety-sahan-2 uvx --python 3.12 modal@latest run \\
        scripts/sibling_probe/probe_sweep_modal.py
    uv run python scripts/sibling_probe/probe_sweep_summary.py   # -> summary.md
"""

import json
import time
from pathlib import Path, PurePosixPath

import modal

ACTS_MOUNT = Path("/acts")
OUT_MOUNT = Path("/out")
_HERE = Path(__file__).resolve()
_PROBES_PY = (
    _HERE.parents[2] / "src" / "hallucination_bench" / "evals" / "sibling_probe" / "probes.py"
    if len(_HERE.parents) > 2
    else _HERE
)
TAIL = 9  # template-tail tokens: <|im_end|> \n <|im_start|> assistant \n <think> \n\n </think> \n\n
LAMS_W = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
LAMS_LR = (3e-1, 3e-2, 3e-3, 3e-4)  # descending: LBFGS warm starts

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch>=2.6", index_url="https://download.pytorch.org/whl/cpu")
    .uv_pip_install("numpy>=2.0", "scipy>=1.13", "safetensors>=0.5")
    .env({"PYTHONUNBUFFERED": "1"})
)
if _PROBES_PY.name == "probes.py" and _PROBES_PY.is_file():
    image = image.add_local_file(str(_PROBES_PY), "/root/sibprobes.py")

app = modal.App("sibling-probe-sweep")
acts_vol = modal.Volume.from_name("wsbench-acts", create_if_missing=True)
out_vol = modal.Volume.from_name("sibling-probes", create_if_missing=True)
volumes: dict[str | PurePosixPath, modal.Volume] = {
    PurePosixPath(ACTS_MOUNT): acts_vol.read_only(),
    PurePosixPath(OUT_MOUNT): out_vol,
}


@app.function(image=image, volumes=volumes, cpu=16, memory=65536, timeout=4 * 3600)
def sweep_category(cat: str, inputs: dict, cfg: dict) -> dict:
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import sibprobes as P  # type: ignore
    import torch
    from safetensors import safe_open
    from scipy.stats import rankdata

    torch.set_num_threads(16)
    t0 = time.time()
    layers = [int(x) for x in cfg["layers"]]
    order = inputs["concepts"][cat]
    C = len(order)
    cidx = {c: i for i, c in enumerate(order)}

    def manifest(sub):
        return json.loads((ACTS_MOUNT / sub / "manifest.json").read_text())

    ev_man, pr_man = manifest(cfg["eval_acts"]), manifest(cfg["probe_acts"])
    ev_entries = {p["label"]: p for p in ev_man["prompts"]}
    pr_entries = {p["label"]: p for p in pr_man["prompts"]}
    items = [
        it
        for it in inputs["items"]
        if it["category"] == cat and it["id"] in ev_entries and it["gold"] in cidx
    ]
    texts = [t for t in inputs["probe_texts"] if t["category"] == cat and t["id"] in pr_entries]
    folds = P.assign_folds(texts, salt=cat)

    cache = {}

    def fetch(sub, man, entry):
        lis = [man["layers"].index(L) for L in layers]
        pos = list(entry.get("eval_positions") or range(entry["n_pos"]))
        p0, p1 = min(pos), max(pos) + 1
        a, b = min(lis), max(lis) + 1
        with safe_open(str(ACTS_MOUNT / sub / entry["file"]), framework="pt", device="cpu") as f:
            x = f.get_slice("acts")[a:b, p0:p1].float().numpy()
        x = x[[li - a for li in lis]][:, [p - p0 for p in pos]].astype(np.float16)
        cache[(sub, entry["label"])] = (x, [entry["n_pos"] - 1 - p for p in pos])

    tr = time.time()
    with ThreadPoolExecutor(64) as ex:
        list(
            ex.map(
                lambda e: fetch(cfg["eval_acts"], ev_man, e), [ev_entries[i["id"]] for i in items]
            )
        )
        list(
            ex.map(
                lambda e: fetch(cfg["probe_acts"], pr_man, e), [pr_entries[t["id"]] for t in texts]
            )
        )
    print(f"[sweep {cat}] read {len(texts)} texts + {len(items)} items in {time.time() - tr:.0f}s")

    # ------------------------------------------------------------------ helpers
    def auc(pos, neg):
        pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
        if not pos.size or not neg.size:
            return float("nan")
        r = rankdata(np.concatenate([pos, neg]))
        return float((r[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))

    def q95(x):
        x = x[np.isfinite(x)]
        return float(np.quantile(x, 0.95)) if x.size else float("nan")

    class Model:
        """mu [K, D] per-key centring, U [G, C, D] unit directions, gidx [K] key -> group,
        sigma [K, C]."""

        def __init__(self, mu, U, gidx, sigma):
            self.mu, self.U, self.gidx, self.sigma = mu, U, gidx, sigma

        def score(self, H, rkey):
            out = np.full((H.shape[0], C), np.nan)
            for k in np.unique(rkey):
                if k < 0 or not np.isfinite(self.sigma[k]).any():
                    continue
                r = rkey == k
                out[r] = ((H[r] - self.mu[k]) @ self.U[self.gidx[k]].T) / self.sigma[k]
            return out

    def finish(st, Dirs, gidx):
        """Unit directions + the probes.fit_fold scaling on the training rows of ``st``."""
        U = Dirs / np.maximum(np.linalg.norm(Dirs, axis=-1, keepdims=True), 1e-12)
        sig = np.full((K, C), np.nan)
        for k in range(K):
            r = st["rkey"] == k
            if r.sum() > 1:
                p = st["Xc"][r] @ U[gidx[k]].T
                sd = p.std(axis=0, ddof=1)
                sig[k] = np.where(sd > 0, sd, np.nan)
        return Model(st["mu"], U.astype(np.float32), gidx, sig)

    # ------------------------------------------------------------------ per layer
    results = {}
    for li, L in enumerate(layers):
        tl = time.time()
        ts = P.TextSet.build(
            [cache[(cfg["probe_acts"], t["id"])][0][li].astype(np.float32) for t in texts],
            [t["id"] for t in texts],
            [cidx[t["concept"]] if t["concept"] is not None else P.NEUTRAL for t in texts],
            [t["kind"] for t in texts],
            [t["request_idx"] for t in texts],
            [folds[t["id"]] for t in texts],
            offsets=[cache[(cfg["probe_acts"], t["id"])][1] for t in texts],
        )
        keys = sorted(set(zip(ts.ending.tolist(), ts.offset.tolist(), strict=True)))
        K = len(keys)
        kix = {k: i for i, k in enumerate(keys)}
        rkey = np.array([kix[(e, o)] for e, o in zip(ts.ending, ts.offset, strict=True)])
        groups = sorted({(e if o >= TAIL else -1, o) for e, o in keys})
        gix = {g: i for i, g in enumerate(groups)}
        gidx1 = np.array([gix[(e if o >= TAIL else -1, o)] for e, o in keys])
        gidx0 = np.zeros(K, np.int64)
        rc, rf = ts.row_concept, ts.row_fold
        Vmeans = ts.text_means()
        # eval rows
        ev_H = np.concatenate(
            [cache[(cfg["eval_acts"], it["id"])][0][li].astype(np.float32) for it in items]
        )
        ev_off = np.concatenate([cache[(cfg["eval_acts"], it["id"])][1] for it in items])
        ev_end = np.concatenate(
            [[it["request_idx"]] * len(cache[(cfg["eval_acts"], it["id"])][1]) for it in items]
        )
        ev_item = np.concatenate(
            [[i] * len(cache[(cfg["eval_acts"], it["id"])][1]) for i, it in enumerate(items)]
        )
        ev_key = np.array(
            [kix.get((int(e), int(o)), -1) for e, o in zip(ev_end, ev_off, strict=True)]
        )
        ev_gold = np.array([cidx[items[i]["gold"]] for i in ev_item])
        ev_named = [set(items[i].get("named") or ()) for i in ev_item]

        stats = {}

        def get_stats(fs):
            fs = frozenset(fs)
            if fs in stats:
                return stats[fs]
            T = np.flatnonzero(np.isin(ts.fold, list(fs)) & (ts.concept >= 0))
            intr = np.zeros(len(ts.ids), bool)
            intr[T] = True
            rows = np.flatnonzero(intr[ts.text])
            Hr, kr, cr = ts.H[rows].astype(np.float64), rkey[rows], rc[rows]
            gid = kr * C + cr
            m = np.zeros((K * C, Hr.shape[1]))
            cnt = np.bincount(gid, minlength=K * C).astype(np.float64)
            np.add.at(m, gid, Hr)
            m = (m / np.maximum(cnt, 1)[:, None]).reshape(K, C, -1)
            cnt = cnt.reshape(K, C)
            mu = np.stack([m[k][cnt[k] > 0].mean(axis=0) for k in range(K)])
            Xc = (Hr - mu[kr]).astype(np.float32)
            # V0 text-mean DiffMean
            mt = np.stack([Vmeans[T[ts.concept[T] == c]].mean(axis=0) for c in range(C)])
            D0 = mt - (mt.sum(0, keepdims=True) - mt) / (C - 1)
            # V1 per-group DiffMean of centred position means
            G = len(groups)
            mg = np.zeros((G, C, Hr.shape[1]))
            wg = np.zeros((G, C))
            for k in range(K):
                mg[gidx1[k]] += cnt[k][:, None] * (m[k] - mu[k])
                wg[gidx1[k]] += cnt[k]
            mg /= np.maximum(wg, 1)[..., None]
            D1 = mg - (mg.sum(1, keepdims=True) - mg) / (C - 1)
            st = {
                "rows": rows,
                "rkey": kr,
                "rc": cr,
                "mu": mu,
                "Xc": Xc,
                "D0": D0,
                "D1": D1,
                "m": m,
                "n": len(rows),
            }
            stats[fs] = st
            return st

        def eig(st):
            if "E" not in st:
                R = (
                    st["Xc"].astype(np.float64)
                    - (st["m"] - st["mu"][:, None])[st["rkey"], st["rc"]]
                )
                S = (R.T @ R) / max(st["n"] - K * C, 1)
                E, V = np.linalg.eigh(S)
                st["E"], st["V"] = np.maximum(E, 0), V
            return st["E"], st["V"]

        def whiten(st, Dm, lam):
            E, V = eig(st)
            a = lam * E.mean()
            sh = Dm.shape
            X = Dm.reshape(-1, sh[-1]).T
            return (V @ ((V.T @ X) / (E + a)[:, None])).T.reshape(sh)

        lr_cache = {}

        def fit_lr(fs, lam):
            fs = frozenset(fs)
            if (fs, lam) in lr_cache:
                return lr_cache[(fs, lam)]
            st = get_stats(fs)
            if "Xs" not in st:
                sd = st["Xc"].std(axis=0) + 1e-6
                st["sd"] = sd
                st["Xs"] = torch.from_numpy(st["Xc"] / sd)
            X = st["Xs"]
            Y = torch.from_numpy((st["rc"][:, None] == np.arange(C)[None]).astype(np.float32))
            npos = Y.sum(0)
            wpos = X.shape[0] / (2 * npos)
            wneg = X.shape[0] / (2 * (X.shape[0] - npos))
            Wt = Y * wpos + (1 - Y) * wneg
            prev = [v for (f2, l2), v in lr_cache.items() if f2 == fs]
            W0 = torch.from_numpy(prev[-1][0]) if prev else torch.zeros(X.shape[1], C)
            W = W0.clone().requires_grad_(True)
            b = torch.zeros(C, requires_grad=True)
            opt = torch.optim.LBFGS(
                [W, b],
                lr=1,
                max_iter=300,
                history_size=20,
                tolerance_grad=1e-6,
                tolerance_change=1e-9,
                line_search_fn="strong_wolfe",
            )

            def closure():
                opt.zero_grad()
                z = X @ W + b
                loss = (
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        z, Y, weight=Wt, reduction="sum"
                    )
                    / X.shape[0]
                ) + 0.5 * lam * (W * W).sum()
                loss.backward()
                return loss

            opt.step(closure)
            w = W.detach().numpy()
            lr_cache[(fs, lam)] = (w, st["sd"])
            return lr_cache[(fs, lam)]

        def build(variant, fs, h):
            st = get_stats(fs)
            if variant == "V1":
                return finish(st, st["D1"], gidx1)
            if variant == "V2":
                return finish(st, whiten(st, st["D0"], h)[None], gidx0)
            if variant == "V3":
                return finish(st, whiten(st, st["D1"], h), gidx1)
            if variant == "V4":
                w, sd = fit_lr(fs, h)
                return finish(st, (w / sd[:, None]).T[None].astype(np.float64), gidx0)
            raise ValueError(variant)

        def fold_rows(j, concept_only=True):
            r = rf == j
            return r & (rc >= 0) if concept_only else r

        def mean_valid(S, rows):
            return float(
                np.nanmean(
                    [
                        auc(S[rows & (rc == c), c], S[rows & (rc != c) & (rc >= 0), c])
                        for c in range(C)
                    ]
                )
            )

        inner_memo = {}

        def inner_score(variant, fs, j, h):
            key = (variant, frozenset(fs), j, h)
            if key not in inner_memo:
                mdl = build(variant, fs, h)
                r = fold_rows(j)
                S = np.full((len(rc), C), np.nan)
                S[r] = mdl.score(ts.H[r], rkey[r])
                inner_memo[key] = mean_valid(S, r)
            return inner_memo[key]

        grids = {"V1": (None,), "V2": LAMS_W, "V3": LAMS_W, "V4": LAMS_LR}
        variants = {}
        # V0: the pipeline's own cross-fit
        cf = P.cross_fit(ts, C)
        variants["V0"] = {
            "models": [
                lambda H, rk, m=m_: m.score(
                    H,
                    np.array([keys[k][0] if k >= 0 else -9 for k in rk]),
                    np.array([keys[k][1] if k >= 0 else -9 for k in rk]),
                )
                for m_ in cf.models
            ],
            "oof": cf.oof,
            "hyper": [None] * 4,
        }
        for v in cfg["variants"]:
            if v == "V0":
                continue
            tv = time.time()
            oof = np.full((len(rc), C), np.nan)
            models, hyper, inner_tab = [], [], []
            for k in range(4):
                F = [f for f in range(4) if f != k]
                grid = grids[v]
                if len(grid) > 1:
                    sc = {
                        h: float(
                            np.mean([inner_score(v, [f for f in F if f != j], j, h) for j in F])
                        )
                        for h in grid
                    }
                    hbest = max(sc, key=sc.get)
                    inner_tab.append(sc)
                else:
                    hbest = grid[0]
                mdl = build(v, F, hbest)
                r = rf == k
                oof[r] = mdl.score(ts.H[r], rkey[r])
                models.append(lambda H, rk, m=mdl: m.score(H, rk))
                hyper.append(hbest)
            variants[v] = {"models": models, "oof": oof, "hyper": hyper, "inner": inner_tab}
            print(
                f"[sweep {cat}] L{L} {v} done in {time.time() - tv:.0f}s hyper={hyper}", flush=True
            )

        # ------------------------------------------------------------------ metrics
        rng = np.random.default_rng([L, 11])
        res_L = {}
        for v, d in variants.items():
            oof = d["oof"]
            tau = np.full((4, C), np.nan)
            fpr_ho, fpr_neu = [], []
            for k in range(4):
                rk = rf == k
                for c in range(C):
                    tau[k, c] = q95(oof[rk & (rc != c) & (rc >= 0), c])
                Tk = np.flatnonzero((ts.fold == k) & (ts.concept >= 0))
                half = np.zeros(len(ts.ids), bool)
                half[rng.permutation(Tk)[: len(Tk) // 2]] = True
                hA = half[ts.text] & rk
                hB = (~half[ts.text]) & rk & (rc >= 0)
                row_ho, row_neu = [], []
                for c in range(C):
                    o = (rc != c) & (rc >= 0)
                    ta, tb = q95(oof[hA & o, c]), q95(oof[hB & o, c])
                    x1 = oof[hB & o, c]
                    x2 = oof[hA & o, c]
                    row_ho.append(
                        0.5
                        * (np.mean(x1[np.isfinite(x1)] > ta) + np.mean(x2[np.isfinite(x2)] > tb))
                    )
                    xn = oof[rk & (rc == P.NEUTRAL), c]
                    row_neu.append(
                        float(np.mean(xn[np.isfinite(xn)] > tau[k, c])) if xn.size else float("nan")
                    )
                fpr_ho.append(row_ho)
                fpr_neu.append(row_neu)
            fpr_ho, fpr_neu = np.array(fpr_ho), np.array(fpr_neu)
            # eval
            Sk = np.stack([f(ev_H, ev_key) for f in d["models"]])  # [4, P, C]
            with np.errstate(invalid="ignore"):
                s_ev = np.nanmean(Sk, axis=0)
                det = np.where(np.isfinite(Sk), (Sk > tau[:, None, :]).astype(float), np.nan)
                det = np.nanmean(det, axis=0)
            per_c = {}
            offs_all = sorted(set(ts.offset.tolist()))
            for c, cid in enumerate(order):
                pos_r = rc == c
                neg_r = (rc != c) & (rc >= 0)
                a_all = auc(oof[pos_r, c], oof[neg_r, c])
                a_req = auc(
                    oof[pos_r & (ts.offset >= TAIL), c], oof[neg_r & (ts.offset >= TAIL), c]
                )
                a_tail = auc(oof[pos_r & (ts.offset < TAIL), c], oof[neg_r & (ts.offset < TAIL), c])
                by_off = {
                    int(o): auc(oof[pos_r & (ts.offset == o), c], oof[neg_r & (ts.offset == o), c])
                    for o in offs_all
                }
                gpos = (ev_gold == c) & np.array([cid not in n for n in ev_named])
                gneg = ev_gold != c
                fneg = gneg & np.array([cid not in n for n in ev_named])
                t_pool = auc(s_ev[gpos, c], s_ev[gneg, c])
                t_off = [
                    auc(s_ev[gpos & (ev_off == o), c], s_ev[gneg & (ev_off == o), c])
                    for o in sorted(set(ev_off.tolist()))
                ]
                t_off = [x for x in t_off if np.isfinite(x)]
                t_req = auc(s_ev[gpos & (ev_off >= TAIL), c], s_ev[gneg & (ev_off >= TAIL), c])
                t_tail = auc(s_ev[gpos & (ev_off < TAIL), c], s_ev[gneg & (ev_off < TAIL), c])
                sens = float(np.nanmean(det[gpos, c])) if gpos.any() else float("nan")
                efpr = float(np.nanmean(det[fneg, c])) if fneg.any() else float("nan")
                per_c[cid] = {
                    "auroc": a_all,
                    "auroc_req": a_req,
                    "auroc_tail": a_tail,
                    "auroc_by_offset": by_off,
                    "transfer": t_pool,
                    "transfer_offmean": float(np.mean(t_off)) if t_off else float("nan"),
                    "transfer_req": t_req,
                    "transfer_tail": t_tail,
                    "n_gold_items": len({int(i) for i in ev_item[gpos]}),
                    "sens": sens,
                    "eval_fpr": efpr,
                    "fpr_heldout": float(np.nanmean(fpr_ho[:, c])),
                    "fpr_neutral": float(np.nanmean(fpr_neu[:, c])),
                }
            res_L[v] = {"per_concept": per_c, "hyper": d["hyper"], "inner": d.get("inner", [])}
        results[str(L)] = res_L
        stats.clear()
        lr_cache.clear()
        print(f"[sweep {cat}] L{L} total {time.time() - tl:.0f}s", flush=True)

    out = {
        "category": cat,
        "concepts": order,
        "layers": layers,
        "n_texts": len(texts),
        "n_items": len(items),
        "results": results,
        "runtime_s": round(time.time() - t0, 1),
        "groups_note": "V1/V3: (ending, offset) groups for offset>=9, offset-only groups (endings pooled) for tail offsets 0..8",
    }
    d = OUT_MOUNT / cfg["run"]
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{cat}.json").write_text(json.dumps(out))
    out_vol.commit()
    return out


@app.local_entrypoint()
def main(
    inputs: str = "outputs/sibling/probes/inputs.json",
    items_json: str = "data/sibling_probe/items.json",
    eval_acts: str = "acts-sibling-eval",
    probe_acts: str = "acts-sibling-probe",
    run: str = "probe_sweep",
    out_dir: str = "outputs/sibling/probe_sweep",
    layers: str = "28,36,42,44,52",
    variants: str = "V0,V1,V2,V3,V4",
    categories: str = "",
) -> None:
    inp = json.loads(Path(inputs).read_text())
    named = {
        it["id"]: it.get("named") or [] for it in json.loads(Path(items_json).read_text())["items"]
    }
    for it in inp["items"]:
        it["named"] = named.get(it["id"], [])
    inp.pop("token_ids", None)
    cfg = {
        "eval_acts": eval_acts,
        "probe_acts": probe_acts,
        "run": run,
        "layers": [int(x) for x in layers.split(",")],
        "variants": variants.split(","),
    }
    cats = categories.split(",") if categories else list(inp["concepts"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for res in sweep_category.map(cats, kwargs={"inputs": inp, "cfg": cfg}):
        (out / f"{res['category']}.json").write_text(json.dumps(res))
        print(f"[sweep] {res['category']} done ({res['runtime_s']}s)")
    (out / "config.json").write_text(
        json.dumps({**cfg, "categories": cats, "wall_s": round(time.time() - t0)})
    )
    print(f"[sweep] wrote {len(cats)} categories -> {out}")
