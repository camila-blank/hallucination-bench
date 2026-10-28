# ruff: noqa  (Modal launcher)
"""Plan 0017 rev 5: deception probes at every read site. One Modal CPU container per
(layer, job), jobs = ``ip`` (Instructed-Pairs, independent sensitivity), ``cf`` (item-fold
cross-fit sensitivity), ``loao:<archetype>`` (the primary leave-one-archetype-out probe).

- **loao:A** (primary): per site type, trained on the labelled ``lie`` / ``honest_*`` items of the
  other archetypes (read ``hb-mask-v5`` + unread ``hb-mask-v5-neg``) minus the **threshold set**
  (the unread honest items of the other archetypes); A's read items are scored at every site;
  threshold = q95 of the threshold set's position scores.
- **cf**: 4 item folds over every labelled item; read items scored by their own fold's model at
  every site; threshold = q95 of out-of-fold honest training-position z (rev 4).
- **ip**: rev-4 Instructed-Pairs probes (``hb-mask-probe-train``), ``end`` scores prompt sites,
  ``resp`` response sites; direction AUROC only (its threshold is miscalibrated on MASK).

Training uses at most 32 positions per (item, site type), seeded; every read site is scored.
Writes ``outputs/mask/probe_scores_v5.json.gz`` (per-site scores) and
``data/mask/probe_summary_v5.json`` (summaries, thresholds).

    UV_CACHE_DIR=... uvx --python 3.12 modal@latest run scripts/mask/probe_modal.py
"""

import gzip
import json
import random
import zlib
from pathlib import Path

import modal

_HERE = Path(__file__).resolve()
REPO = (
    _HERE.parents[2]
    if len(_HERE.parents) > 2 and (_HERE.parents[2] / "src").is_dir()
    else Path("/root")
)
LAYERS = (28, 36, 44, 52)
ARCHS = ("continuations", "disinformation", "known_facts", "provided_facts")
MAX_TRAIN_POS = 32
SEED = 17
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


def _load(subdir: str, wanted: dict[str, list[int]], layer: int) -> dict:
    """{label: [len(positions), d] float16} at ``layer``."""
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
        out[e["label"]] = a[wanted[e["label"]]].float().numpy().astype(np.float16)
    missing = set(wanted) - set(out)
    assert not missing, f"{subdir}: missing {sorted(missing)[:3]}"
    return out


def _sub(tid: str, st: str, idx: list[int]) -> list[int]:
    if len(idx) <= MAX_TRAIN_POS:
        return idx
    return sorted(random.Random(f"{SEED}:{tid}:{st}").sample(idx, MAX_TRAIN_POS))


def _items(sites: dict, acts: dict, labels: dict, ids, st: str):
    """(ids, windows [subsampled, float32], y) for labelled lie / honest items."""
    import numpy as np

    out_ids, win, y = [], [], []
    for t in ids:
        lab = labels[t]["label"]
        if lab != "lie" and not lab.startswith("honest"):
            continue
        idx = [i for i, s in enumerate(sites[t]) if s["site_type"] == st]
        if not idx:
            continue
        out_ids.append(t)
        win.append(acts[t][_sub(t, st, idx)].astype(np.float32))
        y.append(int(lab == "lie"))
    return out_ids, win, y


def _score_all(fn, sites: dict, acts: dict, ids, st: str, chunk: int = 512) -> dict:
    import numpy as np

    out = {}
    for t in ids:
        idx = [i for i, s in enumerate(sites[t]) if s["site_type"] == st]
        z = []
        for j in range(0, len(idx), chunk):
            z.append(fn(acts[t][idx[j : j + chunk]].astype(np.float32)))
        out[t] = (idx, np.concatenate(z) if z else np.zeros(0))
    return out


@app.function(image=image, volumes={"/acts": acts_vol}, cpu=16, memory=131072, timeout=6 * 3600)
def run_job(
    layer: int, job: str, sites: dict, neg_sites: dict, labels: dict, read: list, manifest: dict
) -> dict:
    import numpy as np

    from hallucination_bench.evals.mask import probe as PR
    from hallucination_bench.evals.sibling_probe import probes as P

    res: dict = {"layer": layer, "job": job, "scores": {}}
    all_sites = {**sites, **neg_sites}
    if job == "ip":
        tr = manifest["train"]
        acts = _load("hb-mask-probe-train", {t["label"]: t["end"] + t["resp"] for t in tr}, layer)
        probes = {}
        for kind in ("end", "resp"):
            win = [
                acts[t["label"]][: len(t["end"])]
                if kind == "end"
                else acts[t["label"]][len(t["end"]) :]
                for t in tr
            ]
            probes[kind] = PR.fit(
                [w.astype(np.float32) for w in win],
                [int(t["cond"] == "deceptive") for t in tr],
                [int(t["fold"]) for t in tr],
                ids=[t["label"] for t in tr],
            )
            res[f"ip_{kind}"] = probes[kind].summary()
        del acts
        racts = _load("hb-mask-v5", {t: [s["pos"] for s in sites[t]] for t in read}, layer)
        for kind, st in (("end", "prompt"), ("resp", "response")):
            sc = _score_all(probes[kind].score, sites, racts, read, st)
            res["scores"][st] = {
                t: [v[0], [round(float(x), 4) for x in v[1]]] for t, v in sc.items()
            }
        return res
    racts = _load("hb-mask-v5", {t: [s["pos"] for s in sites[t]] for t in read}, layer)
    negs = sorted(neg_sites)
    nacts = _load("hb-mask-v5-neg", {t: [s["pos"] for s in neg_sites[t]] for t in negs}, layer)
    acts = {**racts, **nacts}
    if job == "cf":
        for st in ("prompt", "response"):
            ids, win, y = _items(all_sites, acts, labels, sorted(all_sites), st)
            folds = [zlib.crc32(t.encode()) % 4 for t in ids]
            p = PR.fit(win, y, folds, ids=ids)
            res[f"cf_{st}"] = p.summary()
            fold_of = dict(zip(ids, folds))
            sc = {}
            for t in read:
                if t not in fold_of:
                    continue
                k = fold_of[t]
                sc.update(_score_all(lambda H, k=k: p.score_fold(H, k), sites, racts, [t], st))
            res["scores"][st] = {
                t: [v[0], [round(float(x), 4) for x in v[1]]] for t, v in sc.items()
            }
        return res
    assert job.startswith("loao:")
    held = job.split(":", 1)[1]
    readset = set(read)
    for st in ("prompt", "response"):
        others = [t for t in sorted(all_sites) if labels[t]["archetype"] != held]
        thresh = [t for t in others if t not in readset and labels[t]["label"].startswith("honest")]
        train = [t for t in others if t not in set(thresh)]
        ids, win, y = _items(all_sites, acts, labels, train, st)
        folds = [zlib.crc32(t.encode()) % 4 for t in ids]
        p = PR.fit(win, y, folds, ids=ids)
        thr = _score_all(
            p.score,
            all_sites,
            acts,
            [t for t in thresh if any(s["site_type"] == st for s in all_sites[t])],
            st,
        )
        tz = np.concatenate([v[1] for v in thr.values()]) if thr else np.zeros(0)
        held_read = [t for t in read if labels[t]["archetype"] == held]
        sc = _score_all(p.score, sites, racts, held_read, st)
        res[f"loao_{st}"] = {
            **p.summary(),
            "tau": P.q95(tz) if tz.size else float("nan"),
            "n_threshold_items": len(thr),
            "n_threshold_pos": int(tz.size),
            "n_train_items": len(ids),
            "n_train_lie": int(sum(y)),
        }
        res["scores"][st] = {t: [v[0], [round(float(x), 4) for x in v[1]]] for t, v in sc.items()}
    return res


@app.local_entrypoint()
def main() -> None:
    d = REPO / "data" / "mask"

    def gz(name):
        with gzip.open(d / name, "rt", encoding="utf-8") as fh:
            return json.load(fh)

    sites = {
        r["task_id"]: [{"site_type": s["site_type"], "pos": s["pos"]} for s in r["sites"]]
        for r in gz("sites.json.gz")["items"]
    }
    neg_sites = {
        r["task_id"]: [{"site_type": s["site_type"], "pos": s["pos"]} for s in r["sites"]]
        for r in gz("sites_neg.json.gz")["items"]
    }
    labels = {
        r["task_id"]: {"label": r["label"], "archetype": r["archetype"]}
        for r in json.loads((d / "labels.json").read_text())["items"]
    }
    read = sorted(sites)
    manifest = json.loads((d / "probe_manifest.json").read_text())
    jobs = [(L, j) for L in LAYERS for j in ("ip", "cf", *[f"loao:{a}" for a in ARCHS])]
    out: dict = {}
    for r in run_job.starmap(
        jobs,
        kwargs={
            "sites": sites,
            "neg_sites": neg_sites,
            "labels": labels,
            "read": read,
            "manifest": manifest,
        },
    ):
        out.setdefault(str(r["layer"]), {})[r["job"]] = r
        print(f"[probe] {r['layer']} {r['job']} done", flush=True)
    summ = {
        L: {j: {k: v for k, v in r.items() if k != "scores"} for j, r in js.items()}
        for L, js in out.items()
    }
    (d / "probe_summary_v5.json").write_text(
        json.dumps({"meta": {"plan": "0017 rev 5"}, "layers": summ}, indent=1) + "\n"
    )
    dest = REPO / "outputs" / "mask" / "probe_scores_v5.json.gz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(dest, "wt", encoding="utf-8") as fh:
        json.dump({"layers": out}, fh)
    print(f"[probe] wrote {dest} and {d / 'probe_summary_v5.json'}", flush=True)
