"""Plan 0018 rev 6 outputs from the probe job (``outputs/sad/probes/L{layer}.json``) and the
arms' ``cells.jsonl``:

1. **premise check** -- per (layer, position type), the share of deploy-belief pilot items whose
   site fires the EVAL probe, and of eval-belief items whose site fires the DEPLOY probe (body:
   the share of body sites; each suffix position: the share of items);
2. **probe-checked claims** -- every verified literal claim is tagged probe-supported (the
   matching probe fires at that exact (item, layer, site)) or not; the probe-checked
   hallucination rate = item-weighted share of readouts carrying an UNSUPPORTED verified claim,
   beside the gate-based rates (wrong direction vs Qwen's belief);
3. **misses** -- of the sites where a probe fires, the share where the lens states nothing of
   that kind (no verified claim of the matching label in any readout).

Probe = ``primary_wildchat`` (headline); ``primary_all`` and ``secondary`` are sensitivity rows."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROBES = ("primary_wildchat", "primary_all", "secondary")
POS_NAMES = {0: "assistant", 1: "<|im_start|>", 2: "\\n", 3: "<|im_end|>"}
LABEL_PROBE = {"eval_claim": "eval", "deploy_claim": "deploy"}


def load_probes(d: Path) -> dict[int, dict[str, Any]]:
    return {int(p.stem[1:]): json.loads(p.read_text()) for p in sorted(d.glob("L*.json"))}


def site_table(res: dict[str, Any], inputs: dict[str, Any], probe: str) -> dict:
    """``{(item id, pos): {eval, deploy, key}}`` for one layer and probe."""
    out = {}
    pil = inputs["pilot"]
    for s in res[probe]["pilot_sites"]:
        it = pil[s["i"]]
        out[(it["id"], it["pos"][s["j"]])] = {
            "eval": s["eval"],
            "deploy": s["deploy"],
            "key": tuple(it["keys"][s["j"]]),
        }
    return out


def premise(probes: dict[int, dict], inputs: dict, probe: str) -> dict:
    belief = {p["id"]: ("eval" if p["label"] == 0 else "deploy") for p in inputs["pilot"]}
    out: dict[str, Any] = {}
    for layer, res in probes.items():
        st = site_table(res, inputs, probe)
        row: dict[str, Any] = {}
        for name, key in [("body", (0, 0))] + [(POS_NAMES[j], (1, j)) for j in (3, 2, 1, 0)]:
            for grp, fire in (("deploy", "eval"), ("eval", "deploy")):
                vals = [v[fire] for (i, _), v in st.items() if belief[i] == grp and v["key"] == key]
                row[f"{name}|{fire}_fires_on_{grp}"] = float(np.mean(vals)) if vals else None
        # item level: fires at >= 1 of the 4 suffix sites
        for grp, fire in (("deploy", "eval"), ("eval", "deploy")):
            per: dict[str, bool] = defaultdict(bool)
            for (i, _), v in st.items():
                if belief[i] == grp and v["key"][0] == 1:
                    per[i] |= v[fire]
            row[f"any_suffix|{fire}_fires_on_{grp}"] = float(np.mean(list(per.values())))
        out[str(layer)] = row
    return out


def claims_and_misses(
    probes: dict[int, dict], inputs: dict, cells_path: Path, probe: str
) -> dict[str, Any]:
    belief = {p["id"]: ("eval" if p["label"] == 0 else "deploy") for p in inputs["pilot"]}
    tables = {L: site_table(r, inputs, probe) for L, r in probes.items()}
    per_layer: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "claims": [],
            "item_unsup": defaultdict(lambda: [0, 0]),
            "item_gate": defaultdict(lambda: [0, 0]),
            "fires": {"eval": [0, 0], "deploy": [0, 0]},
        }
    )
    with cells_path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            L = r["layer"]
            if L not in tables or r["status"] != "ok":
                continue
            site = tables[L].get((r["id"], r["pos"]))
            if site is None:
                continue
            acc = per_layer[L]
            unsup = [0] * r["k"]
            wrong = [0] * r["k"]
            said = {"eval": False, "deploy": False}
            for m in r["mentions"]:
                if not m.get("verified"):
                    continue
                p = LABEL_PROBE[m["label"]]
                said[p] = True
                sup = bool(site[p])
                acc["claims"].append(
                    {
                        "id": r["id"],
                        "pos": r["pos"],
                        "label": m["label"],
                        "belief": belief[r["id"]],
                        "supported": sup,
                        "quote": m["quote"][:160],
                    }
                )
                if not sup:
                    unsup[m["readout"]] = 1
                if p != belief[r["id"]]:
                    wrong[m["readout"]] = 1
            a = acc["item_unsup"][r["id"]]
            a[0] += sum(unsup)
            a[1] += r["k"]
            g = acc["item_gate"][r["id"]]
            g[0] += sum(wrong)
            g[1] += r["k"]
            for p in ("eval", "deploy"):
                if site[p]:
                    acc["fires"][p][1] += 1
                    acc["fires"][p][0] += 0 if said[p] else 1
    out = {}
    for L, acc in sorted(per_layer.items()):
        cl = acc["claims"]
        out[str(L)] = {
            "n_verified_claims": len(cl),
            "n_supported": sum(c["supported"] for c in cl),
            "probe_checked_rate": float(
                np.mean([a / b for a, b in acc["item_unsup"].values() if b])
            ),
            "gate_rate": float(np.mean([a / b for a, b in acc["item_gate"].values() if b])),
            "n_sites_eval_fires": acc["fires"]["eval"][1],
            "miss_eval": acc["fires"]["eval"][0] / acc["fires"]["eval"][1]
            if acc["fires"]["eval"][1]
            else None,
            "n_sites_deploy_fires": acc["fires"]["deploy"][1],
            "miss_deploy": acc["fires"]["deploy"][0] / acc["fires"]["deploy"][1]
            if acc["fires"]["deploy"][1]
            else None,
            "claims": cl,
        }
    return out


def aurocs(probes: dict[int, dict]) -> dict[str, Any]:
    out = {}
    for L, r in probes.items():
        out[str(L)] = {
            p: {
                k: r[p].get(k)
                for k in (
                    "auroc_item",
                    "auroc_assistant",
                    "auroc_body",
                    "transfer_auroc_item",
                    "transfer_auroc_assistant",
                    "n_items",
                    "n_by_label",
                    "lams",
                )
            }
            for p in PROBES
        }
    return out


def report(out: Path, judge: str, probes_dir: Path, inputs_path: Path) -> dict[str, Any]:
    probes = load_probes(probes_dir)
    inputs = json.loads(inputs_path.read_text())
    root = out / "sad" / judge
    arms = sorted(p.name for p in root.iterdir() if (p / "cells.jsonl").exists())
    from hallucination_bench.evals.sad import bank

    selected = json.loads(bank.SELECTED_JSON.read_text())
    doc: dict[str, Any] = {
        "aurocs": aurocs(probes),
        "transfer": transfer_checks(probes, inputs, selected),
        "premise": {},
        "arms": {},
    }
    for probe in PROBES:
        doc["premise"][probe] = premise(probes, inputs, probe)
        doc["arms"][probe] = {
            a: claims_and_misses(probes, inputs, root / a / "cells.jsonl", probe) for a in arms
        }
    (root / "probes.json").write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def transfer_checks(probes: dict[int, dict], inputs: dict, selected: dict) -> dict[str, Any]:
    """Pilot transfer AUROC of the per-item mean eval score (all sites / the ``assistant`` site),
    on all 200 items, on WildChat items only (no benchmark format), and length-stratified
    (within prompt-length terciles of the WildChat items, pooled over pairs) -- because Qwen's
    gate belief tracks text length (short casual turns -> deploy)."""
    from hallucination_bench.evals.sibling_probe.probes import auroc

    meta = {it["id"]: it for it in selected["items"]}
    pil = inputs["pilot"]
    out: dict[str, Any] = {}
    for L, res in probes.items():
        row: dict[str, Any] = {}
        for probe in PROBES:
            per: dict[int, list] = defaultdict(list)
            last: dict[int, float] = {}
            for s in res[probe]["pilot_sites"]:
                per[s["i"]].append(s["s_eval"])
                if tuple(pil[s["i"]]["keys"][s["j"]]) == (1, 0):
                    last[s["i"]] = s["s_eval"]
            ids = [p["id"] for p in pil]
            lab = np.array([p["label"] for p in pil])
            m = np.array([np.nanmean(per[i]) for i in range(len(pil))])
            wc = np.array([meta[i]["source"].startswith("wildchat") for i in ids])
            ntok = np.array([meta[i]["n_tokens"] for i in ids], dtype=float)

            def au(sel: np.ndarray, x: np.ndarray = m, lab: np.ndarray = lab) -> float:
                return float(auroc(x[sel & (lab == 0)], x[sel & (lab == 1)]))

            # length terciles over the WildChat pilot items; AUROC pooled over within-bin pairs
            q = np.quantile(ntok[wc], [1 / 3, 2 / 3])
            b = np.digitize(ntok, q)
            num = den = 0.0
            for k in range(3):
                sel = wc & (b == k)
                pe, ne = m[sel & (lab == 0)], m[sel & (lab == 1)]
                if len(pe) and len(ne):
                    num += au(sel) * len(pe) * len(ne)
                    den += len(pe) * len(ne)
            a_last = np.array([last.get(i, np.nan) for i in range(len(pil))])
            row[probe] = {
                "all": au(np.ones(len(pil), bool)),
                "assistant": au(np.ones(len(pil), bool), a_last),
                "wildchat_only": au(wc),
                "wildchat_length_stratified": num / den if den else None,
                "length_alone_wildchat": float(auroc(ntok[wc & (lab == 0)], ntok[wc & (lab == 1)])),
            }
        out[str(L)] = row
    return out
