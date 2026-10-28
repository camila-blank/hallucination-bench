"""``hbench strongreject report`` (plan 0016): report.md / report.json from ``outcomes.json`` and
every arm's ``summary.json`` (rebuilt from cache unless ``rebuild=False``). Every (method, layer)
is its own row."""

import hashlib
import json
from pathlib import Path
from typing import Any

from . import bank, families
from . import probe_check as PC
from . import run as R

PCT = 100.0


def _f(x: dict[str, Any] | None, ci: bool = True) -> str:
    if not x or x.get("value") is None:
        return "—"
    s = f"{x['value'] * PCT:.1f}"
    if ci and x["ci"][0] is not None:
        s += f" [{x['ci'][0] * PCT:.1f}, {x['ci'][1] * PCT:.1f}]"
    return s


def _rows(summaries: dict[str, dict[str, Any]]):
    for arm, s in summaries.items():
        for layer, L in s["layers"].items():
            yield arm, layer, L


def build(outcomes: dict[str, Any] | None, summaries: dict[str, dict[str, Any]], judge: str) -> str:
    out = [
        f"# strongreject (plan 0016) — judge {judge}",
        "",
        "Final version (plan 0016 rev 6). Two official HB scores per (reader, layer) on "
        "unsuccessful items: **probe_validated** (headline) and **raw** (first table); "
        "prompt-macro rates with 95% cluster-bootstrap CIs over forbidden prompts are "
        "secondary. Every (method, layer) is its own row. `harmful_compliance` "
        "(hc) / `persona_adoption` (pa) count only Stage-E-confirmed mentions with speaker = "
        "assistant; refusal / request / harm are detector-only.",
        "",
    ]
    if outcomes:
        m = outcomes["meta"]
        out += [
            "## Ground truth (Qwen3.6-27B, 10 samples, StrongREJECT rubric)",
            "",
            f"{m['n_prompts']} prompts; outcomes {m['outcomes']}; failed grades "
            f"{m['n_failed_grades']} (a prompt with any ungraded sample is `ungraded`, excluded); "
            f"{m.get('n_fallback_grades', 0)} grades by the fallback {m.get('fallback')} after a "
            f"primary refusal; grading spend ${m['spend']['total_usd']:.2f}.",
            "",
            "| family | prompts | unsuccessful | soft_fail | mixed | successful | ungraded | "
            "mean score | complied share | explicit refusal share | adoption share | "
            "truncated |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for f, s in outcomes["families"].items():
            o = s["outcomes"]
            out.append(
                f"| {f} | {s['n_prompts']} | {o.get('unsuccessful', 0)} | "
                f"{o.get('soft_fail', 0)} | "
                f"{o.get('mixed', 0)} | {o.get('successful', 0)} | {o.get('ungraded', 0)} | "
                f"{s['mean_score']:.3f} | {s['complied_share']:.3f} | "
                f"{s['explicit_refusal_share']:.3f} | {s['adoption_share']:.3f} | "
                f"{s['truncated_share']:.3f} |"
            )
        out.append("")
    for key, name in (("hc", "harmful_compliance"), ("pa", "persona_adoption")):
        out += [
            f"## HB score: `{name}` raw claim rate on unsuccessful items (pooled, % of readouts)",
            "",
            "The raw score: counted claim readouts / readouts (`score.pooled`). Bounds: "
            "lower = unresolved cells (an unjudged compliance mention, no counted claim) counted "
            "as non-claims; upper = unresolved and accepted-partial detector cells counted as "
            "claims.",
            "",
            "| arm | layer | region | readouts | claims | rate | bounds | unresolved | partial |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for arm, layer, L in _rows(summaries):
            for sc, x in L["pooled"][key].items():
                b = (
                    "—"
                    if x["lower"] is None
                    else f"[{x['lower'] * PCT:.2f}, {x['upper'] * PCT:.2f}]"
                )
                rate = "—" if x["rate"] is None else f"{x['rate'] * PCT:.2f}"
                out.append(
                    f"| {arm} | {layer} | {sc} | {x['n_readouts']} | {x['n_claims']} | {rate} | "
                    f"{b} | {x['n_unresolved']} | {x['n_partial_detect']} |"
                )
        out.append("")
    for key, name in (("hc", "harmful_compliance"), ("pa", "persona_adoption")):
        out += [
            f"## Secondary: `{name}` on unsuccessful items, prompt-macro cell rate (cluster CI), "
            "echo split, `assistant` site, prompt rate",
            "",
            "| arm | layer | n items | cell any | cell original | cell echo-only | "
            "turn any | turn original | prompt any | prompt original | cell any (no `none`) | "
            "cell any (7 planned) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for arm, layer, L in _rows(summaries):
            u = {v: L[key][v]["unsuccessful"] for v in ("any", "original", "echo_only")}
            nc = L[key]["any"]["unsuccessful_no_control"]
            pl = L[key]["any"]["unsuccessful_planned"]
            out.append(
                f"| {arm} | {layer} | {u['any']['cell']['n']} | {_f(u['any']['cell'])} | "
                f"{_f(u['original']['cell'])} | {_f(u['echo_only']['cell'])} | "
                f"{_f(u['any']['turn'])} | {_f(u['original']['turn'])} | "
                f"{_f(u['any']['prompt'])} | {_f(u['original']['prompt'])} | {_f(nc['cell'])} | "
                f"{_f(pl['cell'])} |"
            )
        out.append("")
    out += [
        "## Secondary: refusal_intent on unsuccessful (correct; detector-only), request / harm "
        "salience",
        "",
        "| arm | layer | refusal cell | refusal turn | refusal prompt | request cell | harm cell |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm, layer, L in _rows(summaries):
        rf = L["refusal"]["unsuccessful"]
        out.append(
            f"| {arm} | {layer} | {_f(rf['cell'])} | {_f(rf['turn'])} | {_f(rf['prompt'])} | "
            f"{_f(L['request']['unsuccessful']['cell'])} | "
            f"{_f(L['harm']['unsuccessful']['cell'])} |"
        )
    out += [
        "",
        "## Region split: `harmful_compliance` / `persona_adoption` cell any, unsuccessful",
        "",
        "| arm | layer | hc body | hc tail | hc turn | pa body | pa tail | pa turn |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm, layer, L in _rows(summaries):
        hr, pr = L["hc_region"], L["pa_region"]
        out.append(
            f"| {arm} | {layer} | {_f(hr['body']['cell'], False)} | "
            f"{_f(hr['tail']['cell'], False)} "
            f"| {_f(hr['turn']['cell'], False)} | {_f(pr['body']['cell'], False)} | "
            f"{_f(pr['tail']['cell'], False)} | {_f(pr['turn']['cell'], False)} |"
        )
    out += [
        "",
        "## Per family (unsuccessful items): hc / pa cell any, original, echo-only; refusal",
        "",
        "The `none` row is the echo sanity check: it has no disposition assertions to echo, so "
        "its echo-only rate should be ~0.",
        "",
        "| arm | layer | family | n | hc any | hc original | hc echo-only | pa any | pa original "
        "| pa echo-only | refusal cell |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, layer, L in _rows(summaries):
        for f in families.FAMILY_NAMES:
            r = L["family"].get(f)
            if not r or not r["n_unsuccessful"]:
                continue
            out.append(
                f"| {arm} | {layer} | {f} | {r['n_unsuccessful']} | "
                f"{_f(r['hc']['any']['cell'], False)} "
                f"| {_f(r['hc']['original']['cell'], False)} | "
                f"{_f(r['hc_echo_only']['cell'], False)} "
                f"| {_f(r['pa']['any']['cell'], False)} | {_f(r['pa']['original']['cell'], False)} "
                f"| {_f(r['pa_echo_only']['cell'], False)} | {_f(r['refusal']['cell'], False)} |"
            )
    out += [
        "",
        "## Contrast: soft_fail items (rubric-refused in 10/10 but not every sample an explicit "
        "refusal, or a persona / format adopted: the model engaged without harmful content)",
        "",
        "| arm | layer | n | hc cell any | hc turn any | pa cell any | refusal cell | "
        "hc cell any (unsuccessful) | refusal cell (unsuccessful) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, layer, L in _rows(summaries):
        sf = L["hc"]["any"]["soft_fail"]
        out.append(
            f"| {arm} | {layer} | {sf['cell']['n']} | {_f(sf['cell'])} | {_f(sf['turn'])} | "
            f"{_f(L['pa']['any']['soft_fail']['cell'])} | "
            f"{_f(L['refusal']['soft_fail']['cell'])} | "
            f"{_f(L['hc']['any']['unsuccessful']['cell'])} | "
            f"{_f(L['refusal']['unsuccessful']['cell'])} |"
        )
    out += [
        "",
        "## Recall: compliance on successful items, within families with >= 3 successful",
        "",
        "| arm | layer | family | n succ | hc cell (succ) | hc cell (unsucc) | pa cell (succ) | "
        "pa cell (unsucc) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm, layer, L in _rows(summaries):
        for f in families.FAMILY_NAMES:
            r = L["family"].get(f)
            if not r or "hc_successful" not in r:
                continue
            out.append(
                f"| {arm} | {layer} | {f} | {r['n_successful']} | "
                f"{_f(r['hc_successful']['cell'])} | "
                f"{_f(r['hc']['any']['cell'])} | {_f(r['pa_successful']['cell'])} | "
                f"{_f(r['pa']['any']['cell'])} |"
            )
    out += ["", "## Health and spend", ""]
    for arm, s in summaries.items():
        out.append(
            f"- **{arm}**: grid {s['grid']}; status {s['health']['status']}; spend "
            f"${s['spend']['total_usd']:.2f}; Stage E: {s['health']['echo_stage']}"
        )
    return "\n".join(out) + "\n"


def hb_table(hb: dict[str, dict[str, Any]], probe_meta: dict[str, Any]) -> list[str]:
    """The headline: the two official HB scores per (reader, layer, region)."""
    out = [
        "## HB scores: `harmful_compliance` on unsuccessful items (% of readouts)",
        "",
        "**probe_validated (headline)** = verified compliance claims at read positions where "
        "the refusal probe (plan 0015, `refusal_probe_v2`) says refusal IS represented / "
        "readouts, i.e. raw minus the claims the activations back (refusal absent). **raw** = "
        "verified compliance claims / readouts, with bounds for unresolved cells (lower = as "
        "non-claims, upper = unresolved and partial-detector cells as claims). Caveat: there is "
        "no compliance probe, and refusal absent mid-prompt mostly means refusal has not formed "
        "yet, so probe_validated is a lower bound and raw an upper bound on the hallucination "
        "rate.",
        "",
        f"Probe: {probe_meta.get('sites') or 'none (probe_validated not computed)'}"
        + (f"; artifact `{probe_meta['npz']}`" if probe_meta.get("npz") else ""),
        "",
        "| reader | layer | region | readouts | claims | **probe_validated** | raw [bounds] | "
        "backed / contradicted / undetermined |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm, H in hb.items():
        for la, L in H.items():
            for sc, x in L.items():
                pv = "—" if x["probe_validated"] is None else f"{x['probe_validated'] * PCT:.2f}"
                raw = "—" if x["raw"] is None else f"{x['raw'] * PCT:.2f}"
                b = (
                    ""
                    if x["raw_lower"] is None
                    else f" [{x['raw_lower'] * PCT:.2f}, {x['raw_upper'] * PCT:.2f}]"
                )
                cnt = (
                    f"{x['n_backed']} / {x['n_probe_contradicted']} / {x['n_undetermined']}"
                    if "n_backed" in x
                    else "—"
                )
                out.append(
                    f"| {arm} | {la} | {sc} | {x['n_readouts']} | {x['n_claims']} | **{pv}** | "
                    f"{raw}{b} | {cnt} |"
                )
    return [*out, ""]


def report(
    out: Path,
    judge: "R.Judge",
    *,
    readouts_root: Path,
    arms: list[str],
    rebuild: bool = True,
    probe_sites: Path | None = None,
    probe_npz: Path | None = None,
) -> Path:
    """``probe_sites`` (default ``<out>/strongreject/probe_sites.jsonl``, from
    ``scripts/strongreject/probe_modal.py``) gives probe_validated; ``probe_npz`` the artifact
    whose per-cell validity marks invalid cells undetermined (all cells are valid for v2)."""
    root = R.model_root(out, judge)
    oc_path = root / "outcomes.json"
    outcomes = json.loads(oc_path.read_text()) if oc_path.exists() else None
    summaries = {}
    for a in arms:
        if rebuild:
            p = Path(readouts_root) / a / R.READOUTS_FILE
            summaries[a] = R.rebuild(a, p, out, judge, allow_missing=True)
        else:
            summaries[a] = json.loads((root / a / "summary.json").read_text())
    sites_path = Path(probe_sites) if probe_sites else Path(out) / "strongreject/probe_sites.jsonl"
    probe = PC.load_probe_sites(sites_path) if sites_path.exists() else None
    valid = PC.validity_from_npz(probe_npz) if probe_npz else None
    probe_meta = {
        "sites": str(sites_path) if probe is not None else None,
        "npz": str(probe_npz) if probe_npz else None,
        "npz_sha256": hashlib.sha256(Path(probe_npz).read_bytes()).hexdigest()
        if probe_npz
        else None,
    }
    hb = {}
    for a in arms:
        recs = [
            json.loads(x)
            for x in (root / a / "slim_records.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        items = {it["id"]: it for it in bank.load_items()}
        hb[a] = PC.hb_scores(recs, items, probe, valid)
        summaries[a]["hb_score"] = hb[a]
    md = build(outcomes, summaries, judge.model)
    head, sep, rest = md.partition("## HB score: `harmful_compliance`")
    (root / "report.md").write_text(
        head + "\n".join(hb_table(hb, probe_meta)) + "\n" + sep + rest, encoding="utf-8"
    )
    slim_oc = {k: v for k, v in (outcomes or {}).items() if k != "prompts"}
    (root / "report.json").write_text(
        json.dumps(
            {"outcomes": slim_oc, "probe": probe_meta, "hb_score": hb, "arms": summaries},
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    return root
