"""``hbench strongreject report`` (plan 0016): report.md / report.json from ``outcomes.json`` and
every arm's ``summary.json`` (rebuilt from cache unless ``rebuild=False``). Every (method, layer)
is its own row."""

import json
from pathlib import Path
from typing import Any

from . import families
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
        "Final version (plan 0016 rev 5). The HB score is the pooled claim rate on unsuccessful "
        "items (first tables); prompt-macro rates with 95% cluster-bootstrap CIs over forbidden "
        "prompts are secondary. Every (method, layer) is its own row. `harmful_compliance` "
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
            f"## HB score: `{name}` claim rate on unsuccessful items (pooled, % of readouts)",
            "",
            "THE strongreject score: counted claim readouts / readouts (`score.pooled`). Bounds: "
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


def report(
    out: Path,
    judge: "R.Judge",
    *,
    readouts_root: Path,
    arms: list[str],
    rebuild: bool = True,
) -> Path:
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
    (root / "report.md").write_text(build(outcomes, summaries, judge.model), encoding="utf-8")
    slim_oc = {k: v for k, v in (outcomes or {}).items() if k != "prompts"}
    (root / "report.json").write_text(
        json.dumps({"outcomes": slim_oc, "arms": summaries}, indent=1) + "\n", encoding="utf-8"
    )
    return root
