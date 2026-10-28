"""``hbench sad report`` (plan 0018 §Report): rebuild every arm dir under ``<out>/sad/<judge>/``
from its readouts + cache, then ``report.md`` / ``report.json`` beside them.

The report lives under gitignored ``outputs/`` and may show readouts; never copy a SAD text or
a readout that echoes one into a committed doc (paraphrase)."""

import json
import random
from pathlib import Path
from typing import Any

from . import bank, run

HEAD_DIRS = ("deploy_on_eval", "eval_on_deploy")
CTX_DIRS = ("eval_on_eval", "deploy_on_deploy")


def _f(st: dict[str, Any] | None, ci: bool = True) -> str:
    if not st or st.get("mean") is None:
        return "—"
    s = f"{st['mean']:.3f}"
    if ci and st.get("lo") is not None:
        s += f" [{st['lo']:.3f}, {st['hi']:.3f}]"
    return s


def examples(arm_root: Path, items: dict[str, Any], n: int, seed: int = 0) -> list[dict[str, Any]]:
    """Seeded flagged readouts: wrong-direction headline mentions first, then correct ones."""
    wrong, right = [], []
    with (arm_root / "cells.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            for m in r["mentions"]:
                if m.get("about") != "situation" or m.get("verified") is not True:
                    continue
                b = items[r["id"]]["belief"]
                ex = {
                    "id": r["id"],
                    "belief": b,
                    "provenance": items[r["id"]]["provenance"],
                    "layer": r["layer"],
                    "pos": r["pos"],
                    "region": r["region"],
                    "label": m["label"],
                    "quote": m["quote"],
                    "readout": m["readout"],
                    "key": r["key"],
                }
                (wrong if m["label"] != f"{b}_belief" else right).append(ex)
    rng = random.Random(seed)
    rng.shuffle(wrong)
    rng.shuffle(right)
    return (wrong[: n // 2 + n % 2] + right)[:n] if len(wrong) >= n // 2 else (wrong + right)[:n]


def report(out: Path, judge: str, *, readouts_root: Path, n_examples: int = 10) -> Path:
    root = out / "sad" / judge
    selected = run.load_selected()
    items = {it["id"]: it for it in selected["items"]}
    gate = json.loads(bank.GATE_JSON.read_text())["meta"]
    arms = sorted(p.name for p in root.iterdir() if (p / "cache.jsonl").exists())
    summaries: dict[str, Any] = {}
    for arm in arms:
        cells = run.load_arm(readouts_root / arm / run.READOUTS_FILE, selected, None)
        eff = "low"
        summaries[arm] = run.rebuild(arm, cells, selected, out, judge, eff)
    L: list[str] = [f"# sad (plan 0018) — judge {judge}", ""]
    L += ["## Gate (Qwen3.6-27B's own belief, 10 samples)", ""]
    for k, v in gate["survival"].items():
        if k.startswith(">="):
            L.append(
                f"- {k}: n={v['n']} by belief {v['by_belief']} belief|provenance "
                f"{v['belief_x_provenance']} agree w/ provenance "
                f"{v['agree_with_provenance']:.3f}"
            )
    m = gate["survival"]["majority_all_items"]
    L.append(
        f"- majority over all items: n={m['n']} agree w/ provenance "
        f"{m['agree_with_provenance']:.3f} {m['belief_x_provenance']}"
    )
    L += ["", f"Selection: {selected['meta']}", ""]
    for variant in ("headline", "detector", "with_format"):
        L += [f"## Rates — variant `{variant}`", ""]
        L += [
            "| arm | layer | deploy_on_eval readout | deploy_on_eval final | eval_on_deploy "
            "readout | eval_on_deploy final | eval_on_eval readout | deploy_on_deploy readout | "
            "gap_eval readout |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for arm, s in summaries.items():
            for layer, row in s["scores"][variant].items():
                if layer == "excluded_low_coverage":
                    continue
                L.append(
                    f"| {arm} | {layer} | {_f(row['deploy_on_eval']['readout'])} | "
                    f"{_f(row['deploy_on_eval']['final'])} | "
                    f"{_f(row['eval_on_deploy']['readout'])} | "
                    f"{_f(row['eval_on_deploy']['final'])} | "
                    f"{_f(row['eval_on_eval']['readout'])} | "
                    f"{_f(row['deploy_on_deploy']['readout'])} | "
                    f"{_f(row['gap_eval']['readout'])} |"
                )
        L.append("")
    L += ["## Belief × provenance cells (headline, readout rate)", ""]
    L += ["| arm | layer | direction | prov=eval | prov=deploy |", "|---|---|---|---|---|"]
    for arm, s in summaries.items():
        for layer, row in s["scores"]["headline"].items():
            if layer == "excluded_low_coverage":
                continue
            for d in (*HEAD_DIRS, *CTX_DIRS):
                pe, pd = row[f"{d}|prov=eval"]["readout"], row[f"{d}|prov=deploy"]["readout"]
                L.append(
                    f"| {arm} | {layer} | {d} | {_f(pe)} (n={pe['n']}) | {_f(pd)} (n={pd['n']}) |"
                )
    L += ["", "## Suffix item rate (headline; compare within an arm only)", ""]
    for arm, s in summaries.items():
        for layer, row in s["scores"]["headline"].items():
            if layer == "excluded_low_coverage":
                continue
            L.append(
                f"- {arm} L{layer}: "
                + ", ".join(
                    f"{d} {_f(row[d]['suffix_item'], False)}" for d in (*HEAD_DIRS, *CTX_DIRS)
                )
            )
    L += ["", "## Health and spend", ""]
    for arm, s in summaries.items():
        L.append(
            f"- {arm}: cells {s['n_cells']}, health {s['health']}, situation mentions "
            f"{s['n_situation_mentions']}, verify sample {s['n_verify_sample']} "
            f"(sampled={s['verify_sampled']}), spend ${s['spend']['total_usd']:.2f}, items "
            f"excluded for coverage < 0.8: "
            f"{ {k: len(v) for k, v in s['scores']['excluded_low_coverage'].items()} }"
        )
    L += ["", "## Examples (verified situation mentions; wrong direction first)", ""]
    for arm in arms:
        L.append(f"### {arm}")
        for ex in examples(root / arm, items, n_examples):
            L.append(
                f"- {ex['id']} belief={ex['belief']} prov={ex['provenance']} L{ex['layer']} "
                f"p{ex['pos']} ({ex['region']}) **{ex['label']}** r{ex['readout']}: "
                f"{json.dumps(ex['quote'], ensure_ascii=False)}"
            )
        L.append("")
    (root / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (root / "report.json").write_text(
        json.dumps({"gate": gate, "selection": selected["meta"], "arms": summaries}, indent=1)
        + "\n"
    )
    return root
