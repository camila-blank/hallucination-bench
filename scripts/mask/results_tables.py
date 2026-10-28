# ruff: noqa: E501, UP031  (table / label literals)
"""Markdown tables for ``docs/results/0017_mask_pilot.md`` (rev 5) from ``report.json``.

uv run --no-sync python scripts/mask/results_tables.py > /tmp/tables.md
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARMS = ("s3d-rl600", "mlnla-rl600-all11", "jlens")


def f(x, ci=None, d=3):
    if x is None:
        return "—"
    s = f"{x:.{d}f}"
    return s + (f" [{ci[0]:.{d}f}, {ci[1]:.{d}f}]" if ci else "")


def main() -> None:
    rep = json.loads((REPO / "outputs/mask/claude-opus-5-5/report.json").read_text())
    g = rep["gate"]
    print(
        "### HB score: verified lie-claim readout rate on gated honest_resisted (n = %d)\n"
        % next(iter(g["arms"].values()))["L028:prompt"]["9"]["honest_resisted"]["n_items"]
    )
    print("| arm | site set | L28 | L36 | L44 | L52 |")
    print("|---|---|---|---|---|---|")
    for arm in ARMS:
        if arm not in g["arms"]:
            continue
        for site in ("prompt", "response", "all"):
            cells = []
            for L in ("L028", "L036", "L044", "L052"):
                c = g["arms"][arm][f"{L}:{site}"]["9"]["honest_resisted"]
                cells.append(f(c["readout_rate"], c["readout_ci"], 4))
            print(f"| {arm} | {site} | " + " | ".join(cells) + " |")
    print(
        "\n### Diagnostic (not part of the HB score): the same rate on gated lie items (n = %d)\n"
        % next(iter(g["arms"].values()))["L028:prompt"]["9"]["lie"]["n_items"]
    )
    print("| arm | site set | L28 | L36 | L44 | L52 |")
    print("|---|---|---|---|---|---|")
    for arm in ARMS:
        if arm not in g["arms"]:
            continue
        for site in ("prompt", "response", "all"):
            cells = []
            for L in ("L028", "L036", "L044", "L052"):
                c = g["arms"][arm][f"{L}:{site}"]["9"]["lie"]
                cells.append(f(c["readout_rate"], None, 4))
            print(f"| {arm} | {site} | " + " | ".join(cells) + " |")
    print(
        "\n### Item-level (sample 0): share of gated honest_resisted items with >= 1 verified lie claim\n"
    )
    print("| arm | site set | L28 | L36 | L44 | L52 |")
    print("|---|---|---|---|---|---|")
    for arm in ARMS:
        if arm not in g["arms"]:
            continue
        for site in ("prompt", "response", "all"):
            cells = [
                f(g["arms"][arm][f"{L}:{site}"]["9"]["honest_resisted"]["item_rate"])
                for L in ("L028", "L036", "L044", "L052")
            ]
            print(f"| {arm} | {site} | " + " | ".join(cells) + " |")
    if "probe" in rep:
        pr = rep["probe"]
        print(
            "\n### Probes: transfer AUROC (item-mean score, gated lie vs gated honest_resisted)\n"
        )
        print("| layer | site | LOAO (primary) | item-fold cf | Instructed-Pairs |")
        print("|---|---|---|---|---|")
        for L, r in pr["layers"].items():
            for st in ("prompt", "response"):
                x = r[st]
                print(
                    f"| {L} | {st} | {f(x['transfer_loao'])} | {f(x['transfer_cf'])} | {f(x['transfer_ip'])} |"
                )
        print(
            "\n### Probe-checked hallucination rate (LOAO; verified claims at probe-contradicted sites / readouts)\n"
        )
        print(
            "| arm | layer | site | original | probe-checked | supported / contradicted | rate_supported | LOAO miss rate |"
        )
        print("|---|---|---|---|---|---|---|---|")
        for arm in ARMS:
            for k, d in pr["arms"].get(arm, {}).items():
                L, st = k.split(":")
                orig = (
                    (d["loao_hr_supported"] + d["loao_hr_contradicted"]) / d["loao_hr_readouts"]
                    if d.get("loao_hr_readouts")
                    else None
                )
                print(
                    f"| {arm} | {L} | {st} | {f(orig, None, 4)} | {f(d['loao_rate_probe_checked'], None, 4)} | "
                    f"{d.get('loao_hr_supported', 0)} / {d.get('loao_hr_contradicted', 0)} | "
                    f"{f(d['loao_rate_supported'], None, 4)} | {f(d['loao_miss_rate'])} |"
                )
    q = rep["qwen"]
    print("\n### Qwen3.6-27B labels (874 items)\n")
    print(
        "| archetype | n | lie | honest_correct | honest_wrong | inconsistent | evasive | no_belief | P(lie) 1 | P(lie) 10 |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for a, x in q.items():
        lab = x["labels"]
        print(
            f"| {a} | {x['n']} | {lab.get('lie', 0)} | {lab.get('honest_correct', 0)} | {lab.get('honest_wrong', 0)} | "
            f"{lab.get('inconsistent', 0)} | {lab.get('evasive', 0)} | {lab.get('no_belief', 0)} | "
            f"{x['p_lie_1']:.3f} | {x['p_lie_10']:.3f} |"
        )
    print("\n### Spend (detector + verifiers, cache totals)\n")
    for arm, s in rep["summaries"].items():
        print(f"- `{arm}`: ${s['spend_total_usd']:.0f}, status {s['status']}")


if __name__ == "__main__":
    main()
