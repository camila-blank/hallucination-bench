# ruff: noqa: E501
"""Plan 0009 full-run numbers -> ``outputs/sibling_gate/full_summary.md`` (task 5 of the driver).

    uv run python scripts/sibling_gate/full_summary.py [--out outputs/sibling_gate]

Reads (all local, no API calls): ``labels/<arm>.jsonl`` + ``fill/rung1`` (units, rung 1 / final),
``report.json`` / ``report_per_key.json`` (``hbench sibling-gate report``), ``rev20_summaries/``
(rev-20 rebuilds, empty labels dir) vs ``wsbench_chat/<judge>/<arm>/summary_v2.json`` (rev 21),
the three LLM caches (spend by stage), the probe scores (z_S vs the best distractor in the
examples), ``stage_wall.json`` (wall-clock per stage, written by the drivers) and the capture /
fit / score logs (Modal estimate).
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JUDGE = "claude-opus-5-5"
HALL = ("unstable", "probe_unconfirmed")
H200_USD_PER_H = 4.54  # Modal list price, 2026
CPU_USD_PER_CORE_H = 0.0000131 * 3600  # Modal CPU list price; non-preemptible functions bill 3x


def jl(p: Path):
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def get(d, *path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def pct(v, ci=None, digits=1):
    if v is None:
        return "—"
    s = f"{100 * v:.{digits}f}%"
    if ci:
        s += f" [{100 * ci[0]:.{digits}f}, {100 * ci[1]:.{digits}f}]"
    return s


def spend_by_stage(out: Path):
    rows = defaultdict(lambda: Counter())
    for stage, rel in (
        ("fill (rung-1 judge)", "fill"),
        ("classes/dedupe/concepts", "classes"),
        ("probe texts", "texts"),
    ):
        for r in jl(out / rel / "cache.jsonl"):
            u = ((r.get("meta") or {}).get("usage")) or {}
            c = rows[stage]
            c["usd"] += u.get("usd", 0.0)
            c["calls"] += u.get("calls", 0)
            c["in"] += u.get("input_tokens", 0)
            c["out"] += u.get("output_tokens", 0)
    return rows


def block_line(a):
    return ", ".join(
        f"{x['block']}:{x['answer']}" + (f"({'/'.join(x['objects'])[:40]})" if x["objects"] else "")
        for x in a
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs" / "sibling_gate"))
    ap.add_argument("--cells", default=str(ROOT / "outputs" / "wsbench_chat" / JUDGE))
    ap.add_argument("--n-examples", type=int, default=10)
    a = ap.parse_args()
    out, cells = Path(a.out), Path(a.cells)
    labels = {p.stem: jl(p) for p in sorted((out / "labels").glob("*.jsonl"))}
    rung1 = {}
    for p in (out / "fill" / "rung1").glob("*.jsonl"):
        for r in jl(p):
            rung1[r["unit"]] = r
    rep = json.loads((out / "report.json").read_text()) if (out / "report.json").exists() else {}
    perkey = (
        json.loads((out / "report_per_key.json").read_text())
        if (out / "report_per_key.json").exists()
        else {}
    )
    md = ["# Plan 0009 full run: summary numbers", ""]

    # 1. units by rung1 / final, branch split
    md += ["## 1. Units (segments) by rung-1 and final status", ""]
    r1s = ("unstable", "stable", "sparse", "deterministic", "resample_failed")
    fs = (
        "unstable", "probe_faithful", "probe_nearby", "cross_method", "probe_unconfirmed",
        "probe_na", "resample_failed", "pending",
    )  # fmt: skip
    md += ["| arm | units | segments | " + " | ".join(r1s) + " |", "|" + "---|" * (len(r1s) + 3)]
    for arm, s in rep.items():
        md.append(
            f"| {arm} | {s['n_units']} | {s['n_segments']} | "
            + " | ".join(
                f"{s['rung1'].get(k, [0, 0])[0]} ({s['rung1'].get(k, [0, 0])[1]})" for k in r1s
            )
            + " |"
        )
    md += ["", "Final status (units (segments)); gated = unstable + probe_unconfirmed:", ""]
    md += ["| arm | " + " | ".join(fs) + " | gated |", "|" + "---|" * (len(fs) + 2)]
    for arm, s in rep.items():
        md.append(
            f"| {arm} | "
            + " | ".join(
                f"{s['final'].get(k, [0, 0])[0]} ({s['final'].get(k, [0, 0])[1]})" for k in fs
            )
            + f" | {s['gated'][0]} ({s['gated'][1]}) |"
        )
    md += ["", "### Unstable branch split (A: n_other >= 2, B: n_actual >= 4, both)", ""]
    md += [
        "| arm | A only | B only | both | no_gold | no_class | diverse |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm, s in rep.items():
        b = s["branch"]
        md.append(
            f"| {arm} | {b['A']} | {b['B']} | {b['both']} | {s['no_gold']} | {s['no_class']} | {s['diverse']} |"
        )

    # 2. rung-2 status mix with na_reason
    md += [
        "",
        "## 2. Rung-2 status mix and na_reason (rung-2 units = stable / sparse / deterministic)",
        "",
    ]
    rs = (
        "probe_faithful",
        "probe_nearby",
        "cross_method",
        "probe_unconfirmed",
        "probe_na",
        "pending",
    )
    reasons = sorted(
        {
            (r.get("rung2") or {}).get("na_reason")
            for rows in labels.values()
            for r in rows
            if (r.get("rung2") or {}).get("na_reason")
        }
    )
    md += [
        "| arm | rung-2 units | "
        + " | ".join(rs)
        + " | "
        + " | ".join(f"na:{x}" for x in reasons)
        + " |"
    ]
    md += ["|" + "---|" * (len(rs) + len(reasons) + 2)]
    for arm, rows in labels.items():
        r2 = [r for r in rows if r.get("rung1") in ("stable", "sparse", "deterministic")]
        st = Counter((r.get("final") or "pending") for r in r2)
        nr = Counter(
            (r.get("rung2") or {}).get("na_reason")
            for r in r2
            if (r.get("rung2") or {}).get("na_reason")
        )
        md.append(
            f"| {arm} | {len(r2)} | "
            + " | ".join(str(st.get(k, 0)) for k in rs)
            + " | "
            + " | ".join(str(nr.get(x, 0)) for x in reasons)
            + " |"
        )

    # 3. per key
    md += ["", "## 3. Per-key table (all arms pooled)", ""]
    md += [
        "| key | units | rung-2 units | share of rung 2 | probe_na rate | probe_faithful rate | rung-1 mix |",
        "|---|---|---|---|---|---|---|",
    ]
    for k, d in perkey.items():
        md.append(
            f"| {k} | {d['n_units']} | {d['n_rung2']} | {pct(d['rung2_share'])} | {pct(d['probe_na_rate'])} | "
            f"{pct(d['probe_faithful_rate'])} | {d['rung1']} |"
        )

    # 4. hallucination share rev 20 vs 21
    md += ["", "## 4. Hallucination share, rev 20 vs rev 21 (item-bootstrap 95% CI)", ""]
    md += [
        "| arm | char-weighted rev 20 | char-weighted rev 21 | readout-level rev 20 | readout-level rev 21 |",
        "|---|---|---|---|---|",
    ]
    for arm in labels:
        old = (
            json.loads((out.parent / "rev20_summaries" / f"{arm}.json").read_text())
            if (out.parent / "rev20_summaries" / f"{arm}.json").exists()
            else None
        )
        newp = cells / arm / "summary_v2.json"
        new = json.loads(newp.read_text()) if newp.exists() else None
        row = [arm]
        for d in (old, new):
            c = get(d, "overall", "char_share_verdict", "hallucination") or {}
            row.append(pct(c.get("value"), c.get("ci95")))
        for d in (old, new):
            c = get(d, "overall", "rates", "hallucination", "all") or {}
            row.append(pct(c.get("value"), c.get("ci95")))
        md.append("| " + " | ".join(row) + " |")

    # 5. sensitivity rows
    md += ["", "## 5. Sensitivity rows (gated units (segments) under each variant)", ""]
    sens = (
        "rung1_rev1",
        "rung1_branch_a_only",
        "rung1_said_le1",
        "rung1_said_le3",
        "cross_same_pos",
        "cross_same_family",
        "cross_shared_prior",
    )
    md += ["| arm | main | " + " | ".join(sens) + " |", "|" + "---|" * (len(sens) + 2)]
    for arm, s in rep.items():
        md.append(
            f"| {arm} | {s['gated'][0]} ({s['gated'][1]}) | "
            + " | ".join(f"{s['sensitivity'][k][0]} ({s['sensitivity'][k][1]})" for k in sens)
            + " |"
        )

    # 6. matched null
    mn = out / "labels" / "matched_null.json"
    md += [
        "",
        "## 6. Matched null (S vs the same distractors, at the item's other sites and other items with the same gold, where no arm names S)",
        "",
    ]
    if mn.exists():
        d = json.loads(mn.read_text())
        md.append(
            f"n_sites {d.get('n_sites')}, n_fire {d.get('n_fire')}, fire rate "
            f"{pct(d['n_fire'] / d['n_sites']) if d.get('n_sites') else '—'}"
        )

    # 7. examples
    md += ["", f"## 7. Examples ({a.n_examples} each, seeded random across arms)", ""]
    data = Path(out.parent.parent / "data" / "sibling_gate")
    cps = classes = dis = None
    try:
        from hallucination_bench.evals.sibling_gate import ladder as L

        classes = json.loads((data / "classes.json").read_text())["classes"]
        concepts = json.loads((data / "concepts.json").read_text())["classes"]
        dis = json.loads((data / "distractors.json").read_text())
        cps = L.load_class_probes(out / "probes", concepts)
    except Exception as e:
        md.append(f"(probe scores unavailable: {e})")
    allrows = [r for rows in labels.values() for r in rows]
    rng = random.Random(0)

    def pick(pred):
        xs = [r for r in allrows if pred(r)]
        rng.shuffle(xs)
        return xs[: a.n_examples], len(xs)

    groups = {
        "unstable": lambda r: r.get("final") == "unstable",
        "stable-faithful (rung 1 stable, probe_faithful)": lambda r: (
            r.get("rung1") == "stable" and r.get("final") == "probe_faithful"
        ),
        "stable-unconfirmed (rung 1 stable, probe_unconfirmed)": lambda r: (
            r.get("rung1") == "stable" and r.get("final") == "probe_unconfirmed"
        ),
    }
    for title, pred in groups.items():
        ex, n = pick(pred)
        md += [f"### {title} (n = {n})", ""]
        for r in ex:
            r1 = rung1.get(r["unit"]) or {}
            line = f"- `{r['unit']}` said={r['said']!r} actual={r['actual']!r}; counts {r.get('counts')}"
            md.append(line)
            if title == "unstable":
                md.append(f"  - blocks: {block_line(r1.get('answers') or [])}")
            else:
                md.append(f"  - rung 1 blocks: {block_line(r1.get('answers') or [])}")
                if cps and classes and dis:
                    sn = "".join(ch for ch in "")  # noqa: F841 (normalised key is in the unit key)
                    # the unit key is arm:id__Lxxx__ppos:said_norm:actual_norm
                    said_norm = r["unit"].split(":")[-2]
                    v = classes.get(f"{said_norm}|{r['id']}")
                    if v and v["class"] in cps and cps[v["class"]].z is not None:
                        cp = cps[v["class"]]
                        zs = cp.z.get(r["id"], r["layer"], r["pos"], v["said_canonical"])
                        ds = (dis.get(r["unit"]) or {}).get("distractors", [])
                        # the fire rule compares against VALID distractors only (ladder.fires)
                        ds = [d for d in ds if cp.valid(d, r["layer"]) and d != v["said_canonical"]]
                        zd = {d: cp.z.get(r["id"], r["layer"], r["pos"], d) for d in ds}
                        top = max(zd.items(), key=lambda kv: kv[1], default=(None, float("nan")))
                        md.append(
                            f"  - class {v['class']!r}, S={v['said_canonical']!r}: z_S={zs:.2f} vs best valid distractor "
                            f"{top[0]!r} {top[1]:.2f}; na/trace {r.get('rung2')}"
                        )
        md.append("")

    # 8. spend
    md += ["## 8. Spend", "", "API (cache meta usage; claude-opus-5-5):", ""]
    md += ["| stage | calls | in tokens | out tokens | USD |", "|---|---|---|---|---|"]
    tot = 0.0
    for stage, c in spend_by_stage(out).items():
        md.append(f"| {stage} | {c['calls']} | {c['in']} | {c['out']} | ${c['usd']:.2f} |")
        tot += c["usd"]
    md += [f"| total | | | | ${tot:.2f} |", ""]
    sw = out / "stage_wall.json"
    walls = json.loads(sw.read_text()) if sw.exists() else {}
    md += ["## 9. Wall-clock per stage (seconds)", ""]
    md += ["| stage | wall s |", "|---|---|"] + [f"| {k} | {v} |" for k, v in walls.items()]
    md += ["", "## 10. Modal estimate", ""]
    me = out / "modal_estimate.json"
    if me.exists():
        d = json.loads(me.read_text())
        md += [f"- {k}: {v}" for k, v in d.items()]
    (out / "full_summary.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out / 'full_summary.md'}")


if __name__ == "__main__":
    main()
