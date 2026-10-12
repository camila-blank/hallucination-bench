# ruff: noqa: E501
"""Plan 0004 steps 1-6 (after the claim map): claim units, probe verdicts, consensus, the
``sibverdict-v1`` judge (layer-visible on every unit + layer-blind on a 300-unit subsample),
categories, rule labels + cut sweep, calibration samples and the summary.

    uv run python scripts/sibling_probe/verdicts.py units      # -> outputs/sibling/verdict/units.jsonl
    uv run python scripts/sibling_probe/verdicts.py judge      # Opus, cached (resumes)
    uv run python scripts/sibling_probe/verdicts.py label      # categories, summary.json, report.md
    uv run python scripts/sibling_probe/verdicts.py calib      # gold / dev / human todo files
    uv run python scripts/sibling_probe/verdicts.py figures    # plots/sibling_probe/fig{1,2}_*.png

Inputs: judge cells ``outputs/sibling/judge/<arm>/cells.jsonl`` (set A: s3d-rl600-all11,
jlens-all11, nla-rl-L42, mlnla-rl600-all11; set B: br-<arm>), readouts (set A
``outputs/sibling/readouts/<arm>/``, set B ``outputs/sibling/br/readouts/<arm>/``), probes (set
A ``outputs/sibling/probes_all11/{v2,v0}`` for 20..60 and ``outputs/sibling/probes/{v2,v0}``
for L42; set B ``outputs/sibling/br/probes/{v2,v0}``), the brnorm cache.
"""

import argparse
import asyncio
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from hallucination_bench import REPO_ROOT
from hallucination_bench.cache import Cache, fingerprint
from hallucination_bench.evals.sibling_probe import bank as SB
from hallucination_bench.evals.sibling_probe import br_bank as BB
from hallucination_bench.evals.sibling_probe import claims as CL
from hallucination_bench.evals.sibling_probe import verdict as V
from hallucination_bench.evals.sibling_probe.probe_lookup import ProbeTable, want_keys
from hallucination_bench.evals.wsbench_chat.run import HAL_K
from hallucination_bench.llm import LLM, Usage
from hallucination_bench.readouts import decode_token, load_readouts

sys.path.insert(0, str(Path(__file__).resolve().parent))
import br_build as BRB

SIB = REPO_ROOT / "outputs" / "sibling"
OUT = SIB / "verdict"
JUDGE = SIB / "judge"
MODEL, EFFORT = "claude-opus-5-5", "low"
FALLBACK = "claude-sonnet-5-5"  # refusals only; recorded in meta.model
BLIND_N = 300
SEED = 0
READOUT_DIR_A = {
    "s3d-rl600-all11": "s3d-rl600-all11",
    "jlens-all11": "jlens-all11",
    "nla-rl-L42": "nla-rl-L42",
    "mlnla-rl600-all11": "mlnla-rl600-all11",
}
ARM_OF = {("A", m): a for a, m in CL.ARM_METHOD_A.items()} | {
    ("B", m): a for a, m in CL.ARM_METHOD_B.items()
}


def read_jsonl(p: Path):
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def write_jsonl(p: Path, rows) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ units


def items_a() -> dict[str, dict]:
    return {it["id"]: it for it in SB.load_items()}


def items_b() -> dict[str, dict]:
    return {it["id"]: it for it in BB.load_bank() if it.get("in_pipeline")}


def build_units(arms_a=None, arms_b=None) -> list[dict]:
    units: list[CL.Unit] = []
    for arm, m in CL.ARM_METHOD_A.items():
        if arms_a is not None and arm not in arms_a:
            continue
        p = JUDGE / arm / "cells.jsonl"
        if not p.exists():
            print(f"[units] set A {arm}: no cells.jsonl (skipped)")
            continue
        us = CL.units_set_a(m, read_jsonl(p))
        print(f"[units] set A {arm}: {len(us)} units")
        units += us
    norm = BRB.load_brnorm()

    def nf(rec, seg):
        r = norm.get(BRB.norm_key(rec["id"], seg))
        if not r or r.get("kind") == "none":
            return None
        c = r["concept"] if r["kind"] == "member" else BB.snake(r["concept"])
        return r["kind"], c

    for arm, m in CL.ARM_METHOD_B.items():
        if arms_b is not None and arm not in arms_b:
            continue
        p = JUDGE / arm / "cells.jsonl"
        if not p.exists():
            print(f"[units] set B {arm}: no cells.jsonl (skipped)")
            continue
        us = CL.units_set_b(m, read_jsonl(p), nf)
        print(f"[units] set B {arm}: {len(us)} units")
        units += us
    flags = CL.consensus_flags(units)
    rows = []
    for u in units:
        d = u.to_dict()
        d.update(flags[u.uid])
        d["arm"] = ARM_OF[(u.set, u.method)]
        d["depth"] = V.depth(u.layer)
        rows.append(d)
    return rows


def probe_tables(rows: list[dict]) -> dict:
    ia = {
        k: {"category": v["category"], "gold": v["gold"], "named": v.get("named")}
        for k, v in items_a().items()
    }
    ib = {
        k: {"category": v["category"], "gold": v["gold_concept"], "named": []}
        for k, v in items_b().items()
    }
    wa = want_keys(r for r in rows if r["set"] == "A")
    wb = want_keys(r for r in rows if r["set"] == "B")
    t: dict = {}
    for rec in ("v2", "v0"):
        d11 = SIB / "probes_all11" / rec
        d42 = SIB / "probes" / rec
        dB = SIB / "br" / "probes" / rec
        t[("A11", rec)] = (
            ProbeTable(d11, ia, wa, rec) if (d11 / "validity.jsonl").exists() else None
        )
        t[("A42", rec)] = (
            ProbeTable(d42, ia, wa, rec) if (d42 / "validity.jsonl").exists() else None
        )
        t[("B", rec)] = ProbeTable(dB, ib, wb, rec) if (dB / "validity.jsonl").exists() else None
    return t


def attach_probes(rows: list[dict], tables: dict) -> None:
    for r in rows:
        for rec in ("v2", "v0"):
            if r["kind"] != "sibling":
                pv = {"recipe": rec, "status": "na", "na_reason": r["kind"], "detected": False}
            else:
                key = (
                    ("B", rec)
                    if r["set"] == "B"
                    else (("A42", rec) if r["layer"] == 42 else ("A11", rec))
                )
                tb = tables.get(key)
                if tb is None:
                    pv = {
                        "recipe": rec,
                        "status": "na",
                        "na_reason": "probes_not_run",
                        "detected": False,
                    }
                else:
                    pv = tb.verdict(
                        r["item"], r["category"], r["gold"], r["layer"], r["pos"], r["concept"]
                    )
            r["probe" if rec == "v2" else "probe_v0"] = pv


def stage_units() -> None:
    rows = build_units()
    tables = probe_tables(rows)
    attach_probes(rows, tables)
    write_jsonl(OUT / "units.jsonl", rows)
    gates = {}
    for (k, rec), tb in tables.items():
        if tb is not None:
            gt = tb.gate_table()
            gates[f"{k}:{rec}"] = {
                "n": len(gt),
                "pass": sum(g["pass"] for g in gt),
                "reasons": dict(Counter(g["reason"] for g in gt)),
            }
            write_jsonl(OUT / f"gates_{k}_{rec}.jsonl", gt)
    (OUT / "gates.json").write_text(json.dumps(gates, indent=1))
    c = Counter((r["set"], r["method"]) for r in rows)
    print(f"[units] {len(rows)} units: {dict(c)}")
    print(
        f"[units] consensus: {sum(r['consensus'] for r in rows)}, consensus_mlnla: {sum(r['consensus_mlnla'] for r in rows)}"
    )
    print(
        f"[units] probe ok: {sum(r['probe']['status'] == 'ok' for r in rows)}, detected: {sum(r['probe']['detected'] for r in rows)}"
    )
    print(json.dumps(gates, indent=1))


# ------------------------------------------------------------------ judge context


class Context:
    """Prompt / response / readout samples per unit (cached loads per arm)."""

    def __init__(self) -> None:
        self.ia = items_a()
        self.ib = items_b()
        self._cells: dict[str, dict[str, dict]] = {}
        self._samples: dict[str, dict[str, Any]] = {}

    def cells(self, arm: str, keys: set[str]) -> dict[str, dict]:
        if arm not in self._cells:
            self._cells[arm] = {
                r["key"]: {k: r[k] for k in ("key", "summary", "kind", "token_hits")}
                for r in read_jsonl(JUDGE / arm / "cells.jsonl")
                if r["key"] in keys
            }
        return self._cells[arm]

    def samples(self, arm: str, set_: str, keys: set[str]) -> dict[str, Any]:
        if arm not in self._samples:
            if set_ == "A":
                files = sorted((SIB / "readouts" / READOUT_DIR_A[arm]).glob("*.jsonl"))
            else:
                files = [SIB / "br" / "readouts" / arm.removeprefix("br-") / "basic_readout.jsonl"]
            out = {}
            for f in files:
                cs, _ = load_readouts(f)
                for c in cs:
                    if c.key in keys:
                        out[c.key] = c
            self._samples[arm] = out
        return self._samples[arm]

    def render(self, u: dict, blind: bool, keys_by_arm) -> tuple[str, str] | None:
        arm, key = u["arm"], u["cell_key"]
        cell = self.samples(arm, u["set"], keys_by_arm[arm]).get(key)
        rec = self.cells(arm, keys_by_arm[arm]).get(key)
        if cell is None or rec is None:
            return None
        tokens_line = ""
        if cell.kind == "tokens":
            samples = [rec.get("summary") or ""]
            toks = [decode_token(t) for t in cell.tokens or ()][:10]
            tokens_line = (
                "J-lens raw top-10 tokens: "
                + " | ".join(repr(t) for t in toks)
                + "\n(the readout below is a summary of these tokens)"
            )
        else:
            samples = [s for s in cell.samples or () if s.strip()][:HAL_K]
        if u["set"] == "A":
            it = self.ia[u["item"]]
            prompt, response, note = (
                SB.user_text(it["messages"]),
                it.get("response") or "",
                ": the model's internal state while reading the user's message",
            )
        else:
            it = self.ib[u["item"]]
            prompt = it["prompt"]
            response = BB.response_text(it)
            note = (
                (
                    ': a chat question; the reply was prefilled with "'
                    + BB.ANSWER_STEM
                    + '" and the read site is its last token'
                )
                if it.get("eval_render") == "chat_prefill"
                else ": a plain-text sentence to complete; the read site is its last token"
            )
        concept = self.concept_label(u)
        return V.render(
            prompt=prompt,
            response=response,
            concept=concept,
            samples=samples,
            segments=u["segments"],
            layer=None if blind else u["layer"],
            probe=u["probe"],
            resp_note=note,
            tokens_line=tokens_line,
        )

    _labels: dict | None = None

    def concept_label(self, u: dict) -> str:
        if self._labels is None:
            lab = {}
            for cat in SB_inv().values():
                for c in cat.concepts:
                    lab[("A", c.id)] = c.label
            for cname, d in BB.load_br_concepts()["categories"].items():
                for c in d["concepts"]:
                    lab[("B", cname, c["id"])] = c["label"]
            self._labels = lab
        if u["set"] == "A":
            return self._labels.get(("A", u["concept"]), u["concept"])
        lab = self._labels.get(("B", u["category"], u["concept"]), u["concept"].replace("_", " "))
        if u["kind"] == "entity_substitution":
            return f"{lab} (in place of the entity named in the prompt)"
        return lab


def SB_inv():
    from hallucination_bench.evals.sibling_probe.concepts import load_inventory

    return load_inventory()


def blind_sample(rows: list[dict], n: int = BLIND_N) -> set[str]:
    """Seeded, stratified (method x depth x set) sample of the judge-category units (not
    validated, not consensus, not entity substitution, probe available)."""
    pool = [r for r in rows if judge_pool(r)]
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for r in pool:
        strata[(r["method"], r["depth"], r["set"])].append(r)
    tot = len(pool)
    out: set[str] = set()
    if not tot:
        return out
    for _k, rs in sorted(strata.items()):
        m = max(1, round(n * len(rs) / tot))
        rs = sorted(
            rs, key=lambda r: hashlib.sha256(f"blind:{SEED}:{r['uid']}".encode()).hexdigest()
        )
        out.update(r["uid"] for r in rs[:m])
    return out


def judge_pool(r: dict) -> bool:
    cons = r["consensus_mlnla"] if r["method"] == CL.MLNLA else r["consensus"]
    return (
        r["kind"] == "sibling"
        and r["probe"]["status"] == "ok"
        and not r["probe"]["detected"]
        and not cons
    )


async def stage_judge(
    concurrency: int, limit: int | None, blind_only: bool = False, only_set: str | None = None
) -> None:
    rows = list(read_jsonl(OUT / "units.jsonl"))
    blind = blind_sample(rows)
    (OUT / "blind_uids.json").write_text(json.dumps(sorted(blind)))
    if only_set:
        rows = [r for r in rows if r["set"] == only_set]
    if limit:
        rows = rows[:limit]
    keys_by_arm: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        keys_by_arm[r["arm"]].add(r["cell_key"])
    ctx = Context()
    cache = Cache(OUT / "cache.jsonl")
    llm = LLM(rpm=1000, concurrency=concurrency, effort=EFFORT)
    usage = Usage()
    jobs = []
    for r in rows:
        modes = (["b"] if r["uid"] in blind else []) + ([] if blind_only else ["v"])
        for mode in modes:
            jobs.append((r, mode))
    done = 0
    sem = asyncio.Semaphore(concurrency * 2)

    async def one(r, mode):
        nonlocal done
        async with sem:
            pr = ctx.render(r, mode == "b", keys_by_arm)
            if pr is None:
                return
            system, user = pr
            key = f"{mode}:{r['uid']}"
            fp = fingerprint(V.PROMPT_VERSION, MODEL, EFFORT, system, user)
            if cache.get(key, fp) is not None:
                done += 1
                return
            n = len(r["segments"])
            res, used = None, MODEL
            # Opus 5.5 refuses some hacker-occupation units (as in gen_items): Sonnet fallback
            for model in (MODEL, MODEL, FALLBACK, FALLBACK):
                out = await llm.call(model=model, system=system, user=user, schema=V.schema())
                usage.add(out.usage)
                if V.validate(out.result, n):
                    res, used = out.result, model
                    break
            meta = {"usage": out.usage.to_dict(), "model": used}
            cache.put(key, fp, {"result": res, "meta": meta})
            done += 1
            if done % 500 == 0:
                print(f"  [verdict {done}/{len(jobs)}] {usage.report()}", flush=True)

    try:
        await llm.preflight(MODEL)
        await asyncio.gather(*(one(r, m) for r, m in jobs))
    finally:
        await llm.aclose()
        cache.close()
    print(f"[judge] {len(jobs)} jobs ({len(blind)} blind); this run: {usage.report()}")


# ------------------------------------------------------------------ label


def load_verdicts() -> dict[str, dict]:
    out = {}
    p = OUT / "cache.jsonl"
    if p.exists():
        for r in read_jsonl(p):
            if r.get("result") is not None or r["key"] not in out:
                out[r["key"]] = r
    return out


def cache_usd() -> float:
    p = OUT / "cache.jsonl"
    return (
        sum(((r.get("meta") or {}).get("usage") or {}).get("usd", 0) for r in read_jsonl(p))
        if p.exists()
        else 0.0
    )


def label_rows(rows: list[dict], verdicts: dict, samples_of, blind: set[str] | None = None) -> None:
    """``blind``: the fixed layer-blind sample (``blind_uids.json``); blind answers cached for
    other units (an earlier, smaller unit set drew a different sample) are ignored."""
    for r in rows:
        for mode, fld in (("v", "judge"), ("b", "judge_blind")):
            v = verdicts.get(f"{mode}:{r['uid']}")
            if mode == "b" and blind is not None and r["uid"] not in blind:
                v = None
            if v is None:
                r[fld] = None
                continue
            r[fld] = V.parse(v.get("result"), len(r["segments"]), samples_of(r))
        j = r["judge"]
        r["form_label"] = j["form_label"] if j else None
        cons = r["consensus_mlnla"] if r["method"] == CL.MLNLA else r["consensus"]
        r["category"] = V.category(
            kind=r["kind"],
            probe_status=r["probe"]["status"],
            detected=r["probe"]["detected"],
            consensus=cons,
            form_label=r["form_label"],
        )
        r["category_v0"] = V.category(
            kind=r["kind"],
            probe_status=r["probe_v0"]["status"],
            detected=r["probe_v0"]["detected"],
            consensus=cons,
            form_label=r["form_label"],
        )
        r["rule"] = V.rule_label(r["layer"])
        r["rule_cuts"] = {str(c): V.rule_label(r["layer"], c) for c in V.CUTS}
        r["hallucination"] = V.is_hallucination(r["category"], r["form_label"])


def stage_label() -> None:
    rows = list(read_jsonl(OUT / "units.jsonl"))
    verdicts = load_verdicts()
    keys_by_arm: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        keys_by_arm[r["arm"]].add(r["cell_key"])
    ctx = Context()

    def samples_of(r):
        cell = ctx.samples(r["arm"], r["set"], keys_by_arm[r["arm"]]).get(r["cell_key"])
        if cell is None:
            return []
        if cell.kind == "tokens":
            rec = ctx.cells(r["arm"], keys_by_arm[r["arm"]]).get(r["cell_key"]) or {}
            return [rec.get("summary") or ""]
        return [s for s in cell.samples or () if s.strip()][:HAL_K]

    bp = OUT / "blind_uids.json"
    blind = set(json.loads(bp.read_text())) if bp.exists() else None
    label_rows(rows, verdicts, samples_of, blind)
    write_jsonl(OUT / "units_labeled.jsonl", rows)
    import importlib

    R = importlib.import_module("verdict_report")
    summ = R.summarise(rows, api_usd=cache_usd())
    gp = REPO_ROOT / "data" / "sibling_probe" / "verdict_gold.jsonl"
    summ["gold"] = R.gold_accuracy(rows, gp) if gp.exists() else None
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    (OUT / "report.md").write_text(R.markdown(summ))
    print(f"[label] {len(rows)} units -> {OUT / 'units_labeled.jsonl'}, summary.json, report.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["units", "judge", "label", "calib", "figures"])
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--blind-only", action="store_true")
    ap.add_argument("--set", default=None, choices=["A", "B"], help="judge one set only")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "units":
        stage_units()
    elif a.stage == "judge":
        asyncio.run(stage_judge(a.concurrency, a.limit, a.blind_only, a.set))
    elif a.stage == "label":
        stage_label()
    elif a.stage == "calib":
        import verdict_report as R

        R.write_calibration(list(read_jsonl(OUT / "units.jsonl")))
    elif a.stage == "figures":
        from hallucination_bench.evals.sibling_probe import figures as F

        F.make_all(
            list(read_jsonl(OUT / "units_labeled.jsonl")), REPO_ROOT / "plots" / "sibling_probe"
        )


if __name__ == "__main__":
    _ = math
    main()
