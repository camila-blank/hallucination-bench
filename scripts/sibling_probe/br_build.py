# ruff: noqa: E501
"""Plan 0004 set B (WorkspaceBench ``basic_readout``): bank, readouts, inventory, probe texts.

Stages (idempotent; Opus calls cached per prompt version under ``outputs/sibling/br/``):

  bank-in     render every item (wsbench eval render), check against the
              ``wsbench-acts:pubplan-q36-all11`` manifest tokens, merge + dedup each arm's
              readouts into ``outputs/sibling/br/readouts/<arm>/basic_readout.jsonl`` (pos
              alignment check, exit 2 on any problem), write the raw-mode Qwen input
              ``outputs/sibling/br/gen_in.jsonl``
  bank        (after ``qwen_gen_modal.py`` -> ``gen_out.jsonl``) brcat-v1 per item ->
              ``data/sibling_probe/basic_readout_bank.json``
  inventory   brinv-v1 per category (15 members + golds, BEFORE any claim is mapped) ->
              ``data/sibling_probe/br_concepts.json``
  probe-texts brtext-v1: per (category, concept, template) explicit + implied texts and
              neutral texts per (category, template) -> ``data/sibling_probe/br_probe_texts.jsonl``
  capture-rows capture rows for the set-B eval prompts (+ layer 42) and the probe texts

    uv run python scripts/sibling_probe/br_build.py bank-in
    MODAL_PROFILE=safety-sahan-2 uvx --python 3.12 modal@latest run scripts/sibling_probe/qwen_gen_modal.py \\
        --inp outputs/sibling/br/gen_in.jsonl --out outputs/sibling/br/gen_out.jsonl
    uv run python scripts/sibling_probe/br_build.py bank
    uv run python scripts/sibling_probe/br_build.py inventory
    uv run python scripts/sibling_probe/br_build.py probe-texts
    uv run python scripts/sibling_probe/br_build.py capture-rows
"""

import argparse
import asyncio
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from hallucination_bench import REPO_ROOT, paths
from hallucination_bench.cache import Cache, fingerprint
from hallucination_bench.evals.sibling_probe import br_bank as BB
from hallucination_bench.llm import LLM, Usage

MODEL = "claude-opus-5-5"
EFFORT = "low"
BR = BB.BR
PAPER = paths.READOUTS_ROOT
COMBOS = paths.COMBOS_READOUTS
LAYERS11 = tuple(range(20, 61, 4))
ARM_SOURCES: dict[str, list[Path]] = {
    "s3d-rl600": [PAPER / "s3d-rl600" / "basic_readout.jsonl"],
    "jlens": [PAPER / "jlens" / "basic_readout.jsonl"],
    "nla-rl-L42": [PAPER / "nla-rl-L42" / "basic_readout.jsonl"],
    "mlnla-rl600": [COMBOS / f"mlnla600-L{L}" / "basic_readout.jsonl" for L in LAYERS11],
}
PROBES_PER_KIND = 24  # per (concept): explicit / implied, split over the category's templates
NEUTRAL_PER_CAT = 24


class Opus:
    """One cached structured-call helper per prompt version (own cache file)."""

    def __init__(self, version: str, concurrency: int = 24) -> None:
        BR.mkdir(parents=True, exist_ok=True)
        self.version = version
        self.cache = Cache(BR / f"cache_{version}.jsonl")
        self.llm: LLM | None = None
        self.usage = Usage()
        self.n_calls = 0
        self.concurrency = concurrency

    async def call(self, key, system, user, schema, validate=None, retries=2):
        fp = fingerprint(self.version, MODEL, EFFORT, system, user)
        row = self.cache.get(key, fp)
        if row is not None:
            return row["result"]
        if self.llm is None:
            self.llm = LLM(rpm=600, concurrency=self.concurrency, effort=EFFORT)
        for _ in range(retries + 1):
            res = await self.llm.call(model=MODEL, system=system, user=user, schema=schema)
            self.usage.add(res.usage)
            self.n_calls += 1
            if res.result is not None and (validate is None or validate(res.result)):
                self.cache.put(
                    key, fp, {"result": res.result, "meta": {"usage": res.usage.to_dict()}}
                )
                return res.result
        self.cache.put(key, fp, {"result": None, "meta": {"raw": res.result}})
        return None

    async def close(self) -> None:
        if self.llm is not None:
            await self.llm.aclose()
        self.cache.close()


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def write_jsonl(p: Path, rows) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")


# ------------------------------------------------------------------ bank-in


def stage_bank_in() -> None:
    tok = tokenizer()
    items = BB.load_wsb_items()
    man = json.loads((BR / "pubplan_manifest.json").read_text())
    mrows = {p["label"]: p for p in man["prompts"]}
    problems: list[str] = []
    rendered = []
    for it in items:
        lab = BB.label_of(it["name"])
        ids = BB.render_ids(tok, it)
        dec = [tok.decode([i]) for i in ids]
        m = mrows.get(lab)
        if m is None:
            problems.append(f"{lab}: not in pubplan manifest")
        else:
            if m["n_pos"] != len(ids):
                problems.append(f"{lab}: n_pos {m['n_pos']} != {len(ids)}")
            if m["eval_positions"] != [len(ids) - 1]:
                problems.append(f"{lab}: eval_positions {m['eval_positions']}")
        rendered.append(
            {"id": lab, "name": it["name"], "input_ids": ids, "tokens": dec, "pos": len(ids) - 1}
        )
    by_id = {r["id"]: r for r in rendered}
    # readouts: merge + dedup by (id, layer, pos); alignment check
    rows_by_arm = {}
    for arm, files in ARM_SOURCES.items():
        seen, rows, dups = set(), [], 0
        for f in files:
            for r in read_jsonl(f):
                k = (r["id"], int(r["layer"]), int(r["pos"]))
                if k in seen:
                    dups += 1
                    continue
                seen.add(k)
                rows.append(r)
        rows_by_arm[arm] = rows
        write_jsonl(BR / "readouts" / arm / "basic_readout.jsonl", rows)
        cnt = Counter(r["layer"] for r in rows)
        print(
            f"[bank-in] {arm}: {len(rows)} rows ({dups} duplicates dropped) layers={dict(sorted(cnt.items()))}"
        )
    # the bank rows' last decoded token: wsbench readouts store tokenizer.decode([id])
    problems += BB.alignment_problems(
        [{"id": r["id"], "pos": r["pos"], "tokens": r["tokens"]} for r in rendered], rows_by_arm
    )
    for arm, rows in rows_by_arm.items():
        missing = {r["id"] for r in rendered} - {r["id"] for r in rows}
        if missing:
            problems.append(f"{arm}: {len(missing)} items without readouts")
    (BR / "rendered.json").write_text(json.dumps(rendered))
    gen = [
        {"key": r["id"], "input_ids": r["input_ids"], "max_new": BB.RESPONSE_MAX_NEW}
        for r in rendered
    ]
    write_jsonl(BR / "gen_in.jsonl", gen)
    print(f"[bank-in] {len(rendered)} items rendered -> {BR / 'gen_in.jsonl'}")
    if problems:
        print(f"[bank-in] {len(problems)} PROBLEMS:\n  " + "\n  ".join(problems[:40]))
        sys.exit(2)
    print("[bank-in] alignment OK (every arm, every row: pos == last prompt token, token matches)")
    assert by_id


# ------------------------------------------------------------------ bank


async def stage_bank() -> None:
    items = BB.load_wsb_items()
    rendered = {r["id"]: r for r in json.loads((BR / "rendered.json").read_text())}
    gen = {r["key"]: r for r in read_jsonl(BR / "gen_out.jsonl")}
    op = Opus(BB.BRCAT_VERSION)
    try:
        outs = await asyncio.gather(
            *(
                op.call(
                    f"brcat:{BB.label_of(it['name'])}",
                    *BB.render_brcat(it["prompt"], it["target"]),
                    BB.BRCAT_SCHEMA,
                    BB.valid_brcat,
                )
                for it in items
            )
        )
    finally:
        await op.close()
    rows = []
    for it, c in zip(items, outs, strict=True):
        lab = BB.label_of(it["name"])
        r = rendered[lab]
        g = gen.get(lab)
        if c is None:
            raise SystemExit(f"brcat failed for {lab}")
        row = {
            "id": lab,
            "name": it["name"],
            "subfamily": it["subfamily"],
            "prompt": it["prompt"],
            "target": it["target"],
            "match": it.get("intermediates") or [it["target"]],
            "eval_render": it.get("eval_render", "plain"),
            "prefill": BB.ANSWER_STEM if it.get("eval_render") == "chat_prefill" else None,
            "input_ids": r["input_ids"],
            "tokens": r["tokens"],
            "pos": r["pos"],
            "response": g["text"] if g else None,
            "response_n_tokens": g["n_tokens"] if g else None,
            "category": BB.snake(c["category"]),
            "gold": BB.snake(c["gold"]),
            "gold_label": c.get("gold_label") or it["target"],
            "categorical": bool(c["categorical"]),
        }
        row["in_pipeline"] = row["subfamily"] == "entity" or (
            row["subfamily"] == "implicit" and row["categorical"]
        )
        tpl, ent = BB.template_of(row)
        row["template"], row["entity"] = tpl, ent
        rows.append(row)
    # template index within category (probe-text endings)
    tpls: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        if r["in_pipeline"] and r["template"] not in tpls[r["category"]]:
            tpls[r["category"]].append(r["template"])
    for r in rows:
        r["template_idx"] = tpls[r["category"]].index(r["template"]) if r["in_pipeline"] else None
    counts = {
        "items": len(rows),
        "by_subfamily": dict(Counter(r["subfamily"] for r in rows)),
        "in_pipeline": sum(r["in_pipeline"] for r in rows),
        "categories": dict(Counter(r["category"] for r in rows if r["in_pipeline"])),
        "with_response": sum(r["response"] is not None for r in rows),
    }
    doc = {
        "version": "br-bank-v1",
        "provenance": {
            "items": str(BB.WSB_BR_ITEMS),
            "response": f"Qwen/Qwen3.6-27B greedy raw continuation of the eval render, {BB.RESPONSE_MAX_NEW} new tokens (scripts/sibling_probe/qwen_gen_modal.py raw mode)",
            "category": f"{BB.BRCAT_VERSION} ({MODEL}, effort {EFFORT})",
        },
        "counts": counts,
        "items": rows,
    }
    BB.BANK_PATH.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(counts, indent=1))
    print(f"[bank] brcat calls {op.n_calls}: {op.usage.report()} -> {BB.BANK_PATH}")


# ------------------------------------------------------------------ inventory


async def stage_inventory() -> None:
    bank = [r for r in BB.load_bank() if r["in_pipeline"]]
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in bank:
        by_cat[r["category"]].append(r)
    op = Opus(BB.BRINV_VERSION)
    cats = sorted(by_cat)
    try:
        outs = await asyncio.gather(
            *(
                op.call(
                    f"brinv:{c}",
                    *BB.render_brinv(
                        c,
                        sorted({r["gold_label"] for r in by_cat[c]}),
                        [r["prompt"] for r in by_cat[c]],
                    ),
                    BB.BRINV_SCHEMA,
                    lambda x: isinstance(x.get("members"), list) and len(x["members"]) >= 5,
                )
                for c in cats
            )
        )
    finally:
        await op.close()
    doc: dict[str, Any] = {"version": BB.BRINV_VERSION, "categories": {}}
    for c, out in zip(cats, outs, strict=True):
        if out is None:
            raise SystemExit(f"brinv failed for {c}")
        golds = {r["gold"]: r for r in by_cat[c]}
        members: dict[str, dict] = {}
        for m in out["members"]:
            mid = BB.snake(m["id"])
            if not mid or mid in members:
                continue
            members[mid] = {
                "id": mid,
                "label": m["label"],
                "aliases": sorted({a.strip() for a in [m["label"], *m["aliases"]] if a.strip()}),
                "gold": False,
                "source": "opus",
            }
        # golds: match by id, else by alias of the gold label; add if missing
        for g, r in golds.items():
            hit = g if g in members else None
            if hit is None:
                for mid, m in members.items():
                    if BB.named_in(r["gold_label"], m["aliases"]) or BB.named_in(
                        r["target"], m["aliases"]
                    ):
                        hit = mid
                        break
            if hit is None:
                members[g] = {
                    "id": g,
                    "label": r["gold_label"],
                    "aliases": sorted({r["gold_label"], r["target"], *r["match"]}),
                    "gold": True,
                    "source": "gold",
                }
                hit = g
            members[hit]["gold"] = True
            for rr in by_cat[c]:
                if rr["gold"] == g:
                    rr["gold_concept"] = hit
        doc["categories"][c] = {
            "templates": sorted(
                {r["template"] for r in by_cat[c]},
                key=lambda t: by_cat[c][[x["template"] for x in by_cat[c]].index(t)][
                    "template_idx"
                ],
            ),
            "items": [r["id"] for r in by_cat[c]],
            "gold_of": {r["id"]: r["gold_concept"] for r in by_cat[c]},
            "concepts": list(members.values()),
        }
    tok = tokenizer()
    for d in doc["categories"].values():
        for cc in d["concepts"]:
            cc["token_aliases"] = BB.token_aliases(tok, cc["aliases"])
    BB.CONCEPTS_PATH.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    # gold concept ids back into the bank
    full = json.loads(BB.BANK_PATH.read_text())
    gold_of = {i: g for d in doc["categories"].values() for i, g in d["gold_of"].items()}
    for r in full["items"]:
        r["gold_concept"] = gold_of.get(r["id"])
    BB.BANK_PATH.write_text(json.dumps(full, ensure_ascii=False, indent=1) + "\n")
    for c in cats:
        d = doc["categories"][c]
        print(
            f"  {c:28s} {len(d['concepts']):2d} concepts, golds={sorted({v for v in d['gold_of'].values()})}"
        )
    print(f"[inventory] brinv calls {op.n_calls}: {op.usage.report()} -> {BB.CONCEPTS_PATH}")


# ------------------------------------------------------------------ judge (claim map, every cell)

BR_ARMS = ("s3d-rl600", "jlens", "nla-rl-L42", "mlnla-rl600")
JUDGE_ROOT = REPO_ROOT / "outputs" / "sibling" / "judge"


async def judge_br_arm(arm, llm, items, cats, *, workers=96):
    from hallucination_bench.evals.sibling_probe import judge as J
    from hallucination_bench.evals.sibling_probe.concepts import Matcher
    from hallucination_bench.readouts import load_readouts

    by_id = {it["id"]: it for it in items}
    cells, rep = load_readouts(BR / "readouts" / arm / "basic_readout.jsonl", ids=set(by_id))
    cells = sorted(cells, key=lambda c: (c.id, c.layer, c.pos))
    d = JUDGE_ROOT / f"br-{arm}"
    d.mkdir(parents=True, exist_ok=True)
    ja = J.JudgeArm(f"br-{arm}", d, Cache(d / "cache.jsonl"), rep.kind or "prose")
    mts = {c: Matcher(cat) for c, cat in cats.items()}
    models = J.JudgeModels()
    recs = [None] * len(cells)
    queue = list(enumerate(cells))
    done = 0

    async def worker():
        nonlocal done
        while queue:
            j, c = queue.pop()
            it = by_id[c.id]
            recs[j] = await J.process_cell(
                ja,
                c,
                it,
                {"n": c.pos + 1},
                llm,
                models,
                in_sample=False,
                summarize_all=False,
                mt=mts[it["category"]],
                judge_all=True,
                extras=False,
            )
            done += 1
            if llm is not None and done % 200 == 0:
                tot = Usage()
                for u in ja.spend.values():
                    tot.add(u)
                print(
                    f"  [br-{arm} {done}/{len(cells)}] calls={tot.calls} spend=${tot.usd:.2f}",
                    flush=True,
                )

    try:
        await asyncio.gather(
            *(worker() for _ in range(max(1, min(workers if llm else 1, len(cells)))))
        )
    finally:
        ja.cache.close()
    write_jsonl(d / "cells.jsonl", recs)
    counts = Counter(r["status"] for r in recs)
    spend = round(sum(v.usd for v in ja.spend.values()), 4)
    from hallucination_bench.judges import claimmap_v2 as cm

    run = {
        "arm": f"br-{arm}",
        "kind": ja.kind,
        "n_cells": len(recs),
        "counts": dict(counts),
        "claimmap_prompt_version": cm.PROMPT_VERSION,
        "model": models.judge,
        "effort": models.effort,
        "judge_all": True,
        "spend_this_run_usd": spend,
    }
    (d / "run.json").write_text(json.dumps(run, indent=1) + "\n")
    print(f"[br-{arm}] {dict(counts)} spend=${spend}")
    return run


async def stage_judge(arms, rebuild: bool, concurrency: int) -> None:
    cats = BB.categories()
    items = BB.judge_items(BB.load_bank(), cats)
    llm = None if rebuild else LLM(rpm=1500, concurrency=concurrency, effort=EFFORT)
    try:
        if llm is not None:
            await llm.preflight(MODEL)
        await asyncio.gather(
            *(judge_br_arm(a, llm, items, cats, workers=2 * concurrency) for a in arms)
        )
    finally:
        if llm is not None:
            await llm.aclose()


# ------------------------------------------------------------------ brnorm (substitution segments)


def norm_key(item_id: str, seg) -> str:
    t = (seg.get("probe_target") or "").strip().lower()
    return (
        f"brnorm:{item_id}:{t}"
        if t
        else f"brnorm:{item_id}:seg:{' '.join(seg['text'].split()).lower()}"
    )


async def stage_brnorm(arms) -> None:
    from hallucination_bench.evals.sibling_probe.claims import SUB_LEAVES

    doc = BB.load_br_concepts()
    bank = {r["id"]: r for r in BB.load_bank()}
    jobs = {}
    for arm in arms:
        for rec in read_jsonl(JUDGE_ROOT / f"br-{arm}" / "cells.jsonl"):
            for s in rec.get("segments") or ():
                if s.get("leaf") in SUB_LEAVES:
                    jobs.setdefault(norm_key(rec["id"], s), (rec, s))
    op = Opus(BB.BRNORM_VERSION, concurrency=32)

    async def one(k, rec, s):
        cat = doc["categories"][rec["category"]]
        mem = [(c["id"], c["label"]) for c in cat["concepts"]]
        sysm, user = BB.render_brnorm(
            bank[rec["id"]]["prompt"], rec["category"], mem, s["text"], s.get("probe_target") or ""
        )
        return await op.call(
            k, sysm, user, BB.BRNORM_SCHEMA, lambda r: BB.valid_brnorm(r, [m[0] for m in mem])
        )

    try:
        outs = await asyncio.gather(*(one(k, *v) for k, v in jobs.items()))
    finally:
        await op.close()
    print(
        f"[brnorm] {len(jobs)} segments; kinds={dict(Counter((o or {}).get('kind') for o in outs))}"
    )
    print(f"[brnorm] calls {op.n_calls}: {op.usage.report()}")


def load_brnorm() -> dict:
    """norm cache key -> {kind, concept} (latest ok row per key)."""
    out = {}
    p = BR / f"cache_{BB.BRNORM_VERSION}.jsonl"
    if p.exists():
        for r in read_jsonl(p):
            if r.get("result"):
                out[r["key"]] = r["result"]
    return out


# ------------------------------------------------------------------ probe texts (brtext-v1)

MIN_GOLDS_FOR_PROBES = 2  # transfer needs other-gold items of the same category


def probe_categories(doc) -> list[str]:
    """Categories whose concepts can pass the mandatory transfer gate (>= 2 distinct golds);
    single-gold categories (every implicit item but the country one) have no contrast items, so
    every claim there is ``probe n/a (unvalidated)`` and no probes are built."""
    return sorted(
        c
        for c, d in doc["categories"].items()
        if len(set(d["gold_of"].values())) >= MIN_GOLDS_FOR_PROBES
    )


def full_text(template: str, passage: str, name: str) -> tuple[str, bool]:
    """(text, chat): entity templates -> plain text passage + template with NAME; the implicit
    chat template (question + newline + answer stem) -> the user turn is passage + question."""
    if "\n" + BB.ANSWER_STEM in template:
        q = template.split("\n")[0]
        return f"{passage} {q}", True
    return f"{passage} {template.replace('NAME', name or 'they')}", False


async def stage_probe_texts() -> None:
    from hallucination_bench.evals.sibling_probe import bank as SB
    from hallucination_bench.evals.sibling_probe.concepts import Matcher

    doc = BB.load_br_concepts()
    cats = BB.categories(doc)
    bank = BB.load_bank()
    eval_index = SB.ngram_index([r["prompt"] for r in bank])
    eval_entities = {(r.get("entity") or "").lower() for r in bank if r.get("entity")}
    op = Opus(BB.BRTEXT_VERSION, concurrency=24)
    pcs = probe_categories(doc)
    need: dict[tuple, int] = {}
    for c in pcs:
        tpls = doc["categories"][c]["templates"]
        for k in range(len(tpls)):
            for cc in doc["categories"][c]["concepts"]:
                for kind in ("explicit", "implied"):
                    need[(c, cc["id"], kind, k)] = -(-PROBES_PER_KIND // len(tpls))
            need[(c, None, "neutral", k)] = -(-NEUTRAL_PER_CAT // len(tpls))
    kept: dict[tuple, list] = defaultdict(list)
    seen: set[str] = set()
    stats: Counter = Counter()
    for rnd in range(5):
        jobs = []
        for key, n in need.items():
            if len(kept[key]) >= n:
                continue
            c, cid, kind, k = key
            d = doc["categories"][c]
            con = {x["id"]: x for x in d["concepts"]}
            label = con[cid]["label"] if cid else None
            aliases = con[cid]["aliases"] if cid else []
            others = [x["label"] for x in d["concepts"] if x["id"] != cid]
            sysm, user = BB.render_brtext(
                c, d["templates"][k], kind, n - len(kept[key]) + 4, label, aliases, others
            )
            jobs.append((key, rnd, sysm, user))
        if not jobs:
            break
        print(f"[probe-texts] round {rnd}: {len(jobs)} calls", flush=True)
        outs = await asyncio.gather(
            *(
                op.call(
                    f"brtext:{key[0]}:{key[1] or 'neutral'}:{key[2]}:t{key[3]}:r{r}",
                    sm,
                    us,
                    BB.BRTEXT_SCHEMA,
                    lambda x: isinstance(x.get("texts"), list),
                )
                for key, r, sm, us in jobs
            )
        )
        for (key, _r, _s, _u), out in zip(jobs, outs, strict=True):
            c, cid, kind, k = key
            m = Matcher(cats[c])
            tpl = doc["categories"][c]["templates"][k]
            for t in (out or {}).get("texts") or ():
                if len(kept[key]) >= need[key]:
                    break
                passage = " ".join(str(t.get("text", "")).split())
                name = " ".join(str(t.get("name", "")).split())
                stats["generated"] += 1
                hits = m.mentions(passage) | m.mentions(name)
                if not passage or len(passage.split()) < 8:
                    stats["format"] += 1
                elif (kind == "explicit" and hits != {cid}) or (kind != "explicit" and hits):
                    stats["mentions"] += 1
                elif "NAME" in tpl and (not name or name.lower() in eval_entities):
                    stats["name"] += 1
                elif SB.normalize_cue(passage) in seen:
                    stats["duplicate"] += 1
                elif SB.shares_ngram(passage, eval_index):
                    stats["ngram8"] += 1
                else:
                    seen.add(SB.normalize_cue(passage))
                    kept[key].append((passage, name))
                    stats["kept"] += 1
    rows, short = [], []
    for key, n in need.items():
        c, cid, kind, k = key
        if len(kept[key]) < n:
            short.append((key, len(kept[key])))
        tpl = doc["categories"][c]["templates"][k]
        for j, (passage, name) in enumerate(kept[key]):
            text, chat = full_text(tpl, passage, name)
            tag = {"explicit": "e", "implied": "i", "neutral": "n"}[kind]
            rows.append(
                {
                    "id": f"bpt-{c}-{cid or 'neutral'}-{tag}{k}-{j:02d}".replace("_", "-"),
                    "category": c,
                    "concept": cid,
                    "kind": kind,
                    "request_idx": k,
                    "passage": passage,
                    "name": name,
                    "text": text,
                    "chat": chat,
                }
            )
    write_jsonl(BB.PROBE_TEXTS_PATH, rows)
    print(f"[probe-texts] categories {pcs}; {len(rows)} texts; filters {dict(stats)}")
    if short:
        print(f"[probe-texts] SHORT {len(short)}: {short[:10]}")
    print(f"[probe-texts] calls {op.n_calls}: {op.usage.report()}")
    await op.close()


# ------------------------------------------------------------------ capture rows

CAP_LAYERS = "20,24,28,32,36,40,42,44,48,52,56,60"


def stage_capture_rows() -> None:
    tok = tokenizer()
    bank = BB.load_bank()
    man = json.loads((BR / "pubplan_manifest.json").read_text())
    mrows = {p["label"]: p for p in man["prompts"]}
    ev = []
    for r in bank:
        ids = r["input_ids"]
        dec = [tok.convert_ids_to_tokens(i) for i in ids]
        mt = mrows[r["id"]]["tokens"]
        disp = [t.replace("Ġ", " ").replace("▁", " ") for t in dec]
        if disp != mt:
            raise SystemExit(f"{r['id']}: tokens differ from pubplan manifest: {disp} vs {mt}")
        ev.append(
            {
                "label": r["id"],
                "family": "br_eval",
                "input_ids": ids,
                "targets": [],
                "look_for": "",
                "eval_positions": [len(ids) - 1],
            }
        )
    (BR / "capture").mkdir(parents=True, exist_ok=True)
    (BR / "capture" / "eval.json").write_text(json.dumps(ev))
    pr = []
    for t in read_jsonl(BB.PROBE_TEXTS_PATH):
        if t["chat"]:
            ids = BB.render_ids(tok, {"prompt": t["text"], "eval_render": "chat_prefill"})
        else:
            ids = list(tok(t["text"], add_special_tokens=False)["input_ids"])
        pr.append(
            {
                "label": t["id"],
                "family": f"br_probe_{t['category']}",
                "input_ids": ids,
                "targets": [],
                "look_for": "",
                "eval_positions": [len(ids) - 1],
            }
        )
    (BR / "capture" / "probe.json").write_text(json.dumps(pr))
    mx = max(len(r["input_ids"]) for r in ev + pr)
    print(
        f"[capture-rows] eval {len(ev)} rows (tokens == pubplan manifest), probe {len(pr)} rows, max len {mx}"
    )


# ------------------------------------------------------------------ probe inputs


def stage_probe_inputs() -> None:
    from hallucination_bench.evals.sibling_probe.probe_io import git_sha

    doc = BB.load_br_concepts()
    pcs = probe_categories(doc)
    bank = [r for r in BB.load_bank() if r.get("in_pipeline") and r["category"] in pcs]
    texts = read_jsonl(BB.PROBE_TEXTS_PATH)
    out = {
        "concepts": {c: [x["id"] for x in doc["categories"][c]["concepts"]] for c in pcs},
        "items": [
            {
                "id": r["id"],
                "category": r["category"],
                "gold": r["gold_concept"],
                "request_idx": r["template_idx"],
            }
            for r in bank
        ],
        "probe_texts": [
            {k: t[k] for k in ("id", "category", "concept", "kind", "request_idx")} for t in texts
        ],
        "token_ids": {},
        "git_sha": git_sha(),
    }
    p = BR / "probes" / "inputs.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out))
    print(
        f"[probe-inputs] {len(out['items'])} items, {len(out['probe_texts'])} texts, categories {pcs} -> {p}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "stage",
        choices=[
            "bank-in",
            "bank",
            "inventory",
            "judge",
            "brnorm",
            "probe-texts",
            "capture-rows",
            "probe-inputs",
        ],
    )
    ap.add_argument("--arms", default=",".join(BR_ARMS))
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--concurrency", type=int, default=32)
    a = ap.parse_args()
    arms = [x for x in a.arms.split(",") if x]
    if a.stage == "bank-in":
        stage_bank_in()
    elif a.stage == "bank":
        asyncio.run(stage_bank())
    elif a.stage == "inventory":
        asyncio.run(stage_inventory())
    elif a.stage == "judge":
        asyncio.run(stage_judge(arms, a.rebuild, a.concurrency))
    elif a.stage == "brnorm":
        asyncio.run(stage_brnorm(arms))
    elif a.stage == "probe-texts":
        asyncio.run(stage_probe_texts())
    elif a.stage == "capture-rows":
        stage_capture_rows()
    elif a.stage == "probe-inputs":
        stage_probe_inputs()


if __name__ == "__main__":
    main()
