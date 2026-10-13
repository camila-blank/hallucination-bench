"""Plan 0006 driver: the user-modelling (country) sibling ladder. One stage per invocation.

  summarize   J-lens summaries for every country cell with a non-gold decoded token hit that
              has none (plan 0002 interp-v1, verbatim; existing plan 0004 ``summ:`` rows are
              reused read-only when their fingerprint matches) -> um/jlens_summ.jsonl
  mentions    units + strata -> outputs/sibling/um/{mentions,strata}.jsonl, counts.json
  todo        dev (40) / gold (80) calibration units -> data/.../umcommit_{dev,gold}_todo.jsonl
  judge       umcommit-v1 per (unit, sample) -> outputs/sibling/um/commit.jsonl
  calib       judge vs data/sibling_probe/umcommit_{dev,gold}.jsonl -> outputs/sibling/um/calib.json

Inputs come from ``--src-root`` (default /workspace/hb-sibling-probes, read only): readouts,
the plan 0004 J-lens summary cache. Keyless Anthropic auth (repo convention).

    uv run --no-sync python scripts/sibling_probe/um_ladder.py mentions
"""

import argparse
import asyncio
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from hallucination_bench import REPO_ROOT, summarizer
from hallucination_bench.cache import Cache, fingerprint
from hallucination_bench.evals.sibling_probe import um_ladder as U
from hallucination_bench.llm import LLM, Usage
from hallucination_bench.readouts import decode_token

OUT = REPO_ROOT / "outputs" / "sibling" / "um"
DATA = REPO_ROOT / "data" / "sibling_probe"
SRC_DEFAULT = Path("/workspace/hb-sibling-probes")
MODEL = "claude-opus-5-5"
FALLBACK_MODEL = "claude-sonnet-5-5"
EFFORT = "low"
READOUTS = {
    "s3d": "s3d-rl600-all11/sibling_eval.jsonl",
    "mlnla-rl600": "mlnla-rl600-all11/sibling_country.jsonl",
    "jlens": "jlens-all11/sibling_country.jsonl",
    U.NLA42: "nla-rl-L42/sibling_country.jsonl",
}


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def write_jsonl(p: Path, rows: list[dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(p)


def country_items(src: Path) -> dict[str, dict[str, Any]]:
    bank = json.loads((src / "data" / "sibling_probe" / "items.json").read_text())
    return {it["id"]: it for it in bank["items"] if it["category"] == U.CATEGORY}


def readout_rows(src: Path, method: str, items: dict[str, Any]) -> list[dict[str, Any]]:
    p = src / "outputs" / "sibling" / "readouts" / READOUTS[method]
    rows = [r for r in read_jsonl(p) if r["id"] in items]
    keys = Counter((r["id"], r["layer"], r["pos"]) for r in rows)
    dup = [k for k, n in keys.items() if n > 1]
    if dup:
        sys.exit(f"{method}: {len(dup)} duplicated cells, e.g. {dup[:3]}")
    return rows


# ------------------------------------------------------------------ summaries


def summary_request(row: dict[str, Any]) -> tuple[str, str]:
    decoded = [decode_token(t) for t in row.get("tokens") or ()]
    user = summarizer.render_user(summarizer.render_bag(decoded, row.get("scores")))
    fp = fingerprint(
        summarizer.SUMMARIZER_PROMPT_VERSION, MODEL, EFFORT, summarizer.INTERP_SYSTEM, user
    )
    return user, fp


class Summaries:
    """Plan 0004 cache (read only) + this plan's cache, same key and fingerprint."""

    def __init__(self, src: Path, *, readonly: bool = True) -> None:
        old = src / "outputs" / "sibling" / "judge" / "jlens-all11" / "cache.jsonl"
        self.old = Cache(old, readonly=True) if old.exists() else None
        self.new = Cache(OUT / "jlens_summ.jsonl", readonly=readonly)

    def get(self, row: dict[str, Any]) -> tuple[str | None, str]:
        key = "summ:" + U.cell_key(row["id"], int(row["layer"]), int(row["pos"]))
        _, fp = summary_request(row)
        for c in (self.new, self.old):
            if c is None:
                continue
            hit = c.get(key, fp)
            if hit is not None:
                return summarizer.interpretation_of(hit["result"]), "ok"
        failed = self.new.get(key, fp, include_failed=True)
        return None, ("failed" if failed is not None else "missing")


async def _summarize(rows: list[dict[str, Any]], summ: Summaries, rpm: float, conc: int) -> Usage:
    llm = LLM(rpm=rpm, concurrency=conc, effort=EFFORT)
    await llm.preflight(MODEL)
    spend = Usage()

    async def one(row: dict[str, Any]) -> None:
        key = "summ:" + U.cell_key(row["id"], int(row["layer"]), int(row["pos"]))
        user, fp = summary_request(row)
        res = None
        usage = Usage()
        for _ in range(3):  # 1 + 2 in-run retries
            out = await llm.call(
                model=MODEL,
                system=summarizer.INTERP_SYSTEM,
                user=user,
                schema=summarizer.INTERP_SCHEMA,
            )
            usage.add(out.usage)
            if summarizer.interpretation_of(out.result) is not None:
                res = out.result
                break
        spend.add(usage)
        summ.new.put(
            key,
            fp,
            {"result": res, "meta": {"stage": "summary", "model": MODEL, "usage": usage.to_dict()}},
        )

    await asyncio.gather(*(one(r) for r in rows))
    await llm.aclose()
    return spend


def cmd_summarize(a: argparse.Namespace) -> None:
    items = country_items(a.src_root)
    fm = U.FastMatcher(U.load_lexicon())
    ctx = {i: U.item_context(fm, it) for i, it in items.items()}
    summ = Summaries(a.src_root, readonly=False)
    todo, have = [], 0
    for row in readout_rows(a.src_root, "jlens", items):
        if not U.jlens_needs_summary(fm, row, ctx[row["id"]]):
            continue
        _text, status = summ.get(row)
        if status == "ok":
            have += 1
        else:
            todo.append(row)
    print(f"cells needing a summary: {have + len(todo)} (cached {have}, to call {len(todo)})")
    if a.dry_run or not todo:
        return
    spend = asyncio.run(_summarize(todo, summ, a.rpm, a.concurrency))
    print("summaries:", spend.report())


# ------------------------------------------------------------------ mentions


def cmd_mentions(a: argparse.Namespace) -> None:
    items = country_items(a.src_root)
    fm = U.FastMatcher(U.load_lexicon())
    ctx = {i: U.item_context(fm, it) for i, it in items.items()}
    summ = Summaries(a.src_root)
    units: list[dict[str, Any]] = []
    strata: list[dict[str, Any]] = []
    cells = Counter()
    summ_status = Counter()
    for method in U.METHODS:
        for row in readout_rows(a.src_root, method, items):
            cells[method] += 1
            text = status = None
            if method == "jlens" and U.jlens_needs_summary(fm, row, ctx[row["id"]]):
                text, status = summ.get(row)
                summ_status[status] += 1
            us, ss = U.build_mentions(fm, method, row, ctx[row["id"]], text, status)
            for u in us:
                for smp in u["samples"]:
                    for h in smp["hits"]:
                        h["via_city"] = fm.via_city(h["alias"])
            units.extend(us)
            strata.extend(ss)
        print(f"{method}: {cells[method]} cells", flush=True)
    write_jsonl(OUT / "mentions.jsonl", units)
    write_jsonl(OUT / "strata.jsonl", strata)
    by_m = Counter(u["method"] for u in units)
    counts = {
        "cells": dict(cells),
        "units": dict(by_m),
        "samples_to_judge": dict(Counter(u["method"] for u in units for _ in u["samples"])),
        "distinct_concepts": {
            m: len({u["concept"] for u in units if u["method"] == m}) for m in U.METHODS
        },
        "strata": {
            f"{m}:{k}": n
            for (m, k), n in Counter((s["method"], s["stratum"]) for s in strata).items()
        },
        "via_city_units": sum(
            any(h.get("via_city") for s in u["samples"] for h in s["hits"]) for u in units
        ),
        "jlens_summary_status": dict(summ_status),
        "top_concepts": Counter(u["concept"] for u in units).most_common(60),
        "top_pairs": [
            (f"{g}->{c}", n)
            for (g, c), n in Counter((u["gold"], u["concept"]) for u in units).most_common(40)
        ],
        "response_concepts": {
            i: sorted(c["response_concepts"] - c["equiv"])
            for i, c in ctx.items()
            if c["response_concepts"] - c["equiv"]
        },
    }
    (OUT / "counts.json").write_text(json.dumps(counts, indent=1, ensure_ascii=False))
    print(
        json.dumps(
            {
                k: counts[k]
                for k in ("units", "samples_to_judge", "jlens_summary_status", "via_city_units")
            },
            indent=1,
        )
    )
    if summ_status.get("missing"):
        print(f"WARNING: {summ_status['missing']} J-lens cells lack a summary: run `summarize`")


# ------------------------------------------------------------------ commitment judge


def commit_key(uid: str, si: int) -> str:
    return f"c:{uid}:{si}"


def commit_requests(
    units: list[dict[str, Any]], items: dict[str, Any], fm: U.FastMatcher
) -> list[tuple[str, str, str, dict[str, Any], dict[str, Any]]]:
    """(key, system, user, unit, sample) for every (unit, sample)."""
    out = []
    for u in units:
        label = fm.concepts[u["concept"]]["label"]
        for smp in u["samples"]:
            system, user = U.render_commit(u, smp, items[u["item"]], label)
            out.append((commit_key(u["uid"], smp["si"]), system, user, u, smp))
    return out


def commit_fp(model: str, system: str, user: str) -> str:
    return fingerprint(U.COMMIT_PROMPT_VERSION, model, EFFORT, system, user)


def cached_commit(cache: Cache, key: str, system: str, user: str) -> dict[str, Any] | None:
    for model in (MODEL, FALLBACK_MODEL):
        row = cache.get(key, commit_fp(model, system, user))
        if row is not None:
            return row
    return None


async def _judge(reqs: list[Any], cache: Cache, rpm: float, conc: int) -> Usage:
    llm = LLM(rpm=rpm, concurrency=conc, effort=EFFORT)
    await llm.preflight(MODEL)
    spend = Usage()
    done = 0

    async def one(key: str, system: str, user: str, smp: dict[str, Any]) -> None:
        nonlocal done
        usage = Usage()
        res, model, raw = None, MODEL, None
        for model in (MODEL, FALLBACK_MODEL):
            for _ in range(3):  # 1 + 2 in-run retries of an invalid answer
                out = await llm.call(model=model, system=system, user=user, schema=U.COMMIT_SCHEMA)
                usage.add(out.usage)
                if out.result is not None and U.valid_commit(out.result, smp):
                    res = out.result
                    break
                raw = out.result if out.result is not None else raw
            if res is not None or usage.refusals == 0:
                break  # fall back to Sonnet only after a refusal
        spend.add(usage)
        cache.put(
            key,
            commit_fp(model, system, user),
            {
                "result": res,
                "meta": {
                    "stage": "umcommit",
                    "model": model,
                    "usage": usage.to_dict(),
                    "raw": None if res is not None else raw,
                },
            },
        )
        done += 1
        if done % 250 == 0:
            print(f"  judged {done}/{len(reqs)}  {spend.report()}", flush=True)

    await asyncio.gather(*(one(k, s, u, smp) for k, s, u, _, smp in reqs))
    await llm.aclose()
    return spend


def load_units() -> list[dict[str, Any]]:
    return read_jsonl(OUT / "mentions.jsonl")


def select_units(units: list[dict[str, Any]], ids_file: Path | None) -> list[dict[str, Any]]:
    if ids_file is None:
        return units
    want = {json.loads(line)["uid"] for line in ids_file.read_text().splitlines() if line.strip()}
    return [u for u in units if u["uid"] in want]


def cmd_judge(a: argparse.Namespace) -> None:
    items = country_items(a.src_root)
    fm = U.FastMatcher(U.load_lexicon())
    units = select_units(load_units(), a.ids)
    cache = Cache(OUT / f"cache_{U.COMMIT_PROMPT_VERSION}.jsonl")
    reqs = commit_requests(units, items, fm)
    todo = [r for r in reqs if cached_commit(cache, r[0], r[1], r[2]) is None]
    print(f"{U.COMMIT_PROMPT_VERSION}: {len(reqs)} (unit, sample) calls, {len(todo)} to call")
    if todo and not a.dry_run:
        spend = asyncio.run(_judge(todo, cache, a.rpm, a.concurrency))
        print("judge:", spend.report())
        (OUT / f"judge_run_{U.COMMIT_PROMPT_VERSION}.json").write_text(
            json.dumps(
                {"model": MODEL, "effort": EFFORT, "n_calls": len(todo), "usage": spend.to_dict()},
                indent=1,
            )
        )
    if a.dry_run:
        print(reqs[0][1][:400] + "\n...\n" + reqs[0][2])
        return
    rows, failed = [], 0
    for key, system, user, u, smp in reqs:
        row = cached_commit(cache, key, system, user)
        v = U.parse_commit(row["result"] if row else None, smp)
        failed += v["fail_closed"]
        rows.append(
            {
                "uid": u["uid"],
                "si": smp["si"],
                **v,
                "model": (row or {}).get("meta", {}).get("model"),
            }
        )
    by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_uid[r["uid"]].append(r)
    out = []
    for u in units:
        vs = by_uid[u["uid"]]
        out.append(
            {
                "uid": u["uid"],
                "method": u["method"],
                "item": u["item"],
                "layer": u["layer"],
                "pos": u["pos"],
                "concept": u["concept"],
                "gold": u["gold"],
                **U.rollup(vs),
                "samples": vs,
            }
        )
    name = "commit.jsonl" if a.ids is None else f"commit_{a.ids.stem}.jsonl"
    write_jsonl(OUT / name, out)
    c = Counter((r["method"], r["committed"]) for r in out)
    print(f"wrote {name}: {len(out)} units, fail-closed samples {failed}")
    for m in U.METHODS:
        n1, n0 = c[(m, True)], c[(m, False)]
        if n1 + n0:
            print(f"  {m}: committed {n1}/{n1 + n0} = {n1 / (n1 + n0):.2f}")


# ------------------------------------------------------------------ calibration sets

TODO_FIELDS = ("uid", "method", "item", "layer", "pos", "concept", "gold")


def _todo_row(u: dict[str, Any], items: dict[str, Any], fm: U.FastMatcher) -> dict[str, Any]:
    smps = []
    for smp in u["samples"]:
        _, user = U.render_commit(u, smp, items[u["item"]], fm.concepts[u["concept"]]["label"])
        smps.append(
            {"si": smp["si"], "shown": user, "committed": None, "form": None, "hedged": None}
        )
    return {
        **{k: u[k] for k in TODO_FIELDS if k != "gold"},
        "samples": smps,
        "unit_committed": None,
    }


def cmd_todo(a: argparse.Namespace) -> None:
    """Dev 40 / gold 80: stratified by method (10 / 20 each), seeded, disjoint. The gold is not
    shown (the judge does not see it either)."""
    items = country_items(a.src_root)
    fm = U.FastMatcher(U.load_lexicon())
    units = load_units()
    rng = random.Random(6)
    by_m: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for u in units:
        by_m[u["method"]].append(u)
    dev, gold = [], []
    for m in U.METHODS:
        pool = sorted(by_m[m], key=lambda u: u["uid"])
        rng.shuffle(pool)
        dev += pool[:10]
        gold += pool[10:30]
    write_jsonl(DATA / "umcommit_dev_todo.jsonl", [_todo_row(u, items, fm) for u in dev])
    write_jsonl(DATA / "umcommit_gold_todo.jsonl", [_todo_row(u, items, fm) for u in gold])
    print(f"dev {len(dev)}, gold {len(gold)} written to {DATA}")


def cmd_calib(a: argparse.Namespace) -> None:
    """Judge (unit level and sample level) vs hand labels; Wilson CIs."""
    from math import sqrt

    def wilson(k: int, n: int) -> list[float]:
        if n == 0:
            return [float("nan")] * 2
        z, p = 1.96, k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return [round(c - h, 3), round(c + h, 3)]

    res = {}
    for split in ("dev", "dev2", "gold"):
        lab_p = DATA / f"umcommit_{split}.jsonl"
        com_p = OUT / f"commit_umcommit_{split}_todo.jsonl"
        if not lab_p.exists() or not com_p.exists():
            continue
        labs = {r["uid"]: r for r in read_jsonl(lab_p)}
        judged = {r["uid"]: r for r in read_jsonl(com_p)}
        k = n = 0
        conf: Counter[tuple[bool, bool]] = Counter()
        sk = sn = 0
        dis = []
        for uid, lab in labs.items():
            j = judged.get(uid)
            if j is None or lab.get("unit_committed") is None:
                continue
            g, p = bool(lab["unit_committed"]), bool(j["committed"])
            conf[(g, p)] += 1
            n += 1
            k += g == p
            if g != p:
                dis.append(
                    {
                        "uid": uid,
                        "gold": g,
                        "judge": p,
                        "judge_quote": j["quote"],
                        "judge_forms": j["forms"],
                    }
                )
            jv = {s["si"]: s for s in j["samples"]}
            for s in lab["samples"]:
                if s.get("committed") is None or s["si"] not in jv:
                    continue
                sn += 1
                sk += bool(s["committed"]) == bool(jv[s["si"]]["committed"])
        prec = conf[(True, True)] / max(1, conf[(True, True)] + conf[(False, True)])
        rec = conf[(True, True)] / max(1, conf[(True, True)] + conf[(True, False)])
        res[split] = {
            "n": n,
            "accuracy": round(k / max(1, n), 3),
            "wilson": wilson(k, n),
            "sample_n": sn,
            "sample_accuracy": round(sk / max(1, sn), 3),
            "committed_precision": round(prec, 3),
            "committed_recall": round(rec, 3),
            "confusion": {f"gold={g},judge={p}": v for (g, p), v in conf.items()},
            "target_met": (k / max(1, n)) >= 0.85,
            "disagreements": dis,
        }
    (OUT / "calib.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for sp, r in res.items():
        print(sp, {k: v for k, v in r.items() if k != "disagreements"})


# ------------------------------------------------------------------ step 4a: concept set C

CONCEPTS_PATH = DATA / "um_country_concepts.json"
PROBE_TEXTS_PATH = DATA / "um_country_probe_texts.jsonl"
MIN_COMMITTED = 3
CAP = 80
POOL_PER_GOLD = 2
# informal own-concept forms accepted in an explicit probe text
INFORMAL = {
    "australia": ["Aussie", "Aussies"],
    "new_zealand": ["Kiwi", "Kiwis"],
    "united_kingdom": ["Brit", "Brits"],
    "united_states": ["Yank", "Yanks"],
}


def cmd_concepts(a: argparse.Namespace) -> None:
    from hallucination_bench.evals.sibling_probe.um_regions import ADJACENT, REGION_OF, REGIONS

    items = country_items(a.src_root)
    golds = sorted({U.gold_concept(it["gold"]) for it in items.values()})
    units = load_units()
    commits = {r["uid"]: r for r in read_jsonl(OUT / "commit.jsonl")}
    missing = [u["uid"] for u in units if u["uid"] not in commits]
    if missing:
        sys.exit(f"{len(missing)} units have no commit verdict: run `judge` first")
    n_com = Counter(u["concept"] for u in units if commits[u["uid"]]["committed"])
    n_men = Counter(u["concept"] for u in units)
    sibs = sorted(
        (c for c, n in n_com.items() if n >= MIN_COMMITTED and c not in golds and c != U.SOVIET),
        key=lambda c: (-n_com[c], c),
    )
    C = set(golds) | set(sibs)
    pool: list[tuple[str, str]] = []  # (gold, concept)
    for g in golds:
        took = 0
        for reg in (REGION_OF[g], *ADJACENT[REGION_OF[g]]):
            cands = sorted((c for c in REGIONS[reg] if c not in C), key=lambda c: (-n_men[c], c))
            for c in cands:
                if took >= POOL_PER_GOLD:
                    break
                pool.append((g, c))
                C.add(c)
                took += 1
            if took >= POOL_PER_GOLD:
                break
    # cap: trim pool first (least-mentioned, never below 1 per gold), then siblings
    trimmed_pool, trimmed_sibs = [], []
    while len(C) > CAP:
        per_g = Counter(g for g, _ in pool)
        cand = sorted((x for x in pool if per_g[x[0]] > 1), key=lambda x: (n_men[x[1]], x[1]))
        if cand:
            pool.remove(cand[0])
            C.discard(cand[0][1])
            trimmed_pool.append(cand[0][1])
            continue
        drop = min(sibs, key=lambda c: (n_com[c], c))
        sibs.remove(drop)
        C.discard(drop)
        trimmed_sibs.append(drop)
    lex = U.load_lexicon()["categories"][U.CATEGORY]["concepts"]
    role = {c: [] for c in C}
    for g in golds:
        role[g].append("gold")
    for c in sibs:
        role[c].append("sibling")
    for g, c in pool:
        role[c].append(f"pool:{g}")
    doc = {
        "version": "umconcepts-v1",
        "rule": f"golds + siblings with >= {MIN_COMMITTED} committed units + {POOL_PER_GOLD} "
        f"most-mentioned same-region countries per gold; cap {CAP} (pool trimmed first)",
        "concepts": [
            {
                "id": c,
                "label": lex[c]["label"],
                "roles": role[c],
                "region": REGION_OF[c],
                "n_committed": n_com[c],
                "n_mentions": n_men[c],
                "informal": INFORMAL.get(c, []),
            }
            for c in sorted(C)
        ],
        "n_golds": len(golds),
        "n_siblings": len(sibs),
        "n_pool": len(pool),
        "trimmed_pool": trimmed_pool,
        "trimmed_siblings": trimmed_sibs,
        "committed_not_in_C": sorted(c for c in n_com if c not in C and c != U.SOVIET),
    }
    CONCEPTS_PATH.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    print(
        f"C = {len(C)}: {len(golds)} golds, {len(sibs)} siblings, {len(pool)} pool; trimmed "
        f"pool {len(trimmed_pool)}, siblings {len(trimmed_sibs)}; committed without a probe "
        f"{len(doc['committed_not_in_C'])} concepts"
    )


def load_C() -> list[dict[str, Any]]:
    return json.loads(CONCEPTS_PATH.read_text())["concepts"]


# ------------------------------------------------------------------ step 4a: probe texts

PER_KIND = 24
NEUTRAL_N = 24


def _probe_category(C: list[dict[str, Any]], request: str) -> Any:
    from hallucination_bench.evals.sibling_probe.concepts import Category, Concept

    lex = U.load_lexicon()
    con = lex["categories"][U.CATEGORY]["concepts"]
    cities = set(lex.get("set_a_city_aliases", {}))
    concepts = tuple(
        Concept(
            id=c["id"],
            label=c["label"],
            category=U.CATEGORY,
            gold="gold" in c["roles"],
            aliases=tuple(
                a for a in [*con[c["id"]]["en"], *con[c["id"]]["cjk"]] if a not in cities
            ),
        )
        for c in C
    )
    return Category(
        name=U.CATEGORY, question=U.ATTRIBUTE_QUESTION, requests=(request,), concepts=concepts
    )


def text_check(
    fm: U.FastMatcher, cid: str | None, kind: str, text: str, informal: dict[str, list[str]]
) -> str | None:
    """None if a probe text passes the country check, else the failure cause."""
    found = {U._resolve(c, "") if c != U.SOVIET else c for c in fm.concepts_in(text)}
    if kind == "explicit":
        own = U.gold_equivalents(cid or "")
        other = found - own
        if other:
            return "names_other:" + ",".join(sorted(other))
        if not (found & own) and not any(
            re.search(rf"(?<![A-Za-z]){re.escape(f)}(?![A-Za-z])", text)
            for f in informal.get(cid or "", [])
        ):
            return "own_missing"
        return None
    if found:
        return "names_country:" + ",".join(sorted(found))
    return None


def cmd_probe_texts(a: argparse.Namespace) -> None:
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "sibling_probe"))
    import gen_items as G  # type: ignore[import-not-found]
    from transformers import AutoTokenizer

    from hallucination_bench.evals.sibling_probe import bank as B

    items = country_items(a.src_root)
    request = next(iter(items.values()))["request"]
    assert all(it["request"] == request for it in items.values())
    C = load_C()
    ids_C = [c["id"] for c in C]
    informal = {c["id"]: c["informal"] for c in C}
    fm = U.FastMatcher(U.load_lexicon())
    wc = B.WindowChecker(AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B"))
    all_items = json.loads((a.src_root / "data" / "sibling_probe" / "items.json").read_text())
    index = B.ngram_index([it["cue"] for it in all_items["items"]])
    old = [
        t
        for t in read_jsonl(a.src_root / "data" / "sibling_probe" / "probe_texts.jsonl")
        if t["category"] == U.CATEGORY
    ]
    kept: dict[tuple[str | None, str], list[dict[str, Any]]] = defaultdict(list)
    causes: Counter[str] = Counter()
    seen: set[str] = set()
    for t in old:
        cid = U.gold_concept(t["concept"]) if t["concept"] else None
        if cid is not None and cid not in ids_C:
            continue
        why = text_check(fm, cid, t["kind"], t["context"], informal)
        if why is not None:
            causes["old:" + why.split(":")[0]] += 1
            continue
        kept[(cid, t["kind"])].append({**t, "concept": cid, "source": "plan0003"})
        seen.add(B.normalize_cue(t["context"]))
    print(
        f"plan 0003 texts kept {sum(len(v) for v in kept.values())}/{len(old)}; drops "
        f"{dict(causes)}"
    )
    need = {(c, k): PER_KIND for c in ids_C for k in ("explicit", "implied")}
    need[(None, "neutral")] = NEUTRAL_N
    cat = _probe_category(C, request)
    gen = G.Gen()
    attempts: Counter[tuple[str | None, str]] = Counter()

    async def run() -> None:
        for rnd in range(6):
            jobs = []
            for key, n in need.items():
                if len(kept[key]) >= n:
                    continue
                cid, kind = key
                want = n - len(kept[key]) + 6
                prompt = G.probe_prompt(cat, cid, kind, want, request)
                jobs.append((key, attempts[key], prompt))
                attempts[key] += 1
            if not jobs:
                return
            print(f"[probe-texts] round {rnd}: {len(jobs)} Opus calls", flush=True)
            outs = await asyncio.gather(
                *(
                    gen.call(f"umprobe:{k[0] or 'neutral'}:{k[1]}:a{n}", p, "texts")
                    for k, n, p in jobs
                )
            )
            for (key, _n, _p), texts in zip(jobs, outs, strict=True):
                cid, kind = key
                for t in texts:
                    if len(kept[key]) >= need[key]:
                        break
                    t = " ".join(t.split())
                    if t.endswith(request):
                        t = t[: -len(request)].strip()
                    why = text_check(fm, cid, kind, t, informal)
                    if not B.cue_ok(t):
                        why = "format"
                    elif why is None and B.normalize_cue(t) in seen:
                        why = "duplicate"
                    elif why is None and B.shares_ngram(t, index):
                        why = "ngram8"
                    elif why is None and not wc.ok(B.messages_for(t, request), request):
                        why = "window"
                    if why is not None:
                        causes["new:" + why.split(":")[0]] += 1
                        continue
                    seen.add(B.normalize_cue(t))
                    j = len([x for x in kept[key] if x["source"] == "umgen"])
                    tag = {"explicit": "e", "implied": "i", "neutral": "n"}[kind]
                    slug = (cid or "neutral").replace("_", "-")
                    kept[key].append(
                        {
                            "id": f"up-country-{slug}-{tag}0-{j:02d}",
                            "category": U.CATEGORY,
                            "concept": cid,
                            "kind": kind,
                            "request_idx": 0,
                            "context": t,
                            "messages": B.messages_for(t, request),
                            "source": "umgen",
                        }
                    )
        await gen.close()

    asyncio.run(run())
    rows = [r for key in need for r in kept[key][: need[key]]]
    short = {f"{k[0]}:{k[1]}": len(kept[k]) for k in need if len(kept[k]) < need[k]}
    write_jsonl(PROBE_TEXTS_PATH, rows)
    print(
        f"[probe-texts] {len(rows)} texts -> {PROBE_TEXTS_PATH}; causes {dict(causes)}; "
        f"short {short}; Opus {gen.usage.report()}"
    )
    (OUT / "probe_texts_report.json").write_text(
        json.dumps(
            {
                "drops_by_cause": dict(causes),
                "short": short,
                "n": len(rows),
                "from_plan0003": sum(r["source"] == "plan0003" for r in rows),
            },
            indent=1,
        )
    )


# ------------------------------------------------------------------ step 4a: capture rows

CAPTURE_PREFIX = "umprobe"


def cmd_capture_rows(a: argparse.Namespace) -> None:
    from hallucination_bench.evals.sibling_probe import bank as B

    rows = read_jsonl(PROBE_TEXTS_PATH)
    n = B.write_plan_files(rows, OUT / "plan_probe", CAPTURE_PREFIX, layers=U.PROBE_LAYERS)
    print(f"plan -> {OUT / 'plan_probe'}: {n}; layers {list(U.PROBE_LAYERS)}")


# ------------------------------------------------------------------ step 4: probe inputs

PROBE_RUN = "um-country-v2"
PROBE_ACTS = "acts-um-country-probe"
EVAL_ACTS = {str(L): "acts-sibling-eval-all11" for L in U.GRID_LAYERS} | {"42": "acts-sibling-eval"}
PROBES_DIR = OUT / "probes"
DISTRACTORS_PATH = DATA / "um_country_distractors.json"


def cmd_probe_inputs(a: argparse.Namespace) -> None:
    from hallucination_bench.evals.sibling_probe import probes as P

    C = [c["id"] for c in load_C()]
    ci = {c: i for i, c in enumerate(C)}
    texts = read_jsonl(PROBE_TEXTS_PATH)
    folds = P.assign_folds(texts, salt="country-um")
    rows = [
        {
            "id": t["id"],
            "c": ci[t["concept"]] if t["concept"] else P.NEUTRAL,
            "kind": t["kind"],
            "request_idx": t["request_idx"],
            "fold": folds[t["id"]],
        }
        for t in texts
    ]
    items = sorted(country_items(a.src_root))
    inp = {
        "run": PROBE_RUN,
        "concepts": C,
        "texts": rows,
        "items": items,
        "layers": list(U.PROBE_LAYERS),
        "probe_acts": PROBE_ACTS,
        "pack_run": "um-country-v1",
        "eval_acts": EVAL_ACTS,
    }
    (OUT / "probe_inputs.json").write_text(json.dumps(inp))
    print(
        f"probe inputs: {len(C)} concepts, {len(rows)} texts, {len(items)} items, "
        f"layers {list(U.PROBE_LAYERS)} -> {OUT / 'probe_inputs.json'}"
    )


def load_fit() -> tuple[dict[tuple[str, int, str], float], dict[int, Any]]:
    import numpy as np

    C = json.loads((OUT / "probe_inputs.json").read_text())["concepts"]
    validity: dict[tuple[str, int, str], float] = {}
    cents: dict[int, Any] = {}
    for f in sorted((PROBES_DIR / "fit").glob("fit_L*.json")):
        r = json.loads(f.read_text())
        for t in U.TYPES:
            for c, v in zip(C, r["validity"][t], strict=True):
                validity[(c, int(r["layer"]), t)] = float(v)
        cp = PROBES_DIR / "fit" / f"centroids_L{int(r['layer']):02d}.npy"
        if cp.exists():
            cents[int(r["layer"])] = np.load(cp)
    return validity, cents


def cmd_distractors(a: argparse.Namespace) -> None:
    if (PROBES_DIR / "score").exists():
        sys.exit("eval scores already exist: distractors must be chosen before Phase B")
    items = country_items(a.src_root)
    C = json.loads((OUT / "probe_inputs.json").read_text())["concepts"]
    validity, cents = load_fit()
    L = U.DISTRACTOR_LAYER
    cent = {c: cents[L][i] for i, c in enumerate(C)}
    valid44 = {
        c: all(validity.get((c, L, t), 0) >= U.VALIDITY_MIN for t in U.PRIMARY_TYPES) for c in C
    }
    committed: dict[str, set[str]] = defaultdict(set)
    for r in read_jsonl(OUT / "commit.jsonl"):
        if r["committed"]:
            committed[r["item"]].add(r["concept"])
    out = {}
    for i, it in sorted(items.items()):
        out[i] = U.select_distractors(i, it["gold"], C, cent, valid44, committed[i], seed=6)
    doc = {
        "version": "umdist-v1",
        "layer": L,
        "rule": "4 nearest (grand-mean-centred implied "
        "centroids, cosine) + 4 seeded random among C, valid at L44 in both types, not gold "
        "/ equivalent, not committed for the item",
        "items": out,
    }
    DISTRACTORS_PATH.write_text(json.dumps(doc, indent=1))
    few = [i for i, d in out.items() if len(d["distractors"]) < U.MIN_DISTRACTORS]
    print(f"distractors -> {DISTRACTORS_PATH}; items with < {U.MIN_DISTRACTORS}: {few}")


def load_ztable(C: list[str]) -> Any:
    import numpy as np

    zt = U.ZTable(C)
    for f in sorted((PROBES_DIR / "score").glob("L*.npz")):
        z = np.load(f)
        zt.add_layer(
            int(f.stem[1:]),
            [str(x) for x in z["item"]],
            [int(x) for x in z["pos"]],
            z["za"],
            z["zb"],
        )
    return zt


# ------------------------------------------------------------------ steps 5-6: label + metrics

LABEL_DIR = OUT / "label"


def _rate(k: int, n: int) -> float | None:
    return round(k / n, 4) if n else None


def cmd_label(a: argparse.Namespace) -> None:
    import hashlib

    import numpy as np

    from hallucination_bench.evals.sibling_probe.um_regions import REGION_OF

    items = country_items(a.src_root)
    gold_of = {i: U.gold_concept(it["gold"]) for i, it in items.items()}
    inp = json.loads((OUT / "probe_inputs.json").read_text())
    C = inp["concepts"]
    ssum = json.loads((PROBES_DIR / "score" / "summary.json").read_text())
    sha = hashlib.sha256(DISTRACTORS_PATH.read_bytes()).hexdigest()
    if ssum["distractors_sha256"] != sha:
        sys.exit("distractor file changed after Phase B: refusing to label")
    dist = {
        i: d["distractors"] for i, d in json.loads(DISTRACTORS_PATH.read_text())["items"].items()
    }
    validity, cents = load_fit()
    zt = load_ztable(C)
    pc = U.ProbeContext(zt, validity, dist, gold_of)
    pc_both = U.ProbeContext(zt, validity, dist, gold_of, types=U.TYPES)  # sensitivity row
    cent44 = {c: cents[U.DISTRACTOR_LAYER][i] for i, c in enumerate(C)}

    def cos(x: str, y: str) -> float:
        u, v = cent44[x], cent44[y]
        return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))

    units = {u["uid"]: u for u in load_units()}
    commits = read_jsonl(OUT / "commit.jsonl")
    sidx = U.support_index(commits)
    mentioned = {(u["item"], u["eff_layer"], u["pos"], u["concept"]) for u in units.values()}
    men_item = defaultdict(set)
    com_item = defaultdict(set)
    for r in commits:
        men_item[r["item"]].add(r["concept"])
        if r["committed"]:
            com_item[r["item"]].add(r["concept"])
    prior = U.lens_prior(units.values(), len(items))
    rows = []
    for r in commits:
        u = units[r["uid"]]
        L = u["eff_layer"]
        pr = pc.probe_result(u["item"], L, u["pos"], u["concept"])
        cross = U.cross_support(u, sidx)
        cross_any = U.cross_support(u, sidx, any_layer=True)
        flagged = [
            m
            for m in cross
            if len(prior[(m, u["concept"])] - {u["item"]}) >= 0.5 * (len(items) - 1)
        ]
        lab = U.ladder_label(r["committed"], pr, bool(cross))
        lab_unflagged = U.ladder_label(r["committed"], pr, bool(set(cross) - set(flagged)))
        lab_hedge = U.ladder_label(r["committed_unhedged"], pr, bool(cross))
        pr_both = pc_both.probe_result(u["item"], L, u["pos"], u["concept"])
        lab_both = U.ladder_label(r["committed"], pr_both, bool(cross))
        # recipe sensitivity at L (single type)
        single = {}
        if pr["na_reason"] is None:
            D = [d for d in pc.d_at(u["item"], L) if d != u["concept"]]
            for t in U.TYPES:
                zc = zt.get(u["item"], L, u["pos"], t, u["concept"])
                single[t] = all(zc > zt.get(u["item"], L, u["pos"], t, d) for d in D)
        sim = None
        if u["concept"] in cent44 and dist.get(u["item"]):
            g = gold_of[u["item"]]
            near = json.loads(DISTRACTORS_PATH.read_text())["items"][u["item"]]["nearest"]
            cs = cos(u["concept"], g)
            sim = {
                "cos_to_gold": round(cs, 4),
                "closer_than_all_nearest": all(cs > cos(d, g) for d in near),
            }
        rows.append(
            {
                **{
                    k: r[k]
                    for k in (
                        "uid",
                        "method",
                        "item",
                        "layer",
                        "pos",
                        "concept",
                        "gold",
                        "committed",
                        "committed_unhedged",
                        "fail_closed_only",
                        "via_summary",
                        "quote",
                        "forms",
                    )
                },
                "eff_layer": L,
                "probe": pr,
                "single_type_fire": single,
                "cross": cross,
                "cross_any_layer": cross_any,
                "cross_prior_flagged": flagged,
                "label": lab,
                "label_without_prior_flagged": lab_unflagged,
                "label_hedged_not_committed": lab_hedge,
                "label_both_types": lab_both,
                "probe_both_types": pr_both,
                "in_C": u["concept"] in zt.ci,
                "similarity": sim,
            }
        )
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(LABEL_DIR / "units_labeled.jsonl", rows)

    # ---- nulls
    def null_cells(r: dict[str, Any]) -> dict[str, list[tuple[str, int]]]:
        i, L, S, g = r["item"], r["eff_layer"], r["concept"], r["gold"]
        D = set(pc.d_at(i, L))
        pools: dict[str, list[tuple[str, int]]] = {"i": [], "ii": [], "iii": []}
        for p in zt.positions(i, L):
            if p != r["pos"] and (i, L, p, S) not in mentioned:
                pools["i"].append((i, p))
        for j, gj in gold_of.items():
            if j == i or gj in U.gold_equivalents(S) or S in U.gold_equivalents(gj) or gj in D:
                continue
            if pc.readable_share(j, L) < U.READABLE_SHARE:
                continue
            key = "ii" if gj == g else ("iii" if REGION_OF.get(gj) == REGION_OF.get(g) else None)
            if key is None:
                continue
            for p in zt.positions(j, L):
                if (j, L, p, S) not in mentioned:
                    pools[key].append((j, p))
        return pools

    def fire_vs(i: str, L: int, p: int, S: str, D: list[str]) -> bool:
        return U.beats_all(zt, i, L, p, S, [d for d in D if d != S])

    adj_com = [r for r in rows if r["committed"] and r["probe"]["na_reason"] is None]
    matched = []
    for r in adj_com:
        i, L, S = r["item"], r["eff_layer"], r["concept"]
        D = pc.d_at(i, L)
        pools = null_cells(r)
        res = {k: [fire_vs(j, L, p, S, D) for j, p in v] for k, v in pools.items()}
        union = sorted({(j, p) for v in pools.values() for j, p in v})
        fu = [fire_vs(j, L, p, S, D) for j, p in union]
        matched.append(
            {
                "uid": r["uid"],
                "method": r["method"],
                "item": i,
                "gold": r["gold"],
                "layer": L,
                "concept": S,
                "fire": float(r["probe"]["fire_L"]),
                "matched": float(np.mean(fu)) if fu else None,
                "n_null": len(fu),
                "n_null_beyond_item": len(pools["ii"]) + len(pools["iii"]),
                **{f"matched_{k}": (float(np.mean(v)) if v else None) for k, v in res.items()},
                **{f"n_{k}": len(v) for k, v in res.items()},
            }
        )
    # unmentioned-concept null (overall / hard), at committed adjudicable cells
    um_rows = []
    for r in adj_com:
        i, L, p, g = r["item"], r["eff_layer"], r["pos"], r["gold"]
        D = pc.d_at(i, L)
        cos_g = sorted(cos(c, g) for c in C if c != g)
        q75 = cos_g[int(0.75 * (len(cos_g) - 1))] if cos_g else 1.0
        for c in C:
            if c in D or c in U.gold_equivalents(g) or not pc.valid(c, L):
                continue
            if c in com_item[i]:
                continue
            hard = REGION_OF.get(c) == REGION_OF.get(g) or cos(c, g) >= q75
            never = c not in men_item[i]
            if not (hard or never):
                continue
            um_rows.append(
                {
                    "item": i,
                    "layer": L,
                    "hard": hard,
                    "never": never,
                    "fire": float(fire_vs(i, L, p, c, D)),
                }
            )
    # positive control
    pos_ctrl = {}
    for L in U.PROBE_LAYERS:
        shares = [pc.readable_share(i, L) for i in items if zt.positions(i, L)]
        pos_ctrl[L] = {
            "readable_items": sum(x >= U.READABLE_SHARE for x in shares),
            "n_items": len(shares),
            "mean_cell_gold_on_top": round(float(np.mean(shares)), 4) if shares else None,
        }
    # ---- metrics
    summary = summarize_labels(rows, matched, um_rows, pos_ctrl, validity, C)
    summary["config"] = {
        "commit_prompt": U.COMMIT_PROMPT_VERSION,
        "probe_run": inp["run"],
        "distractors_sha256": sha,
        "validity_min": U.VALIDITY_MIN,
        "readable_share": U.READABLE_SHARE,
        "n_concepts": len(C),
    }
    (LABEL_DIR / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    write_jsonl(LABEL_DIR / "matched_null.jsonl", matched)
    print(json.dumps(summary["headline"], indent=1))


def summarize_labels(
    rows: list[dict[str, Any]],
    matched: list[dict[str, Any]],
    um_rows: list[dict[str, Any]],
    pos_ctrl: dict[int, Any],
    validity: dict[tuple[str, int, str], float],
    C: list[str],
) -> dict[str, Any]:
    import numpy as np

    def hall_rate(
        rs: list[dict[str, Any]], key: str = "label", nearby_ok: bool = True, cluster: str = "item"
    ) -> dict[str, Any]:
        def adj(r: dict[str, Any]) -> bool:
            return r[key] in U.ADJUDICATED

        def hall(r: dict[str, Any]) -> bool:
            return r[key] == U.L_HALL or (not nearby_ok and r[key] == U.L_NEAR)

        return U.bootstrap_rate(
            rs, lambda r: float(adj(r) and hall(r)), lambda r: float(adj(r)), cluster
        )

    out: dict[str, Any] = {
        "labels": {},
        "headline": {},
        "by_layer": {},
        "pairs": [],
        "nulls": {},
        "probe_quality": {},
    }
    for m in U.METHODS:
        rs = [r for r in rows if r["method"] == m]
        com = [r for r in rs if r["committed"]]
        out["labels"][m] = {
            "n_units": len(rs),
            "n_committed": len(com),
            "shares_all": {
                lab: _rate(sum(r["label"] == lab for r in rs), len(rs)) for lab in U.LABELS
            },
            "counts": {lab: sum(r["label"] == lab for r in rs) for lab in U.LABELS},
        }
        nofc = [r for r in rs if not r["fail_closed_only"]]
        out["headline"][m] = {
            "hallucination_rate": hall_rate(rs),
            "by_gold_cluster": hall_rate(rs, cluster="gold"),
            "nearby_not_faithful": hall_rate(rs, nearby_ok=False),
            "fail_closed_excluded": hall_rate(nofc),
            "hedged_not_committed": hall_rate(rs, key="label_hedged_not_committed"),
            "without_prior_flagged_cross": hall_rate(rs, key="label_without_prior_flagged"),
            "both_probe_types": hall_rate(rs, key="label_both_types"),
            "incl_probe_na": U.bootstrap_rate(
                com, lambda r: float(r["label"] in (U.L_HALL, U.L_HALL_NA)), lambda r: 1.0, "item"
            ),
            "faithful_same_share_of_committed": _rate(
                sum(r["label"] == U.L_SAME for r in com), len(com)
            ),
            "faithful_incl_nearby_share_of_committed": _rate(
                sum(r["label"] in (U.L_SAME, U.L_NEAR) for r in com), len(com)
            ),
            "probe_na_reasons": dict(
                Counter(r["probe"]["na_reason"] for r in com if r["probe"]["na_reason"])
            ),
            "via_summary_committed": sum(r["via_summary"] for r in com),
        }
        for L in sorted({r["eff_layer"] for r in rs}):
            rl = [r for r in rs if r["eff_layer"] == L]
            cl = [r for r in rl if r["committed"]]
            out["by_layer"].setdefault(m, {})[L] = {
                "n_units": len(rl),
                "n_committed": len(cl),
                "counts": {lab: sum(r["label"] == lab for r in rl) for lab in U.LABELS},
                "hallucination_rate": hall_rate(rl)["rate"],
                "fire_rate_committed_adjudicable": _rate(
                    sum(r["probe"]["fire_L"] for r in cl if r["probe"]["na_reason"] is None),
                    sum(r["probe"]["na_reason"] is None for r in cl),
                ),
            }
    # pairs
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r["committed"]:
            by_pair[(r["gold"], r["concept"])].append(r)
    mt = {x["uid"]: x for x in matched}
    for (g, c), rs in sorted(by_pair.items(), key=lambda kv: -len(kv[1])):
        adj = [r for r in rs if r["probe"]["na_reason"] is None]
        ex = [
            mt[r["uid"]]["fire"] - mt[r["uid"]]["matched"]
            for r in adj
            if mt.get(r["uid"]) and mt[r["uid"]]["matched"] is not None
        ]
        quotes = {}
        for r in rs:
            quotes.setdefault(r["label"], r["quote"])
        out["pairs"].append(
            {
                "gold": g,
                "sibling": c,
                "n_committed": len(rs),
                "n_items": len({r["item"] for r in rs}),
                "methods": dict(Counter(r["method"] for r in rs)),
                "fire_L": _rate(sum(r["probe"]["fire_L"] for r in adj), len(adj)),
                "fire_incl_nearby": _rate(
                    sum(r["probe"]["fire_L"] or r["probe"]["fire_nearby"] for r in rs), len(rs)
                ),
                "matched_excess": round(float(np.mean(ex)), 4) if ex else None,
                "labels": dict(Counter(r["label"] for r in rs)),
                "quotes": quotes,
            }
        )
    # nulls
    for L in sorted({x["layer"] for x in matched}):
        mL = [x for x in matched if x["layer"] == L and x["matched"] is not None]
        uL = [x for x in um_rows if x["layer"] == L]
        hard = [x for x in uL if x["hard"]]
        out["nulls"][L] = {
            "n_committed_adjudicable": len([x for x in matched if x["layer"] == L]),
            "committed_fire": _rate(int(sum(x["fire"] for x in mL)), len(mL)),
            "matched_null_union": round(float(np.mean([x["matched"] for x in mL])), 4)
            if mL
            else None,
            "excess_unit_weighted": round(float(np.mean([x["fire"] - x["matched"] for x in mL])), 4)
            if mL
            else None,
            **{
                f"matched_{k}": (
                    round(
                        float(
                            np.mean(
                                [x[f"matched_{k}"] for x in mL if x[f"matched_{k}"] is not None]
                            )
                        ),
                        4,
                    )
                    if any(x[f"matched_{k}"] is not None for x in mL)
                    else None
                )
                for k in ("i", "ii", "iii")
            },
            "unmentioned_never": _rate(
                int(sum(x["fire"] for x in uL if x["never"])), sum(x["never"] for x in uL)
            ),
            "unmentioned_hard": _rate(int(sum(x["fire"] for x in hard)), len(hard)),
            "n_hard_cells": len(hard),
            "n_hard_items": len({x["item"] for x in hard}),
            "positive_control": pos_ctrl.get(L),
        }
    ex = [x["fire"] - x["matched"] for x in matched if x["matched"] is not None]
    out["nulls"]["pooled"] = {
        "excess_unit_weighted": round(float(np.mean(ex)), 4) if ex else None,
        "excess_ci_item": U.bootstrap_rate(
            [x for x in matched if x["matched"] is not None],
            lambda x: x["fire"] - x["matched"],
            lambda x: 1.0,
            "item",
        ),
    }
    # probe quality
    for t in U.TYPES:
        vals = [validity.get((c, L, t)) for c in C for L in U.PROBE_LAYERS]
        vals = [v for v in vals if v is not None]
        out["probe_quality"][t] = {
            "median_validity": round(float(np.median(vals)), 4),
            "share_ge_0.90": round(float(np.mean([v >= U.VALIDITY_MIN for v in vals])), 4),
        }
    both28 = [
        all(validity.get((c, L, t), 0) >= U.VALIDITY_MIN for t in U.TYPES)
        for c in C
        for L in U.PROBE_LAYERS
        if L >= 28
    ]
    out["probe_quality"]["both_types_pass_L28_60"] = round(float(np.mean(both28)), 4)
    prim28 = [
        all(validity.get((c, L, t), 0) >= U.VALIDITY_MIN for t in U.PRIMARY_TYPES)
        for c in C
        for L in U.PROBE_LAYERS
        if L >= 28
    ]
    out["probe_quality"]["primary_pass_L28_60"] = round(float(np.mean(prim28)), 4)
    # recipe sensitivity + similarity + not-committed comparison
    adj_c = [r for r in rows if r["committed"] and r["probe"]["na_reason"] is None]
    adj_n = [r for r in rows if not r["committed"] and r["probe"]["na_reason"] is None]
    out["recipe_sensitivity"] = {
        t: _rate(sum(r["single_type_fire"].get(t, False) for r in adj_c), len(adj_c))
        for t in U.TYPES
    } | {"both": _rate(sum(r["probe"]["fire_L"] for r in adj_c), len(adj_c))}
    out["not_committed_fire"] = _rate(sum(r["probe"]["fire_L"] for r in adj_n), len(adj_n))
    close = [r for r in adj_c if r["similarity"] and r["similarity"]["closer_than_all_nearest"]]
    far = [r for r in adj_c if r["similarity"] and not r["similarity"]["closer_than_all_nearest"]]
    out["similarity"] = {
        "closer_than_all_nearest": {
            "n": len(close),
            "fire": _rate(sum(r["probe"]["fire_L"] for r in close), len(close)),
        },
        "not_closer": {
            "n": len(far),
            "fire": _rate(sum(r["probe"]["fire_L"] for r in far), len(far)),
        },
    }
    # gate
    lay = [L for L in U.PROBE_LAYERS if L >= 28]
    hard_ok, not_eval = [], []
    for L in lay:
        n = out["nulls"].get(L, {})
        if (n.get("n_hard_cells") or 0) >= 200 and (n.get("n_hard_items") or 0) >= 10:
            hard_ok.append((n.get("unmentioned_hard") or 0) <= 0.10)
        else:
            not_eval.append(L)
    readable = [pos_ctrl[L]["readable_items"] / max(1, pos_ctrl[L]["n_items"]) for L in lay]
    pairs_beyond = [p for p in out["pairs"]]
    beyond = defaultdict(list)
    for x in matched:
        beyond[(x["gold"], x["concept"])].append(x["n_null_beyond_item"])
    pop = [max(v) >= 20 for v in beyond.values()]
    out["gate"] = {
        "probe_validity_primary_ge_80pct": out["probe_quality"]["primary_pass_L28_60"] >= 0.8,
        "probe_validity_both_ge_80pct": out["probe_quality"]["both_types_pass_L28_60"] >= 0.8,
        "readable_ge_80pct": float(np.mean(readable)) >= 0.8 if readable else False,
        "readable_share_L28_60": round(float(np.mean(readable)), 4) if readable else None,
        "hard_null_le_10pct": all(hard_ok) and len(not_eval) <= 3,
        "hard_null_not_evaluable_layers": not_eval,
        "matched_null_populated_80pct": (float(np.mean(pop)) >= 0.8) if pop else False,
        "n_pairs": len(pairs_beyond),
    }
    return out


# ------------------------------------------------------------------ figures

PLOTS = REPO_ROOT / "plots" / "sibling_probe"
# validated categorical order (dataviz validator, light surface): faithful same, faithful nearby,
# cross-method, hallucination; the probe-n/a variants reuse their hue with a hatch texture
LABEL_STYLE = {
    U.L_NOT: ("#c9c8c2", None),
    U.L_SAME: ("#2a78d6", None),
    U.L_NEAR: ("#4a3aa7", None),
    U.L_CROSS: ("#1baf7a", None),
    U.L_CROSS_NA: ("#1baf7a", "///"),
    U.L_HALL: ("#eb6834", None),
    U.L_HALL_NA: ("#eb6834", "///"),
}


def _style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#52514e",
            "axes.labelcolor": "#0b0b0b",
            "xtick.color": "#52514e",
            "ytick.color": "#52514e",
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
        }
    )


def cmd_figures(a: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    rows = read_jsonl(LABEL_DIR / "units_labeled.jsonl")
    summ = json.loads((LABEL_DIR / "summary.json").read_text())
    PLOTS.mkdir(parents=True, exist_ok=True)
    # (a) label shares of all mentions per method
    fig, ax = plt.subplots(figsize=(13, 5.2))
    names = [U.METHOD_LABEL[m] for m in U.METHODS]
    for yi, m in enumerate(U.METHODS):
        rs = [r for r in rows if r["method"] == m]
        left = 0.0
        for lab in U.LABELS:
            v = sum(r["label"] == lab for r in rs) / max(1, len(rs))
            col, hatch = LABEL_STYLE[lab]
            ax.barh(
                yi,
                v,
                left=left,
                color=col,
                hatch=hatch,
                edgecolor="#fcfcfb",
                linewidth=2,
                height=0.62,
                label=lab if yi == 0 else None,
            )
            if v >= 0.06:
                ax.text(
                    left + v / 2,
                    yi,
                    f"{v:.0%}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="#0b0b0b" if lab == U.L_NOT else "#ffffff",
                )
            left += v
        ax.text(1.01, yi, f"n={len(rs)}", va="center", fontsize=12, color="#52514e")
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("share of sibling-country mention units")
    ax.set_title(
        "Country siblings: what each lens's mentions turn out to be", loc="left", fontsize=17
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS / "um_country_labels.png", dpi=160)
    plt.close(fig)
    # (b) committed units by layer, one panel per method
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.2), sharey=True)
    com_labels = U.LABELS[1:]
    for ax, m in zip(axes, U.METHODS, strict=True):
        rs = [r for r in rows if r["method"] == m and r["committed"]]
        Ls = sorted({r["eff_layer"] for r in rs})
        bottom = [0.0] * len(Ls)
        for lab in com_labels:
            vals = []
            for L in Ls:
                rl = [r for r in rs if r["eff_layer"] == L]
                vals.append(sum(r["label"] == lab for r in rl) / max(1, len(rl)))
            col, hatch = LABEL_STYLE[lab]
            ax.bar(
                [str(L) for L in Ls],
                vals,
                bottom=bottom,
                color=col,
                hatch=hatch,
                edgecolor="#fcfcfb",
                linewidth=2,
                width=0.8,
                label=lab,
            )
            bottom = [b + v for b, v in zip(bottom, vals, strict=True)]
        ax.set_title(f"{U.METHOD_LABEL[m]} (n={len(rs)})", fontsize=14, loc="left")
        ax.set_xlabel("layer")
        ax.tick_params(axis="x", labelsize=10, rotation=90)
    axes[0].set_ylabel("share of committed units")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=6, frameon=False, fontsize=11)
    fig.suptitle("Committed sibling mentions by layer", x=0.01, ha="left", fontsize=17)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(PLOTS / "um_country_layers.png", dpi=160)
    plt.close(fig)
    # (c) fire rate vs nulls by layer
    nl = {int(k): v for k, v in summ["nulls"].items() if k != "pooled"}
    Ls = sorted(nl)
    series = [
        ("committed sibling fires", "committed_fire", "#2a78d6"),
        ("matched null (same sibling, unmentioned cells)", "matched_null_union", "#eb6834"),
        ("hard unmentioned-concept null", "unmentioned_hard", "#1baf7a"),
    ]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for name, key, col in series:
        ys = [nl[L].get(key) for L in Ls]
        xs = [L for L, y in zip(Ls, ys, strict=True) if y is not None]
        ys = [y for y in ys if y is not None]
        ax.plot(xs, ys, color=col, lw=2, marker="o", ms=8, mec="#fcfcfb", mew=2, label=name)
        if xs:
            ax.text(
                xs[-1] + 0.6, ys[-1], name.split(" (")[0], color="#0b0b0b", fontsize=11, va="center"
            )
    rd = [(nl[L]["positive_control"] or {}) for L in Ls]
    ax.plot(
        Ls,
        [r.get("readable_items", 0) / max(1, r.get("n_items", 1)) for r in rd],
        color="#4a3aa7",
        lw=2,
        ls="--",
        label="items readable (gold beats distractors)",
    )
    ax.set_xlabel("layer")
    ax.set_ylabel("rate")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color="#e6e5e0", lw=1)
    ax.set_title(
        "Does the sibling's probe beat the distractors more than chance?", loc="left", fontsize=17
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS / "um_country_nulls.png", dpi=160)
    plt.close(fig)
    print(f"figures -> {PLOTS}/um_country_{{labels,layers,nulls}}.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--src-root", type=Path, default=SRC_DEFAULT)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--rpm", type=float, default=1000)
    s.add_argument("--concurrency", type=int, default=32)
    sub.add_parser("mentions")
    j = sub.add_parser("judge")
    j.add_argument("--ids", type=Path, default=None, help="jsonl with uid fields (a subset)")
    j.add_argument("--dry-run", action="store_true")
    j.add_argument("--rpm", type=float, default=1500)
    j.add_argument("--concurrency", type=int, default=64)
    sub.add_parser("todo")
    sub.add_parser("calib")
    sub.add_parser("concepts")
    sub.add_parser("probe-texts")
    sub.add_parser("capture-rows")
    sub.add_parser("probe-inputs")
    sub.add_parser("distractors")
    sub.add_parser("label")
    sub.add_parser("figures")
    a = ap.parse_args()
    globals()["cmd_" + a.cmd.replace("-", "_")](a)


if __name__ == "__main__":
    main()
