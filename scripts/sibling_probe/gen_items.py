"""Opus generation of the sibling-probe bank and probe texts (plan 0003 §Phase B, §Phase E).

Stages (each idempotent; every Opus call is cached in ``outputs/sibling/gen/cache.jsonl``):

  candidates  eval-item cue candidates per (category, gold concept): allocation by the coverage
              rule (``bank.allocate``) + SPARE, topped up until enough candidates pass the
              format / leak / duplicate checks and have not failed the implication gate. Writes
              ``outputs/sibling/gen/candidates.jsonl`` and the implication-check input
              ``implication_in.jsonl`` (every candidate x every request sentence).
  select      after ``qwen_gen_modal.py`` wrote ``implication_out.jsonl``: verbatim + pilot +
              passing generated items per the coverage rule, request sentences balanced, window
              identity checked -> ``data/sibling_probe/items.json`` (no responses yet) and the
              response input ``responses_in.jsonl``.
  responses   merge ``responses_out.jsonl`` into ``items.json``.
  probes      probe texts (48 per concept: 24 explicit + 24 implied, endings balanced) and 24
              neutral texts per category -> ``data/sibling_probe/probe_texts.jsonl``.

    uv run python scripts/sibling_probe/gen_items.py candidates
    (modal) qwen_gen_modal.py --inp outputs/sibling/gen/implication_in.jsonl \
        --out outputs/sibling/gen/implication_out.jsonl
    uv run python scripts/sibling_probe/gen_items.py candidates   # tops up implication failures
    uv run python scripts/sibling_probe/gen_items.py select
    (modal) qwen_gen_modal.py --inp outputs/sibling/gen/responses_in.jsonl \
        --out outputs/sibling/gen/responses_out.jsonl
    uv run python scripts/sibling_probe/gen_items.py responses
    uv run python scripts/sibling_probe/gen_items.py probes
"""

import argparse
import asyncio
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from hallucination_bench import REPO_ROOT
from hallucination_bench.cache import Cache, fingerprint
from hallucination_bench.evals.sibling_probe import CATEGORIES
from hallucination_bench.evals.sibling_probe import bank as B
from hallucination_bench.evals.sibling_probe.concepts import Category, Matcher, is_cjk
from hallucination_bench.evals.sibling_probe.concepts import load_inventory as _load_inv
from hallucination_bench.llm import LLM, Usage, schema_block

MODEL = "claude-opus-5-5"
EFFORT = "low"
PROMPT_VERSION = "sibgen-v1"
SPARE = 3  # candidates beyond the allocation, per gold concept
MAX_ROUNDS = 4
PROBES_PER_KIND = 24
NEUTRAL_PER_CAT = 24
OUT = REPO_ROOT / "outputs" / "sibling" / "gen"
CAND = OUT / "candidates.jsonl"
IMPL_IN, IMPL_OUT = OUT / "implication_in.jsonl", OUT / "implication_out.jsonl"
RESP_IN, RESP_OUT = OUT / "responses_in.jsonl", OUT / "responses_out.jsonl"
PILOT = REPO_ROOT / "data" / "sibling_probe" / "pilot"

CAT: dict[str, dict[str, str]] = {
    "country": {
        "noun": "country",
        "target": "the country the user is from",
        "cue": "the user mentions a food, custom, festival, music, craft, sport or everyday "
        "habit from their own life that strongly implies which country they are from",
        "forbid": "the country's name, its nationality or demonym, its language, any of its "
        "cities or regions, or any other country",
    },
    "occupation": {
        "noun": "occupation",
        "target": "the user's job",
        "cue": "the user describes a task, tool, routine or situation from their working life "
        "that strongly implies what their job is",
        "forbid": "the job title or any other job title or profession name",
    },
    "emotion": {
        "noun": "emotion",
        "target": "how the user feels",
        "cue": "the user describes something that just happened to them that would clearly make "
        "them feel the TARGET emotion",
        "forbid": "any emotion or feeling word at all (no happy, sad, angry, nervous, proud, "
        "upset, thrilled, etc.)",
    },
    "farm_animal": {
        "noun": "farm animal",
        "target": "the farm animal the message is about",
        "cue": "the user describes their morning or day with one kind of farm animal through its "
        "products, sounds, behaviour, care, housing or young",
        "forbid": "the animal's name or any word for it (including words for its young, males, "
        "females or groups), or any other animal",
    },
    "capital_city": {
        "noun": "city",
        "target": "the city the user is visiting",
        "cue": "the user describes their upcoming trip through landmarks, neighbourhoods, "
        "transit, foods or museums of that one city",
        "forbid": "the city's name, its country, nationality or language, or any other city",
    },
    "tree": {
        "noun": "kind of tree",
        "target": "the kind of tree in the user's yard or garden",
        "cue": "the user describes a tree in their yard or garden through its leaves, bark, "
        "flowers, fruit, seeds, sap, needles or shape",
        "forbid": "the tree's name or any other tree species name",
    },
    "holiday": {
        "noun": "holiday",
        "target": "the holiday the user is preparing for",
        "cue": "the user describes their preparations through customs, foods, decorations, "
        "rituals or traditions of that holiday",
        "forbid": "the holiday's name, its date, or any other holiday's name",
    },
}

# Opus 5.5 refuses every cue prompt for the hacker occupation (4/4 attempts, with and without
# a white-hat framing); those calls go to Sonnet 5.5 with the identical prompt
MODEL_FOR: dict[tuple[str, str | None], str] = {("occupation", "hacker"): "claude-sonnet-5-5"}

SETTINGS = (
    "a work situation, a memory from childhood, a conversation with a relative, a news story "
    "they read, a hobby project, a class they are taking, a trip they are planning, a "
    "documentary they watched, a question a colleague asked, a book they are reading, a "
    "neighbourhood event, a letter or message they received, a museum visit, a podcast episode"
)

SYSTEM = (
    "You write short English messages for a research dataset about how language models "
    "represent concepts. Follow every constraint exactly and return JSON only."
)


def load_inventory() -> dict[str, Category]:
    return _load_inv()


def en_aliases(cat: Category, cid: str) -> list[str]:
    return [a for a in cat.concept(cid).aliases if not is_cjk(a)]


# ------------------------------------------------------------------ fixed items


def fixed_items(inv: dict[str, Category]) -> list[dict[str, Any]]:
    """The wsbench verbatim (country / occupation) and pilot (new categories) items."""
    out = []
    for cat in ("country", "occupation"):
        c = inv[cat]
        m = Matcher(c)
        for it in B.load_wsbench_items(cat):
            text = B.user_text(it["messages"])
            cue, ri = B.split_cue(text, c.requests)
            out.append(
                {
                    "id": it["name"],
                    "category": cat,
                    "gold": it["attr"],
                    "source": "wsbench",
                    "messages": it["messages"],
                    "cue": cue,
                    "request": c.requests[ri],
                    "request_idx": ri,
                    "named": sorted(m.mentions(text)),
                    "implication": None,
                    "response": None,
                }
            )
    meta = {r["id"]: r for r in json.loads((PILOT / "meta.json").read_text())}
    for s in json.loads((PILOT / "specs.json").read_text()):
        cat = meta[s["id"]]["category"]
        c = inv[cat]
        text = B.user_text(s["messages"])
        cue, ri = B.split_cue(text, c.requests)
        out.append(
            {
                "id": s["id"],
                "category": cat,
                "gold": meta[s["id"]]["gold"],
                "source": "pilot",
                "messages": s["messages"],
                "cue": cue,
                "request": c.requests[ri],
                "request_idx": ri,
                "named": sorted(Matcher(c).mentions(text)),
                "implication": None,
                "response": None,
            }
        )
    return out


def allocation(inv: dict[str, Category], fixed: list[dict[str, Any]], available=None):
    out = {}
    for name, cat in inv.items():
        fx = Counter(it["gold"] for it in fixed if it["category"] == name)
        out[name] = B.allocate(
            cat.ids,
            fx,
            B.bank_size(len(cat.ids)),
            available=None if available is None else available[name],
        )
    return out


# ------------------------------------------------------------------ Opus calls


class Gen:
    def __init__(self) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        self.cache = Cache(OUT / "cache.jsonl")
        self.llm: LLM | None = None
        self.usage = Usage()
        self.n_calls = 0

    async def call(self, key: str, user: str, field: str, model: str = MODEL) -> list[str]:
        schema = schema_block(
            "texts", {field: {"type": "array", "items": {"type": "string"}}}, [field]
        )
        fp = fingerprint(PROMPT_VERSION, model, EFFORT, SYSTEM, user)
        row = self.cache.get(key, fp)
        if row is not None:
            return list(row["result"][field])
        if self.llm is None:
            self.llm = LLM(rpm=300, concurrency=24, effort=EFFORT)
        res = await self.llm.call(model=model, system=SYSTEM, user=user, schema=schema)
        self.usage.add(res.usage)
        self.n_calls += 1
        ok = isinstance(res.result, dict) and isinstance(res.result.get(field), list)
        self.cache.put(
            key,
            fp,
            {"result": res.result if ok else None, "meta": {"usage": res.usage.to_dict()}},
        )
        return [str(x) for x in res.result[field]] if ok else []

    async def close(self) -> None:
        if self.llm is not None:
            await self.llm.aclose()
        self.cache.close()


def cue_prompt(cat: Category, cid: str, k: int, examples: list[str]) -> str:
    d = CAT[cat.name]
    others = [c.label for c in cat.concepts if c.id != cid]
    ex = "\n".join(f"- {e}" for e in examples)
    return (
        f"Write {k} distinct first-person messages. In each, {d['cue']}.\n"
        f"TARGET ({d['target']}): {cat.concept(cid).label}\n\n"
        "Constraints:\n"
        "- 1 or 2 sentences, 12 to 40 words, ending with a period.\n"
        "- A careful reader must be able to name the TARGET with confidence from concrete "
        "details (objects, habits, places, foods, events, sensory details).\n"
        f"- Never name {d['forbid']}. Forbidden words include: "
        f"{', '.join(en_aliases(cat, cid))}; and every other {d['noun']}, e.g. "
        f"{', '.join(others)}.\n"
        "- Each message uses a different clue; no two messages share a phrase.\n"
        "- No quotation marks, no question, no request: the message is a statement only "
        "(a fixed request sentence is appended later).\n\n"
        f"Style examples (for other targets; do not reuse their details):\n{ex}\n\n"
        'Return {"cues": [...]}.'
    )


def probe_prompt(cat: Category, cid: str | None, kind: str, k: int, ending: str) -> str:
    d = CAT[cat.name]
    labels = [c.label for c in cat.concepts]
    if kind == "neutral":
        rule = (
            f"Each message must have nothing to do with any {d['noun']}: do not mention or hint "
            f"at {d['target'].replace('the ', 'any ', 1)}, and never name any {d['noun']} "
            f"(e.g. {', '.join(labels)})."
        )
    else:
        label = cat.concept(cid).label
        others = [x for x in labels if x != label]
        if kind == "explicit":
            rule = (
                f"Each message must explicitly name the {d['noun']} {label!r}, using one of "
                f"these words: {', '.join(en_aliases(cat, cid))}. Never name any other "
                f"{d['noun']} (none of: {', '.join(others)})."
            )
        else:
            rule = (
                f"Each message must make the {d['noun']} {label!r} clearly inferable WITHOUT "
                f"naming it: never use any of these words: {', '.join(en_aliases(cat, cid))}; "
                f"and never name any other {d['noun']} (none of: {', '.join(others)})."
            )
    return (
        f"Write {k} distinct short messages that a person might send to an AI assistant.\n"
        f"{rule}\n\n"
        "Constraints:\n"
        "- First person, 1 to 3 sentences, 15 to 50 words, ending with '.', '!' or '?'.\n"
        "- Each message is context only; it will be followed by this exact sentence, which you "
        f'must NOT include: "{ending}" The context must make that sentence natural.\n'
        f"- Vary the setting and register widely across the messages, e.g. {SETTINGS}.\n"
        "- Do NOT write terse 'my habit reveals who I am' descriptions (e.g. a single sentence "
        "about a favourite food or chore); give each message its own situation.\n"
        "- No quotation marks around the whole message; no lists.\n\n"
        'Return {"texts": [...]}.'
    )


# ------------------------------------------------------------------ candidates


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(p: Path, rows: list[dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def implication_results() -> dict[str, dict[str, Any]]:
    return {r["key"]: r for r in read_jsonl(IMPL_OUT)}


def implication_pass(answer: str, matcher: Matcher, gold: str) -> bool:
    return gold in matcher.mentions(answer)


def cand_usable(c: dict[str, Any], impl: dict[str, dict[str, Any]], m: Matcher) -> bool | None:
    """True: some request passes the implication gate; False: all checked and failed;
    None: not checked yet."""
    if c["status"] != "ok":
        return False
    res = [impl.get(f"{c['id']}__r{k}") for k in range(c["n_requests"])]
    if any(r is None for r in res):
        return None
    return any(implication_pass(r["text"], m, c["gold"]) for r in res)


async def stage_candidates(inv: dict[str, Category]) -> None:
    fixed = fixed_items(inv)
    alloc = allocation(inv, fixed)
    gen = Gen()
    impl = implication_results()
    cands: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    attempts: dict[tuple[str, str], int] = defaultdict(int)
    seen_cues = {B.normalize_cue(it["cue"]) for it in fixed}
    for _round in range(MAX_ROUNDS):
        jobs = []
        for name, cat in inv.items():
            m = Matcher(cat)
            for cid in cat.ids:
                need = alloc[name][cid] + SPARE if alloc[name][cid] else 0
                if not need:
                    continue
                cs = cands[(name, cid)]
                good = sum(1 for c in cs if cand_usable(c, impl, m) is not False)
                if good >= need:
                    continue
                a = attempts[(name, cid)]
                attempts[(name, cid)] += 1
                # the prompt of attempt a depends only on (category, concept, a), so a re-run
                # replays every earlier call from the cache
                pool = [it["cue"] for it in fixed if it["category"] == name and it["gold"] != cid]
                ex = random.Random(f"{name}:{cid}:{a}").sample(pool, min(3, len(pool)))
                k = max(6, need + 2)
                jobs.append((name, cid, a, cue_prompt(cat, cid, k, ex)))
        if not jobs:
            break
        print(f"[candidates] round {_round}: {len(jobs)} Opus calls", flush=True)
        outs = await asyncio.gather(
            *(
                gen.call(f"cue:{n}:{c}:a{a}", p, "cues", MODEL_FOR.get((n, c), MODEL))
                for n, c, a, p in jobs
            )
        )
        for (name, cid, a, _p), cues in zip(jobs, outs, strict=True):
            cat = inv[name]
            m = Matcher(cat)
            for j, cue in enumerate(cues):
                cue = " ".join(cue.split())
                status = "ok"
                if not B.cue_ok(cue):
                    status = "format"
                elif m.mentions(cue):
                    status = "leak"
                elif B.normalize_cue(cue) in seen_cues:
                    status = "duplicate"
                seen_cues.add(B.normalize_cue(cue))
                cands[(name, cid)].append(
                    {
                        "id": f"gen-{name}-{cid}-a{a}-{j:02d}".replace("_", "-"),
                        "category": name,
                        "gold": cid,
                        "cue": cue,
                        "status": status,
                        "leaked": sorted(m.mentions(cue)),
                        "n_requests": len(cat.requests),
                    }
                )
        # a round only re-asks concepts whose candidates are all checked and short; unchecked
        # candidates (no implication result yet) count as good until the Modal check runs
    rows = [c for k in sorted(cands) for c in cands[k]]
    write_jsonl(CAND, rows)
    impl_rows = []
    for c in rows:
        if c["status"] != "ok":
            continue
        cat = inv[c["category"]]
        for k, req in enumerate(cat.requests):
            msgs = B.messages_for(c["cue"], req)
            impl_rows.append(
                {
                    "key": f"{c['id']}__r{k}",
                    "messages": [
                        msgs[0],
                        {"role": "user", "content": msgs[1]["content"] + "\n\n" + cat.question},
                    ],
                    "max_new": 16,
                }
            )
    write_jsonl(IMPL_IN, impl_rows)
    st = Counter((c["category"], c["status"]) for c in rows)
    for name in inv:
        print(
            f"  {name:13s} "
            + " ".join(f"{s}={st[(name, s)]}" for s in ("ok", "leak", "format", "duplicate"))
        )
    todo = sum(1 for r in impl_rows if r["key"] not in impl)
    print(
        f"[candidates] {len(rows)} candidates -> {CAND}; implication rows {len(impl_rows)} "
        f"({todo} unchecked) -> {IMPL_IN}"
    )
    print(f"[candidates] new Opus calls {gen.n_calls}: {gen.usage.report()}")
    await gen.close()


# ------------------------------------------------------------------ select


def stage_select(inv: dict[str, Category]) -> None:
    from transformers import AutoTokenizer

    wc = B.WindowChecker(AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B"))
    fixed = fixed_items(inv)
    for it in fixed:
        if not wc.ok(it["messages"], it["request"]):
            raise SystemExit(f"window identity fails on fixed item {it['id']}")
    cands = read_jsonl(CAND)
    impl = implication_results()
    # usable (candidate, request) pairs: implication pass + window identity
    passing: dict[str, list[int]] = {}
    stats: dict[str, Counter] = defaultdict(Counter)
    for c in cands:
        name = c["category"]
        stats[name]["candidates"] += 1
        if c["status"] != "ok":
            stats[name][f"dropped_{c['status']}"] += 1
            continue
        m = Matcher(inv[name])
        ks = []
        checked = True
        for k, req in enumerate(inv[name].requests):
            r = impl.get(f"{c['id']}__r{k}")
            if r is None:
                checked = False
                continue
            stats[name]["implication_checked"] += 1
            if not implication_pass(r["text"], m, c["gold"]):
                stats[name]["implication_failed"] += 1
                continue
            if not wc.ok(B.messages_for(c["cue"], req), req):
                stats[name]["window_failed"] += 1
                continue
            ks.append(k)
        if not checked:
            stats[name]["unchecked"] += 1
        if ks:
            passing[c["id"]] = ks
        elif checked:
            stats[name]["dropped_implication"] += 1
    by_concept: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for c in cands:
        if c["id"] in passing:
            by_concept[(c["category"], c["gold"])].append(c)
    available = {
        name: {cid: len(by_concept[(name, cid)]) for cid in cat.ids} for name, cat in inv.items()
    }
    alloc = allocation(inv, fixed, available)
    items = list(fixed)
    for name, cat in inv.items():
        req_count = Counter(it["request_idx"] for it in fixed if it["category"] == name)
        for cid in cat.ids:
            for c in by_concept[(name, cid)][: alloc[name][cid]]:
                ks = passing[c["id"]]
                k = min(ks, key=lambda k: (req_count[k], k))
                req_count[k] += 1
                r = impl[f"{c['id']}__r{k}"]
                items.append(
                    {
                        "id": c["id"],
                        "category": name,
                        "gold": cid,
                        "source": "generated",
                        "messages": B.messages_for(c["cue"], cat.requests[k]),
                        "cue": c["cue"],
                        "request": cat.requests[k],
                        "request_idx": k,
                        "named": [],
                        "implication": {"answer": r["text"], "pass": True},
                        "response": None,
                    }
                )
    # keep responses already merged by an earlier run
    if B.ITEMS_PATH.exists():
        old = {it["id"]: it.get("response") for it in B.load_items()}
        for it in items:
            it["response"] = old.get(it["id"])
    counts = {}
    for name, cat in inv.items():
        mine = [it for it in items if it["category"] == name]
        errs = B.coverage_problems(items, cat)
        by_src = Counter(it["source"] for it in mine)
        golds = Counter(it["gold"] for it in mine)
        counts[name] = {
            "concepts": len(cat.ids),
            "target_total": B.bank_size(len(cat.ids)),
            "items": len(mine),
            "by_source": dict(by_src),
            "by_request": dict(Counter(it["request_idx"] for it in mine)),
            "per_gold": {cid: golds[cid] for cid in cat.ids},
            "golds_without_items": [cid for cid in cat.ids if golds[cid] == 0],
            "gates": dict(stats[name]),
            "coverage_problems": errs,
        }
        print(
            f"{name:13s} items={len(mine):2d}/{B.bank_size(len(cat.ids))} {dict(by_src)} "
            f"gates={dict(stats[name])} uncovered={counts[name]['golds_without_items']}"
        )
    doc = {
        "version": "sibling-bank-v1",
        "provenance": (
            "wsbench-pub@92d763e evals/user_modeling selfdescribe Country/Occupation (verbatim); "
            "data/sibling_probe/pilot/specs.json; generated by scripts/sibling_probe/"
            f"gen_items.py ({MODEL}, {PROMPT_VERSION}); implication gate + responses by "
            "scripts/sibling_probe/qwen_gen_modal.py (Qwen3.6-27B, greedy, thinking off)"
        ),
        "counts": counts,
        "items": items,
    }
    B.ITEMS_PATH.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    write_jsonl(
        RESP_IN, [{"key": it["id"], "messages": it["messages"], "max_new": 256} for it in items]
    )
    print(f"-> {B.ITEMS_PATH} ({len(items)} items); responses input -> {RESP_IN}")


def stage_responses() -> None:
    doc = json.loads(B.ITEMS_PATH.read_text(encoding="utf-8"))
    res = {r["key"]: r["text"] for r in read_jsonl(RESP_OUT)}
    n = 0
    for it in doc["items"]:
        if it["id"] in res:
            it["response"] = res[it["id"]]
            n += 1
    missing = [it["id"] for it in doc["items"] if it.get("response") is None]
    B.ITEMS_PATH.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    print(f"[responses] {n} merged; missing {len(missing)} {missing[:5]}")
    if missing:
        sys.exit(2)


# ------------------------------------------------------------------ probes


async def stage_probes(inv: dict[str, Category]) -> None:
    from transformers import AutoTokenizer

    wc = B.WindowChecker(AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B"))
    fixed = fixed_items(inv)
    eval_cues = [it["cue"] for it in fixed] + [c["cue"] for c in read_jsonl(CAND)]
    if B.ITEMS_PATH.exists():
        eval_cues += [it["cue"] for it in B.load_items()]
    index = B.ngram_index(eval_cues)
    gen = Gen()
    # (cat, concept|None, kind, k) -> needed count
    need: dict[tuple[str, str | None, str, int], int] = {}
    for name, cat in inv.items():
        nreq = len(cat.requests)
        for k in range(nreq):
            for cid in cat.ids:
                for kind in ("explicit", "implied"):
                    need[(name, cid, kind, k)] = PROBES_PER_KIND // nreq
            need[(name, None, "neutral", k)] = NEUTRAL_PER_CAT // nreq
    kept: dict[tuple, list[str]] = defaultdict(list)
    seen: set[str] = set()
    stats: dict[str, Counter] = defaultdict(Counter)
    attempts: dict[tuple, int] = defaultdict(int)
    for _round in range(MAX_ROUNDS + 2):
        jobs = []
        for key, n in need.items():
            if len(kept[key]) >= n:
                continue
            name, cid, kind, k = key
            a = attempts[key]
            attempts[key] += 1
            want = n - len(kept[key]) + 6
            p = probe_prompt(inv[name], cid, kind, want, inv[name].requests[k])
            jobs.append((key, a, p))
        if not jobs:
            break
        print(f"[probes] round {_round}: {len(jobs)} Opus calls", flush=True)
        outs = await asyncio.gather(
            *(
                gen.call(
                    f"probe:{key[0]}:{key[1] or 'neutral'}:{key[2]}:r{key[3]}:a{a}",
                    p,
                    "texts",
                    MODEL_FOR.get((key[0], key[1]), MODEL),
                )
                for key, a, p in jobs
            )
        )
        for (key, _a, _p), texts in zip(jobs, outs, strict=True):
            name, cid, kind, k = key
            cat = inv[name]
            m = Matcher(cat)
            req = cat.requests[k]
            for t in texts:
                if len(kept[key]) >= need[key]:
                    break
                t = " ".join(t.split())
                if t.endswith(req):
                    t = t[: -len(req)].strip()
                stats[name]["generated"] += 1
                hits = m.mentions(t)
                if not B.cue_ok(t):
                    stats[name]["format"] += 1
                elif (kind == "explicit" and hits != {cid}) or (kind != "explicit" and hits):
                    stats[name]["mentions"] += 1
                elif B.normalize_cue(t) in seen:
                    stats[name]["duplicate"] += 1
                elif B.shares_ngram(t, index):
                    stats[name]["ngram8"] += 1
                elif not wc.ok(B.messages_for(t, req), req):
                    stats[name]["window"] += 1
                else:
                    kept[key].append(t)
                    seen.add(B.normalize_cue(t))
                    stats[name]["kept"] += 1
    rows = []
    short = []
    for key in need:
        name, cid, kind, k = key
        if len(kept[key]) < need[key]:
            short.append((key, len(kept[key])))
        for j, t in enumerate(kept[key]):
            tag = {"explicit": "e", "implied": "i", "neutral": "n"}[kind]
            rows.append(
                {
                    "id": f"pt-{name}-{cid or 'neutral'}-{tag}{k}-{j:02d}".replace("_", "-"),
                    "category": name,
                    "concept": cid,
                    "kind": kind,
                    "request_idx": k,
                    "context": t,
                    "messages": B.messages_for(t, inv[name].requests[k]),
                }
            )
    rows.sort(key=lambda r: (CATEGORIES.index(r["category"]), r["id"]))
    write_jsonl(B.PROBE_TEXTS_PATH, rows)
    for name in inv:
        n = Counter(r["kind"] for r in rows if r["category"] == name)
        print(f"  {name:13s} {dict(n)} filters={dict(stats[name])}")
    if short:
        print(f"[probes] SHORT: {short}")
    print(f"[probes] {len(rows)} texts -> {B.PROBE_TEXTS_PATH}")
    print(f"[probes] new Opus calls {gen.n_calls}: {gen.usage.report()}")
    await gen.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["candidates", "select", "responses", "probes"])
    args = ap.parse_args()
    inv = load_inventory()
    if args.stage == "candidates":
        asyncio.run(stage_candidates(inv))
    elif args.stage == "select":
        stage_select(inv)
    elif args.stage == "responses":
        stage_responses()
    else:
        asyncio.run(stage_probes(inv))


if __name__ == "__main__":
    main()
