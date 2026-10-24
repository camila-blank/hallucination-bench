"""Plan 0011 step 1: comprehensive lexicons for capital_city / holiday.

Seeded with the Set A inventory (this repo's `data/sibling_probe/concepts.json`:
concept ids, aliases, merges, exclusion phrases); `claude-opus-5-5` (prompt `umlex-v1`, cached in
`outputs/sibling/um/lexicon_cache.jsonl`) adds the rest of the domain. Code then enforces:
inventory concepts and aliases kept verbatim; no alias shared by two concepts (the inventory wins,
otherwise the first concept); CJK aliases >= 2 characters; exclusion phrases -> block list.
Output `data/sibling_probe/um_lexicons.json` in the plan 0005 lexicon layout
(`{"categories": {cat: {"concepts": {id: {label, en, cjk, tokens}}, "block": [...]}}}`), plus
`capitalised_only` per category.

    uv run --no-sync python scripts/sibling_probe/um_lexicons.py [--categories a,b]
"""

import argparse
import asyncio
import json
import re
from typing import Any

from hallucination_bench import DATA, REPO_ROOT
from hallucination_bench.cache import Cache, fingerprint
from hallucination_bench.llm import LLM, schema_block

SRC = (
    DATA / "sibling_probe" / "concepts.json"
)  # tracked; was read from the hb-sibling-probes checkout
OUT = REPO_ROOT / "data" / "sibling_probe" / "um_lexicons.json"
CACHE = REPO_ROOT / "outputs" / "sibling" / "um" / "lexicon_cache.jsonl"
MODEL = "claude-opus-5-5"
PROMPT_VERSION = "umlex-v1"
CATEGORIES = ("capital_city", "holiday")
SPEC = {
    "capital_city": dict(
        noun="city a traveller might visit (world capitals and major tourist cities)",
        n="about 150",
        extra="Cities only, never countries or regions. Include the English "
        "name, common alternative names (Peking -> Beijing is a separate "
        "alias), and the Chinese name.",
    ),
    "holiday": dict(
        noun="holiday, festival or celebration day",
        n="about 60",
        extra="Religious, national, cultural and commercial holidays worldwide "
        "(Christmas, Eid al-Fitr, Hanukkah, Diwali, Songkran, Mother's Day, ...). "
        "Not generic words (holiday, festival, birthday, vacation).",
    ),
}
SYSTEM = (
    "You build lexicons for an NLP dataset. Follow the constraints exactly and return JSON only."
)
SCHEMA = schema_block(
    "lex",
    {
        "concepts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "en": {"type": "array", "items": {"type": "string"}},
                    "cjk": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "label", "en", "cjk"],
                "additionalProperties": False,
            },
        }
    },
    ["concepts"],
)
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")


def prompt(cat: str, inv: list[dict[str, Any]]) -> str:
    s = SPEC[cat]
    seed = "\n".join(f"- {c['id']}: {', '.join(c['aliases'])}" for c in inv)
    return (
        f"List {s['n']} distinct instances of: {s['noun']}.\n{s['extra']}\n\n"
        "For each concept give: id (lower_snake_case), label, en (every English form a text "
        "might use to name it: singular, plural, adjective/demonym or noun forms, common "
        "variants), cjk (Simplified Chinese names, each at least 2 characters).\n"
        "These concepts MUST be included with EXACTLY these ids (add more aliases if useful, "
        "never remove one):\n" + seed + '\n\nReturn {"concepts": [...]}.'
    )


async def generate(cats: list[str], inv: dict[str, Any]) -> dict[str, Any]:
    cache = Cache(CACHE)
    llm: LLM | None = None
    out = {}
    for cat in cats:
        user = prompt(cat, inv[cat]["concepts"])
        fp = fingerprint(PROMPT_VERSION, MODEL, "low", SYSTEM, user)
        row = cache.get(cat, fp)
        if row is None:
            llm = llm or LLM(rpm=100, concurrency=4, max_tokens=32000)
            res = await llm.call(model=MODEL, system=SYSTEM, user=user, schema=SCHEMA)
            cache.put(cat, fp, {"result": res.result, "meta": {"usage": res.usage.to_dict()}})
            print(f"[lex] {cat}: {res.usage.report()}", flush=True)
            row = {"result": res.result}
        out[cat] = row["result"]
    if llm:
        await llm.aclose()
    cache.close()
    return out


def build(cat: str, gen: dict[str, Any], inv: dict[str, Any]) -> tuple[dict[str, Any], dict]:
    concepts: dict[str, dict[str, Any]] = {}
    owner: dict[str, str] = {}
    checks = {"dropped_shared_alias": [], "dropped_short_cjk": [], "dropped_empty": []}

    def add(cid: str, label: str, en: list[str], cjk: list[str], seed: bool) -> None:
        c = concepts.setdefault(cid, {"label": label, "en": [], "cjk": [], "tokens": []})
        for key, al in (("en", en), ("cjk", cjk)):
            for a in al:
                a = " ".join(str(a).split())
                if not a:
                    continue
                if _CJK.search(a) and len(a) < 2:
                    checks["dropped_short_cjk"].append((cid, a))
                    continue
                k = a.casefold()
                if k in owner and owner[k] != cid:
                    if not seed:
                        checks["dropped_shared_alias"].append((cid, a, owner[k]))
                    continue
                owner[k] = cid
                if a not in c[key]:
                    c[key].append(a)

    for c in inv["concepts"]:  # the inventory first: it wins every alias clash
        en = [a for a in c["aliases"] if not _CJK.search(a)]
        cjk = [a for a in c["aliases"] if _CJK.search(a)]
        add(c["id"], c["label"], en, cjk, True)
    for c in (gen or {}).get("concepts", []):
        cid = re.sub(r"[^a-z0-9_]", "_", str(c["id"]).lower()).strip("_")
        if cid:
            add(cid, c["label"], c.get("en", []), c.get("cjk", []), False)
    for cid in [k for k, v in concepts.items() if not v["en"] and not v["cjk"]]:
        checks["dropped_empty"].append(cid)
        concepts.pop(cid)
    block = sorted(
        {p for c in inv["concepts"] for v in (c.get("exclude") or {}).values() for p in v}
        | BLOCK_EXTRA.get(cat, set())
    )
    return {"concepts": concepts, "block": block}, checks


# generic / stock phrases that contain an instance word but name no instance
BLOCK_EXTRA: dict[str, set[str]] = {
    "holiday": {"holiday season", "the holidays", "happy holidays"},
    "capital_city": {"George Washington", "Washington State", "Paris Hilton"},
}
CAPITALISED_ONLY = {"capital_city": "all", "holiday": "all"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", default=",".join(CATEGORIES))
    a = ap.parse_args()
    cats = a.categories.split(",")
    inv = json.loads(SRC.read_text())["categories"]
    gen = asyncio.run(generate(cats, inv))
    doc = (
        json.loads(OUT.read_text())
        if OUT.exists()
        else {
            "version": PROMPT_VERSION,
            "provenance": f"{MODEL}, seeded with {SRC.relative_to(REPO_ROOT)}",
            "categories": {},
            "capitalised_only": {},
            "checks": {},
        }
    )
    for cat in cats:
        lex, checks = build(cat, gen[cat], inv[cat])
        doc["categories"][cat] = lex
        doc["capitalised_only"][cat] = CAPITALISED_ONLY[cat]
        doc["checks"][cat] = {k: len(v) for k, v in checks.items()} | {
            "examples": {k: v[:8] for k, v in checks.items()}
        }
        print(
            f"{cat}: {len(lex['concepts'])} concepts, {len(lex['block'])} block phrases, "
            f"checks {doc['checks'][cat]}"
        )
    OUT.write_text(json.dumps(doc, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
