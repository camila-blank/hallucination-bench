# ruff: noqa: E501
"""Plan 0005: basic-readout mention ladder (commitment judge -> probe -> cross-method -> label).

Stages (idempotent; Opus calls cached per prompt version under ``outputs/sibling/ladder/``):

  lexicons    brlex-v1: per-category lexicons (Opus) + code checks -> data/sibling_probe/br_lexicons.json
  mentions    mention units for the 35 entity items x 4 methods x layers -> outputs/sibling/ladder/mentions.jsonl
  judge       brcommit-v1 on every (unit, sample) hit -> outputs/sibling/ladder/commit.jsonl (cache_brcommit-v1.jsonl)
  todo        brcommit dev todo (40 units, stratified) + gold todo (80 units + all Ali units)
  calib       judge vs gold (data/sibling_probe/brcommit_gold.jsonl): accuracy, Wilson CI, per-class P/R
  banks       brbank-v1: entity banks (Opus) -> data/sibling_probe/br_entity_banks.json
  capture-rows rows for the Modal capture (bank entities + the 35 items) -> outputs/sibling/ladder/capture_rows.json
  probes      (after ladder_capture_modal.py) knowledge filter, counts, probes, fire rule -> outputs/sibling/ladder/probes.json
  label       ladder labels -> outputs/sibling/ladder/{units.jsonl,summary.json,report.md}
  figures     plots/sibling_probe/ladder_*.png

    uv run python scripts/sibling_probe/ladder.py lexicons
    uv run python scripts/sibling_probe/ladder.py mentions
    uv run python scripts/sibling_probe/ladder.py judge
    uv run python scripts/sibling_probe/ladder.py todo
    uv run python scripts/sibling_probe/ladder.py banks
    uv run python scripts/sibling_probe/ladder.py capture-rows
    MODAL_PROFILE=safety-sahan-2 uvx --python 3.12 modal@latest run --env camila-blank \\
        scripts/sibling_probe/ladder_capture_modal.py --run ladder1
    uv run python scripts/sibling_probe/ladder.py probes --run ladder1
    uv run python scripts/sibling_probe/ladder.py label
    uv run python scripts/sibling_probe/ladder.py figures
"""

import argparse
import asyncio
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from hallucination_bench import DATA, REPO_ROOT, paths
from hallucination_bench.cache import Cache, fingerprint
from hallucination_bench.evals.sibling_probe import br_bank as BB
from hallucination_bench.evals.sibling_probe import ladder as LD
from hallucination_bench.evals.sibling_probe import ladder_lex as LX
from hallucination_bench.llm import LLM, Usage, schema_block

MODEL = "claude-opus-5-5"
EFFORT = "low"
OUT = REPO_ROOT / "outputs" / "sibling" / "ladder"
SP = DATA / "sibling_probe"
BR_READOUTS = BB.BR / "readouts"
JUDGE = REPO_ROOT / "outputs" / "sibling" / "judge"
ARM_DIR = {
    "s3d": "s3d-rl600",
    "jlens": "jlens",
    "nla-rl-L42": "nla-rl-L42",
    "mlnla-rl600": "mlnla-rl600",
}


class Opus:
    """One cached structured-call helper per prompt version (own cache file)."""

    def __init__(self, version: str, concurrency: int = 24) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        self.version = version
        self.cache = Cache(OUT / f"cache_{version}.jsonl")
        self.llm: LLM | None = None
        self.usage = Usage()
        self.n_calls = 0
        self.concurrency = concurrency

    def cached(self, key, system, user):
        fp = fingerprint(self.version, MODEL, EFFORT, system, user)
        return self.cache.get(key, fp)

    async def call(self, key, system, user, schema, validate=None, retries=2, cache_only=False):
        fp = fingerprint(self.version, MODEL, EFFORT, system, user)
        row = self.cache.get(key, fp)
        if row is not None:
            return row["result"], row.get("meta", {})
        if cache_only:
            return None, {"pending": True}
        if self.llm is None:
            self.llm = LLM(rpm=900, concurrency=self.concurrency, effort=EFFORT)
        res = None
        n_bad = 0
        for _ in range(retries + 1):
            res = await self.llm.call(model=MODEL, system=system, user=user, schema=schema)
            self.usage.add(res.usage)
            self.n_calls += 1
            if res.result is not None and (validate is None or validate(res.result)):
                meta = {"usage": res.usage.to_dict(), "n_invalid_attempts": n_bad}
                self.cache.put(key, fp, {"result": res.result, "meta": meta})
                return res.result, meta
            n_bad += 1
        meta = {"raw": res.result if res else None, "n_invalid_attempts": n_bad}
        self.cache.put(key, fp, {"result": None, "meta": meta})
        return None, meta

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


def entity_items() -> list[dict[str, Any]]:
    doc = json.loads(BB.BANK_PATH.read_text())
    return [x for x in doc["items"] if x["subfamily"] == "entity"]


# ------------------------------------------------------------------ lexicons

LEX_SYSTEM = """You build matching lexicons for finding mentions of category members in text written by \
interpretability tools (English, often with Chinese fragments). Be comprehensive and precise: every alias \
must unambiguously name that concept in ordinary text. Return JSON only."""

LEX_USER_SPORT = """Category: SPORT. List about 80 distinct sports (Olympic and major professional / popular \
sports worldwide). You MUST include these ids exactly: {required}.

For each sport give:
- id: snake_case id;
- label: display name;
- en: English aliases that name the sport in running text: the sport name and its common variants, the \
practitioner noun (e.g. "boxer", "swimmer", "golfer", "quarterback" for american_football), and \
unambiguous league abbreviations (NFL, NBA, MLB, NHL, UFC, WWE, FIFA, ATP, WTA, PGA, F1, NASCAR). Do NOT \
include bare "football" (handled separately), bare "track", bare "ski" or other words with a common \
non-sport sense ("box", "pool" is OK only as "pool (billiards)" -> leave out). "hockey" belongs to ice_hockey; \
"field hockey" to field_hockey. track_and_field includes athletics, sprinting, sprinter, running, runner, \
marathon, hurdles, decathlon, long jump, high jump, shot put, javelin. pro_wrestling = professional wrestling, \
WWE; wrestling = Olympic / amateur wrestling ("wrestling", "wrestler"). soccer includes "association football".
- cjk: Chinese names (2+ characters), e.g. 拳击, 游泳, 足球 (soccer), 美式足球 / 橄榄球 (american_football);
- tokens: extra single-word forms only for matching one-token lists (e.g. "Swim", "Hoops"), usually empty."""

LEX_USER_INSTR = """Category: MUSICAL INSTRUMENT. List about 60 distinct musical instruments (orchestral, \
band, folk / world, electronic). You MUST include these ids exactly: {required}.

For each instrument give:
- id: snake_case id;
- label: display name;
- en: English aliases naming the instrument in running text: the name, variants, and the player noun \
("pianist", "trumpeter", "guitarist", "cellist", "violoncello"). piano includes "keyboard", "keyboards", \
"pianist", "grand piano"; guitar includes "electric guitar", "acoustic guitar", "guitarist" (but bass_guitar \
is separate: "bass guitar", "bassist", "electric bass"); double_bass = "double bass", "upright bass", \
"contrabass". Do not include bare words with common non-music senses ("keys", "horn" alone is OK only for \
french_horn as "French horn"; leave bare "horn" and "bass" out).
- cjk: Chinese names (2+ characters), e.g. 钢琴, 小号, 吉他, 大提琴;
- tokens: extra single-word forms for one-token lists, usually empty."""

LEX_USER_COUNTRY = """Category: COUNTRY. For each country below give its matching lexicon. Use the id given.

For each country:
- id: as given;
- label: common English name;
- en: English aliases that name the country in running text: official and common short names, former \
names still used (e.g. "Burma", "Swaziland", "Zaire" only if unambiguous), abbreviations (UK, USA, U.S., \
UAE, DRC), and DEMONYMS / adjectives ("French", "Egyptian", "Kenyan"). Do NOT include the capital city or \
any city name. united_kingdom MUST include England, English, Britain, British, Great Britain, UK, U.K.; \
russia MUST include USSR, Soviet Union, Soviet. Leave out demonyms that are also common words in other \
senses (e.g. "Chinese" is fine; "Turkish" fine; "Polish" -> leave out; "Swiss" fine).
- cjk: Chinese names (2+ characters), e.g. 法国, 埃及, 英国, 英格兰 (united_kingdom), 苏联 (russia);
- tokens: usually empty;
- capital: the capital city's English name (used only to exclude city mentions).

Countries (id: name):
{countries}"""

UN_MEMBERS = """Afghanistan|Albania|Algeria|Andorra|Angola|Antigua and Barbuda|Argentina|Armenia|Australia|Austria|Azerbaijan|Bahamas|Bahrain|Bangladesh|Barbados|Belarus|Belgium|Belize|Benin|Bhutan|Bolivia|Bosnia and Herzegovina|Botswana|Brazil|Brunei|Bulgaria|Burkina Faso|Burundi|Cabo Verde|Cambodia|Cameroon|Canada|Central African Republic|Chad|Chile|China|Colombia|Comoros|Congo (Republic of the Congo)|Costa Rica|Cote d'Ivoire|Croatia|Cuba|Cyprus|Czechia|Democratic Republic of the Congo|Denmark|Djibouti|Dominica|Dominican Republic|Ecuador|Egypt|El Salvador|Equatorial Guinea|Eritrea|Estonia|Eswatini|Ethiopia|Fiji|Finland|France|Gabon|Gambia|Georgia|Germany|Ghana|Greece|Grenada|Guatemala|Guinea|Guinea-Bissau|Guyana|Haiti|Honduras|Hungary|Iceland|India|Indonesia|Iran|Iraq|Ireland|Israel|Italy|Jamaica|Japan|Jordan|Kazakhstan|Kenya|Kiribati|Kuwait|Kyrgyzstan|Laos|Latvia|Lebanon|Lesotho|Liberia|Libya|Liechtenstein|Lithuania|Luxembourg|Madagascar|Malawi|Malaysia|Maldives|Mali|Malta|Marshall Islands|Mauritania|Mauritius|Mexico|Micronesia|Moldova|Monaco|Mongolia|Montenegro|Morocco|Mozambique|Myanmar|Namibia|Nauru|Nepal|Netherlands|New Zealand|Nicaragua|Niger|Nigeria|North Korea|North Macedonia|Norway|Oman|Pakistan|Palau|Panama|Papua New Guinea|Paraguay|Peru|Philippines|Poland|Portugal|Qatar|Romania|Russia|Rwanda|Saint Kitts and Nevis|Saint Lucia|Saint Vincent and the Grenadines|Samoa|San Marino|Sao Tome and Principe|Saudi Arabia|Senegal|Serbia|Seychelles|Sierra Leone|Singapore|Slovakia|Slovenia|Solomon Islands|Somalia|South Africa|South Korea|South Sudan|Spain|Sri Lanka|Sudan|Suriname|Sweden|Switzerland|Syria|Tajikistan|Tanzania|Thailand|Timor-Leste|Togo|Tonga|Trinidad and Tobago|Tunisia|Turkey|Turkmenistan|Tuvalu|Uganda|Ukraine|United Arab Emirates|United Kingdom|United States|Uruguay|Uzbekistan|Vanuatu|Vatican City|Venezuela|Vietnam|Yemen|Zambia|Zimbabwe|Taiwan|Palestine|Kosovo|Scotland|Wales|Northern Ireland"""

COUNTRY_ID_FIX = {
    "Congo (Republic of the Congo)": "republic_of_the_congo",
    "Democratic Republic of the Congo": "dr_congo",
    "Cote d'Ivoire": "ivory_coast",
}

LEX_CONCEPT = {
    "type": "object",
    "additionalProperties": False,
    "required": ["id", "label", "en", "cjk", "tokens"],
    "properties": {
        "id": {"type": "string"},
        "label": {"type": "string"},
        "en": {"type": "array", "items": {"type": "string"}},
        "cjk": {"type": "array", "items": {"type": "string"}},
        "tokens": {"type": "array", "items": {"type": "string"}},
    },
}
LEX_COUNTRY_CONCEPT = {
    **LEX_CONCEPT,
    "required": [*LEX_CONCEPT["required"], "capital"],
    "properties": {**LEX_CONCEPT["properties"], "capital": {"type": "string"}},
}
LEX_SCHEMA = schema_block(
    "lexicon", {"concepts": {"type": "array", "items": LEX_CONCEPT}}, ["concepts"]
)
LEX_COUNTRY_SCHEMA = schema_block(
    "lexicon", {"concepts": {"type": "array", "items": LEX_COUNTRY_CONCEPT}}, ["concepts"]
)
ALI_CLASSES = [
    "boxing",
    "wrestling",
    "pro_wrestling",
    "mma",
    "track_and_field",
    "swimming",
    "tennis",
    "basketball",
    "baseball",
    "american_football",
    "soccer",
    "cricket",
    "golf",
    "gymnastics",
    "figure_skating",
    "ice_hockey",
    "cycling",
    "motor_racing",
    "skiing",
]
INSTR_REQUIRED = [
    "piano",
    "trumpet",
    "guitar",
    "cello",
    "violin",
    "saxophone",
    "drums",
    "bass_guitar",
    "double_bass",
    "clarinet",
    "flute",
    "trombone",
    "organ",
    "harp",
    "banjo",
    "harmonica",
]
# code-forced equivalences / additions (plan rev 2 item 6), applied after Opus
FORCED = {
    "country": {
        "united_kingdom": [
            "England",
            "English",
            "Britain",
            "British",
            "Great Britain",
            "UK",
            "U.K.",
            "United Kingdom",
            "英国",
            "英格兰",
            "大不列颠",
        ],
        "russia": ["Russia", "Russian", "USSR", "Soviet Union", "Soviet", "苏联", "俄罗斯"],
        "united_states": ["United States", "USA", "U.S.", "U.S.A.", "America", "American", "美国"],
        "turkey": ["Turkey", "Türkiye", "Turkish", "土耳其"],
        # bare names the lexicon call omitted (Jordan/Georgia/Guinea/Niger stay case-sensitive)
        "jordan": ["Jordan"],
        "georgia": ["Georgia"],
        "guinea": ["Guinea"],
        "luxembourg": ["Luxembourg"],
        "niger": ["Niger"],
    },
    "sport": {
        "track_and_field": [
            "track and field",
            "track & field",
            "athletics",
            "sprinting",
            "sprinter",
            "sprint",
            "running",
            "runner",
            "track athlete",
            "田径",
            "短跑",
        ],
        "american_football": [
            "American football",
            "NFL",
            "quarterback",
            "gridiron",
            "running back",
            "美式足球",
            "橄榄球",
        ],
        "soccer": ["soccer", "association football", "足球"],
        "ice_hockey": ["ice hockey", "hockey", "NHL", "冰球"],
        "pro_wrestling": ["professional wrestling", "pro wrestling", "WWE", "职业摔角"],
        "wrestling": ["wrestling", "wrestler", "摔跤", "摔角"],
    },
    "musical_instrument": {
        "piano": [
            "piano",
            "pianist",
            "keyboard",
            "keyboards",
            "keyboardist",
            "grand piano",
            "钢琴",
            "键盘",
        ],
    },
}
DROP_ALIASES = {  # never aliases (ambiguous / non-category senses)
    "sport": {
        "football",
        "footballer",
        "track",
        "ski",
        "box",
        "pool",
        "run",
        "racing",
        "race",
        "fight",
        "fighter",
        "sport",
        "sports",
        "athlete",
        "olympic",
        "olympics",
        "ball",
        "games",
        "play",
        "archer",
        "sailor",
        "投球",
    },
    "musical_instrument": {"keys", "horn", "bass", "instrument", "drum kit", "piper", "celeste"},
    "country": {"polish", "chinese checkers"},
}
AMBIGUOUS = {
    "sport": {
        "football": ["american_football", "soccer"],
        "footballer": ["american_football", "soccer"],
        "Football": ["american_football", "soccer"],
    }
}


def _country_chunks() -> list[list[tuple[str, str]]]:
    names = UN_MEMBERS.split("|")
    pairs = [(COUNTRY_ID_FIX.get(n, BB.snake(n)), n) for n in names]
    return [pairs[i : i + 34] for i in range(0, len(pairs), 34)]


def _clean_concept(cat: str, c: dict[str, Any]) -> dict[str, Any]:
    drop = DROP_ALIASES.get(cat, set())
    en = []
    for a in c.get("en", []):
        a = a.strip()
        if a and LX.norm_alias(a) not in drop and not LX.has_cjk(a) and a not in en:
            en.append(a)
    cjk = []
    for a in [*c.get("cjk", []), *[x for x in c.get("en", []) if LX.has_cjk(x)]]:
        a = a.strip()
        if len(a) >= 2 and a not in cjk:
            cjk.append(a)
    return {
        "label": c["label"],
        "en": en,
        "cjk": cjk,
        "tokens": [t for t in c.get("tokens", []) if t.strip()],
    }


async def stage_lexicons() -> None:
    op = Opus(LX.LEXICON_VERSION, concurrency=8)
    jobs = []
    ali_and_gold = sorted(
        set(ALI_CLASSES) | {x["gold"] for x in entity_items() if x["category"] == "sport"}
    )
    jobs.append(
        ("sport", "lex:sport", LEX_USER_SPORT.format(required=", ".join(ali_and_gold)), LEX_SCHEMA)
    )
    instr_req = sorted(
        set(INSTR_REQUIRED)
        | {x["gold"] for x in entity_items() if x["category"] == "musical_instrument"}
    )
    jobs.append(
        (
            "musical_instrument",
            "lex:instrument",
            LEX_USER_INSTR.format(required=", ".join(instr_req)),
            LEX_SCHEMA,
        )
    )
    chunks = _country_chunks()
    for i, ch in enumerate(chunks):
        jobs.append(
            (
                "country",
                f"lex:country:{i}",
                LEX_USER_COUNTRY.format(countries="\n".join(f"{a}: {b}" for a, b in ch)),
                LEX_COUNTRY_SCHEMA,
            )
        )
    try:
        res = await asyncio.gather(*(op.call(k, LEX_SYSTEM, u, s) for _, k, u, s in jobs))
    finally:
        await op.close()
    cats: dict[str, dict[str, Any]] = {c: {"concepts": {}, "block": []} for c in LX.CATEGORIES}
    capitals: dict[str, str] = {}
    for (cat, key, _, _), (r, _m) in zip(jobs, res, strict=True):
        if r is None:
            raise SystemExit(f"lexicon call failed: {key}")
        for c in r["concepts"]:
            cid = BB.snake(c["id"])
            cats[cat]["concepts"][cid] = _clean_concept(cat, c)
            if cat == "country" and c.get("capital"):
                capitals[cid] = c["capital"].strip()
    # country ids must be exactly the requested ones
    want = {a for ch in chunks for a, _ in ch}
    got = set(cats["country"]["concepts"])
    if want - got:
        raise SystemExit(f"country lexicon misses {sorted(want - got)}")
    for extra in got - want:
        del cats["country"]["concepts"][extra]
    for cat, forced in FORCED.items():
        for cid, aliases in forced.items():
            c = cats[cat]["concepts"].setdefault(
                cid, {"label": cid.replace("_", " ").title(), "en": [], "cjk": [], "tokens": []}
            )
            for a in aliases:
                key = "cjk" if LX.has_cjk(a) else "en"
                if a not in c[key]:
                    c[key].append(a)
            # a forced alias belongs to this concept only
            for oid, oc in cats[cat]["concepts"].items():
                if oid != cid:
                    for key in ("en", "cjk"):
                        oc[key] = [
                            x
                            for x in oc[key]
                            if LX.norm_alias(x) not in {LX.norm_alias(a) for a in aliases}
                        ]
    # remove remaining cross-concept duplicate aliases (ambiguous): keep none, record them
    removed: dict[str, list[str]] = {}
    for cat, body in cats.items():
        owner: dict[str, list[str]] = defaultdict(list)
        for cid, c in body["concepts"].items():
            for key in ("en", "cjk"):
                for a in c[key]:
                    owner[LX.norm_alias(a)].append(cid)
        dup = {a for a, v in owner.items() if len(set(v)) > 1}
        for c in body["concepts"].values():
            for key in ("en", "cjk"):
                c[key] = [a for a in c[key] if LX.norm_alias(a) not in dup]
        removed[cat] = sorted(dup)
    # capitals: never country aliases; block phrases ("X City" too) unless the capital IS the country name
    country_aliases = {
        LX.norm_alias(a) for c in cats["country"]["concepts"].values() for a in c["en"]
    }
    block = set()
    for cap in capitals.values():
        if LX.norm_alias(cap) in country_aliases:
            continue  # city-state (Singapore, Monaco, ...): the name stays a country mention
        block.add(cap)
    block |= {
        "Mexico City",
        "Kuwait City",
        "Panama City",
        "Guatemala City",
        "Vatican City",
        "New Mexico",
        "New Jersey",
        "New Guinea",
        "Georgia Tech",
        "Indian Ocean",
        "Indian Express",
        "Jordan River",
        "Washington",
        "Hong Kong",
        "Macau",
        "Indianapolis",
        "印度洋",
    }
    cats["country"]["block"] = sorted(block)
    lex = {
        "version": LX.LEXICON_VERSION,
        "provenance": f"{MODEL} effort {EFFORT} (brlex-v1, cache outputs/sibling/ladder/cache_brlex-v1.jsonl) + code-forced equivalences (scripts/sibling_probe/ladder.py FORCED / DROP_ALIASES / AMBIGUOUS)",
        "categories": cats,
        "ambiguous": AMBIGUOUS,
        "capitals": capitals,
        "removed_duplicate_aliases": removed,
    }
    lex["checks"] = LX.overlap_report(lex)
    LX.LEXICON_PATH.write_text(json.dumps(lex, ensure_ascii=False, indent=1) + "\n")
    for cat, chk in lex["checks"].items():
        print(
            f"[lexicons] {cat}: {chk['n_concepts']} concepts, {chk['n_aliases']} aliases, dup {len(chk['duplicates'])}, "
            f"short_cjk {len(chk['short_cjk'])}, substring pairs {chk['n_substring_pairs']}, removed {len(removed[cat])}"
        )
    print(f"[lexicons] calls {op.n_calls}: {op.usage.report()} -> {LX.LEXICON_PATH}")


# ------------------------------------------------------------------ mentions


def load_cells() -> dict[str, list[dict[str, Any]]]:
    """method -> readout cells of the 35 entity items: {item, layer, pos, samples | tokens+summary}."""
    ids = {x["id"] for x in entity_items()}
    out: dict[str, list[dict[str, Any]]] = {}
    summaries = {}
    for r in read_jsonl(JUDGE / "br-jlens" / "cells.jsonl"):
        if r["id"] in ids:
            summaries[(r["id"], r["layer"], r["pos"])] = r.get("summary")
    for m, arm in ARM_DIR.items():
        cells = []
        for r in read_jsonl(BR_READOUTS / arm / "basic_readout.jsonl"):
            if r["id"] not in ids:
                continue
            c = {"item": r["id"], "layer": int(r["layer"]), "pos": int(r["pos"])}
            if "tokens" in r:
                c["tokens"] = [LD.decode_tok(t) for t in r["tokens"]]
                c["summary"] = summaries.get((r["id"], int(r["layer"]), int(r["pos"])))
            else:
                c["samples"] = LD.split_samples(r["samples"])
            cells.append(c)
        out[m] = cells
    return out


def stage_mentions() -> None:
    lex = json.loads(LX.LEXICON_PATH.read_text())
    items = {x["id"]: x for x in entity_items()}
    matchers = {c: LX.Matcher(lex, c) for c in LX.CATEGORIES}
    cells = load_cells()
    units = LD.build_units(cells, items, matchers)
    write_jsonl(OUT / "mentions.jsonl", units)
    n_hits = sum(len(u["samples"]) for u in units)
    by = Counter(u["method"] for u in units)
    print(
        f"[mentions] {len(units)} units, {n_hits} (unit, sample) judge calls; by method {dict(by)}"
    )
    print("[mentions] by category", dict(Counter(items[u["item"]]["category"] for u in units)))
    print(
        "[mentions] top concepts",
        Counter((u["category"], u["concept"]) for u in units).most_common(25),
    )
    nocells = {m: len(c) for m, c in cells.items()}
    print(
        f"[mentions] cells per method {nocells}; J-lens cells without summary: "
        f"{sum(1 for c in cells['jlens'] if not c.get('summary'))}"
    )


# ------------------------------------------------------------------ judge


def concept_label(lex: dict[str, Any], cat: str, cid: str) -> str:
    c = lex["categories"][cat]["concepts"].get(cid)
    return c["label"] if c else cid.replace("_", " ")


async def stage_judge(args) -> None:
    lex = json.loads(LX.LEXICON_PATH.read_text())
    items = {x["id"]: x for x in entity_items()}
    units = read_jsonl(OUT / "mentions.jsonl")
    op = Opus(LD.COMMIT_VERSION, concurrency=min(32, args.concurrency))
    jobs = []
    for u in units:
        for s in u["samples"]:
            sys_, user = LD.render_commit(
                u, s, items[u["item"]]["prompt"], concept_label(lex, u["category"], u["concept"])
            )
            jobs.append((u, s, f"c:{u['uid']}:s{s['si']}", sys_, user))
    if args.limit:
        jobs = jobs[: args.limit]
    done = 0

    async def one(j):
        nonlocal done
        _u, s, key, sys_, user = j
        r, meta = await op.call(
            key, sys_, user, LD.COMMIT_SCHEMA, validate=lambda x, s=s: LD.valid_commit(x, s)
        )
        done += 1
        if done % 200 == 0:
            print(f"  [judge {done}/{len(jobs)}] {op.usage.report()}", flush=True)
        return r, meta

    try:
        res = await asyncio.gather(*(one(j) for j in jobs))
    finally:
        await op.close()
    verdicts: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for (u, s, _key, _, _), (r, meta) in zip(jobs, res, strict=True):
        v = LD.parse_commit(r, s)
        v["n_invalid_attempts"] = meta.get("n_invalid_attempts", 0)
        verdicts[u["uid"]][s["si"]] = v
    out = []
    for u in units:
        vs = verdicts.get(u["uid"], {})
        if len(vs) < len(u["samples"]):
            continue  # --limit
        for s in u["samples"]:
            s["verdict"] = vs[s["si"]]
        out.append({**u, **LD.rollup([s["verdict"] for s in u["samples"]])})
    write_jsonl(OUT / "commit.jsonl", out)
    n_s = sum(len(u["samples"]) for u in out)
    fc = sum(s["verdict"]["fail_closed"] for u in out for s in u["samples"])
    print(
        f"[judge] {len(out)} units ({n_s} samples): committed units {sum(u['committed'] for u in out)}, "
        f"fail-closed samples {fc}; forms {Counter(s['verdict']['form'] for u in out for s in u['samples'])}"
    )
    print(f"[judge] calls this run {op.n_calls}: {op.usage.report()}")
    (OUT / "judge_run.json").write_text(
        json.dumps(
            {
                "version": LD.COMMIT_VERSION,
                "model": MODEL,
                "effort": EFFORT,
                "n_units": len(out),
                "n_samples": n_s,
                "n_fail_closed": fc,
                "spend_this_run": op.usage.to_dict(),
            },
            indent=1,
        )
        + "\n"
    )


# ------------------------------------------------------------------ calibration todo / calib

ALI = "br-ent-ath-ali"


def _todo_row(u: dict[str, Any], items, lex) -> dict[str, Any]:
    samples = []
    for s in u["samples"]:
        _, user = LD.render_commit(
            u, s, items[u["item"]]["prompt"], concept_label(lex, u["category"], u["concept"])
        )
        samples.append(
            {"si": s["si"], "shown": user.split("\n\n", 1)[1], "committed": None, "form": None}
        )
    return {
        "uid": u["uid"],
        "method": u["method"],
        "prompt": items[u["item"]]["prompt"],
        "category": u["category"],
        "concept": concept_label(lex, u["category"], u["concept"]),
        "samples": samples,
        "unit_committed": None,
        "note": "",
    }


def stage_todo(args) -> None:
    import random

    lex = json.loads(LX.LEXICON_PATH.read_text())
    items = {x["id"]: x for x in entity_items()}
    units = read_jsonl(OUT / "commit.jsonl")
    rng = random.Random(5)
    non_ali = [u for u in units if u["item"] != ALI]
    # dev: 40, stratified by method x judge verdict (verdicts NOT written to the file)
    strata: dict[tuple[str, bool], list[dict[str, Any]]] = defaultdict(list)
    for u in non_ali:
        strata[(u["method"], u["committed"])].append(u)
    quota = {
        ("s3d", True): 8,
        ("s3d", False): 6,
        ("jlens", True): 4,
        ("jlens", False): 6,
        ("mlnla-rl600", True): 6,
        ("mlnla-rl600", False): 5,
        ("nla-rl-L42", True): 2,
        ("nla-rl-L42", False): 3,
    }
    dev = []
    for k, n in quota.items():
        pool = strata.get(k, [])
        dev += rng.sample(pool, min(n, len(pool)))
    while len(dev) < 40:
        u = rng.choice(non_ali)
        if u not in dev:
            dev.append(u)
    dev_ids = {u["uid"] for u in dev}
    write_jsonl(SP / "brcommit_dev_todo.jsonl", [_todo_row(u, items, lex) for u in dev])
    # gold: 80 units stratified by method x category (never by judge output), disjoint from dev, + all Ali units
    rest = [u for u in non_ali if u["uid"] not in dev_ids]
    by: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for u in rest:
        by[(u["method"], u["category"])].append(u)
    share = {"s3d": 30, "jlens": 26, "mlnla-rl600": 20, "nla-rl-L42": 4}
    gold = []
    for m, n in share.items():
        cats = [c for c in LX.CATEGORIES if by.get((m, c))]
        tot = sum(len(by[(m, c)]) for c in cats)
        for c in cats:
            k = max(1, round(n * len(by[(m, c)]) / tot)) if tot else 0
            gold += rng.sample(by[(m, c)], min(k, len(by[(m, c)])))
    gold = gold[:80]
    left = [u for u in rest if u not in gold]
    gold += rng.sample(left, max(0, 80 - len(gold)))
    ali = [u for u in units if u["item"] == ALI]
    rows = [{**_todo_row(u, items, lex), "set": "gold80"} for u in gold] + [
        {**_todo_row(u, items, lex), "set": "ali"} for u in ali
    ]
    rng.shuffle(rows)
    write_jsonl(SP / "brcommit_gold_todo.jsonl", rows)
    print(
        f"[todo] dev {len(dev)} -> {SP / 'brcommit_dev_todo.jsonl'}; gold {len(gold)} + Ali {len(ali)} -> {SP / 'brcommit_gold_todo.jsonl'}"
    )
    print(
        "[todo] gold by method",
        Counter(u["method"] for u in gold),
        "Ali by method",
        Counter(u["method"] for u in ali),
    )


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (round(c - h, 3), round(c + h, 3))


def calib_stats(pairs: list[tuple[bool, bool]]) -> dict[str, Any]:
    """pairs (gold, judge) committed booleans."""
    n = len(pairs)
    k = sum(g == j for g, j in pairs)
    out: dict[str, Any] = {
        "n": n,
        "accuracy": round(k / n, 3) if n else None,
        "wilson95": wilson(k, n),
    }
    for cls in (True, False):
        tp = sum(g == cls and j == cls for g, j in pairs)
        pp = sum(j == cls for _, j in pairs)
        ap = sum(g == cls for g, _ in pairs)
        name = "committed" if cls else "not_committed"
        out[name] = {
            "precision": round(tp / pp, 3) if pp else None,
            "recall": round(tp / ap, 3) if ap else None,
            "n_gold": ap,
            "n_judge": pp,
        }
    return out


def stage_calib(args) -> None:
    gpath = SP / "brcommit_gold.jsonl"
    if not gpath.exists():
        sys.exit(f"no gold labels at {gpath} (labeler output)")
    gold = {r["uid"]: r for r in read_jsonl(gpath)}
    units = {u["uid"]: u for u in read_jsonl(OUT / "commit.jsonl")}
    res = {}
    for name, sel in (
        ("gold80", lambda r: r.get("set") == "gold80"),
        ("ali", lambda r: r.get("set") == "ali"),
        ("all", lambda r: True),
    ):
        up, sp = [], []
        for uid, r in gold.items():
            if not sel(r) or uid not in units or r.get("unit_committed") is None:
                continue
            u = units[uid]
            up.append((bool(r["unit_committed"]), bool(u["committed"])))
            jv = {s["si"]: s["verdict"]["committed"] for s in u["samples"]}
            for s in r["samples"]:
                if s.get("committed") is not None and s["si"] in jv:
                    sp.append((bool(s["committed"]), bool(jv[s["si"]])))
        res[name] = {"unit": calib_stats(up), "sample": calib_stats(sp)}
        by_m: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
        for uid, r in gold.items():
            if sel(r) and uid in units and r.get("unit_committed") is not None:
                by_m[units[uid]["method"]].append(
                    (bool(r["unit_committed"]), bool(units[uid]["committed"]))
                )
        res[name]["unit_by_method"] = {m: calib_stats(v) for m, v in by_m.items()}
    acc = res["gold80"]["unit"]["accuracy"]
    res["target"] = 0.85
    res["below_target"] = acc is None or acc < 0.85
    (OUT / "calib.json").write_text(json.dumps(res, indent=1) + "\n")
    print(
        json.dumps(
            {k: (v["unit"] if isinstance(v, dict) and "unit" in v else v) for k, v in res.items()},
            indent=1,
        )
    )


# ------------------------------------------------------------------ entity banks

BANK_VERSION = "brbank-v1"
ENTITY_BANKS = SP / "br_entity_banks.json"
ALI_BANK = paths.ALI_PROBE_ROOT / "data" / "ali_probe" / "bank.json"
N_ENT = 50
BANK_TEMPLATES = {
    "sport": [
        "The athlete {e} plays the sport of",
        "The sport that {e} is famous for playing is",
        "In professional sports, {e} is known for the sport of",
    ],
    "country_capital": [
        "The city of {e} is the capital of",
        "{e} is a city in the country of",
        "The city of {e} is located in the country of",
    ],
    "country_landmark": [
        "The {e} is located in the country of",
        "The {e} is a famous landmark in the country of",
        "Tourists visit the {e} in the country of",
    ],
    "musical_instrument": [
        "The musician {e} is famous for playing the",
        "The instrument that {e} plays is the",
        "{e} is best known as a player of the",
    ],
}
BANK_CATEGORY = {
    "sport": "sport",
    "country_capital": "country",
    "country_landmark": "country",
    "musical_instrument": "musical_instrument",
}
CONTROLS = {
    "sport": ["curling", "archery", "sailing"],
    "country_capital": ["thailand", "vietnam", "norway"],
    "country_landmark": ["thailand", "vietnam", "norway"],
    "musical_instrument": ["accordion", "clarinet", "oboe"],
}
BANK_SYSTEM = "You list real, well-known entities for building a knowledge probe. Return JSON only. Never invent names."
BANK_USER = {
    "sport": "List {n} famous real athletes whose main sport is unambiguously {label} (most famous first; any era, any country; full names as commonly written). Do not include: {excl}.",
    "country_capital": "List {n} real cities and towns in {label} that are NOT its capital (best-known first). Common English name only, no region or country suffix, and avoid names that contain the country's name. Do not include: {excl}.",
    "country_landmark": 'List {n} real, well-known landmarks located in {label}: buildings, monuments, temples, churches, palaces, museums, bridges, ruins, natural sites (best-known first). Give each name as it would follow "The" in "The X is located in the country of" (omit a leading "The"). Avoid names that contain the country\'s or a nationality\'s name. Do not include: {excl}.',
    "musical_instrument": "List {n} famous real musicians best known for playing the {label} (most famous first; any genre and era; full names as commonly written). Do not include: {excl}.",
}
BANK_SCHEMA = schema_block(
    "entities", {"entities": {"type": "array", "items": {"type": "string"}}}, ["entities"]
)


def bank_of_item(it: dict[str, Any]) -> str:
    if it["category"] == "country":
        return "country_capital" if "-cap-" in it["id"] else "country_landmark"
    return it["category"]


def bank_classes(units, items) -> dict[str, list[str]]:
    cls: dict[str, set[str]] = defaultdict(set)
    for it in items.values():
        cls[bank_of_item(it)].add(LD.lex_gold(it["gold"]))
    for u in units:
        if u["committed"]:
            cls[bank_of_item(items[u["item"]])].add(u["concept"])
    named = defaultdict(set)
    for u in units:
        named[BANK_CATEGORY[bank_of_item(items[u["item"]])]].add(u["concept"])
    out = {}
    for b, cs in cls.items():
        ctrl = [c for c in CONTROLS[b] if c not in named[BANK_CATEGORY[b]]]
        if len(ctrl) < len(CONTROLS[b]):
            print(
                f"[banks] WARNING {b}: controls named by some method: {set(CONTROLS[b]) - set(ctrl)}"
            )
        if b == "sport":
            ali = json.loads(ALI_BANK.read_text())
            cs |= set(ali["classes"])  # keep the Ali case study's classes
        out[b] = sorted(cs - set(ctrl)) + ctrl
    return out


async def stage_banks(args) -> None:
    lex = json.loads(LX.LEXICON_PATH.read_text())
    items = {x["id"]: x for x in entity_items()}
    units = read_jsonl(OUT / "commit.jsonl")
    classes = bank_classes(units, items)
    item_ents = {x["entity"].casefold() for x in items.values()}
    ali = json.loads(ALI_BANK.read_text())
    ali_ents: dict[str, list[str]] = defaultdict(list)
    for r in ali["rows"]:
        if r["block"] == "train" and r["template_idx"] == 0:
            ali_ents[r["sport"]].append(r["entity"])
    capitals = lex.get("capitals", {})
    op = Opus(BANK_VERSION, concurrency=16)
    jobs = []
    for b, cs in classes.items():
        cat = BANK_CATEGORY[b]
        for c in cs:
            have = ali_ents.get(c, []) if b == "sport" else []
            n = N_ENT - len(have)
            if n <= 0:
                continue
            excl = [*have, *sorted(x["entity"] for x in items.values() if bank_of_item(x) == b)]
            if b == "country_capital" and capitals.get(c):
                excl.append(capitals[c])
            user = BANK_USER[b].format(
                n=n + 8, label=concept_label(lex, cat, c), excl="; ".join(excl) or "none"
            )
            jobs.append((b, c, user))
    try:
        res = await asyncio.gather(
            *(op.call(f"bank:{b}:{c}", BANK_SYSTEM, u, BANK_SCHEMA) for b, c, u in jobs)
        )
    finally:
        await op.close()
    got = {
        (b, c): (r or {}).get("entities", []) for (b, c, _), (r, _m) in zip(jobs, res, strict=True)
    }
    doc: dict[str, Any] = {
        "version": BANK_VERSION,
        "provenance": f"Ali bank ({ALI_BANK}) t0 athletes + {MODEL} effort {EFFORT} top-ups (cache outputs/sibling/ladder/cache_{BANK_VERSION}.jsonl); code filters: dedupe, the 35 item entities, capitals, names containing a category alias",
        "banks": {},
    }
    for b, cs in classes.items():
        cat = BANK_CATEGORY[b]
        m = LX.Matcher(lex, cat)
        ents: dict[str, list[str]] = {}
        dropped: dict[str, list[str]] = defaultdict(list)
        seen: set[str] = set()
        for c in cs:
            lst = []
            for e in [*(ali_ents.get(c, []) if b == "sport" else []), *got.get((b, c), [])]:
                e = e.strip()
                if e.lower().startswith("the ") and b == "country_landmark":
                    e = e[4:]
                k = e.casefold()
                if not e or k in seen:
                    continue
                if k in item_ents:
                    dropped["item_entity"].append(e)
                    continue
                if b == "country_capital" and any(k == v.casefold() for v in capitals.values()):
                    dropped["capital"].append(e)
                    continue
                if b != "sport" and b != "musical_instrument" and any(h.concept for h in m.find(e)):
                    dropped["names_category"].append(e)
                    continue
                seen.add(k)
                lst.append(e)
            ents[c] = lst[:N_ENT]
        doc["banks"][b] = {
            "category": cat,
            "templates": BANK_TEMPLATES[b],
            "classes": cs,
            "controls": [c for c in CONTROLS[b] if c in cs],
            "golds": sorted(
                {LD.lex_gold(x["gold"]) for x in items.values() if bank_of_item(x) == b}
            ),
            "entities": ents,
            "dropped": dict(dropped),
        }
        print(
            f"[banks] {b}: {len(cs)} classes, entities per class {dict((c, len(v)) for c, v in ents.items())}"
        )
    ENTITY_BANKS.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    print(f"[banks] calls {op.n_calls}: {op.usage.report()} -> {ENTITY_BANKS}")


TOPUP_TARGET = 40  # kept entities per class after the knowledge filter (Camila 2026-10-12)
EXTRA_CONTROLS = {"musical_instrument": ["ukulele", "bagpipes"]}  # oboe keeps 4 / 50


async def stage_topup(args) -> None:
    """Top up every class with < TOPUP_TARGET kept entities (probe_counts.json) with fresh Opus
    names (excluding all current ones); add EXTRA_CONTROLS. Appends to br_entity_banks.json and
    records the additions under ``topup``."""
    lex = json.loads(LX.LEXICON_PATH.read_text())
    items = {x["id"]: x for x in entity_items()}
    doc = json.loads(ENTITY_BANKS.read_text())
    counts = json.loads((OUT / "probe_counts.json").read_text())["counts"]
    capitals = lex.get("capitals", {})
    item_ents = {x["entity"].casefold() for x in items.values()}
    jobs = []
    for b, body in doc["banks"].items():
        cat = BANK_CATEGORY[b]
        for c in EXTRA_CONTROLS.get(b, []):
            if c not in body["classes"]:
                body["classes"].append(c)
                body["controls"].append(c)
                body["entities"][c] = []
        for c in body["classes"]:
            kept = counts.get(b, {}).get(c, {}).get("n_kept", 0)
            if c in counts.get(b, {}) and kept >= TOPUP_TARGET:
                continue
            have = body["entities"].get(c, [])
            n = 60 if have else N_ENT + 8
            excl = [*have, *sorted(x["entity"] for x in items.values() if bank_of_item(x) == b)]
            if b == "country_capital" and capitals.get(c):
                excl.append(capitals[c])
            user = BANK_USER[b].format(
                n=n, label=concept_label(lex, cat, c), excl="; ".join(excl) or "none"
            )
            jobs.append((b, c, user))
    op = Opus(BANK_VERSION, concurrency=16)
    try:
        res = await asyncio.gather(
            *(op.call(f"topup:{b}:{c}", BANK_SYSTEM, u, BANK_SCHEMA) for b, c, u in jobs)
        )
    finally:
        await op.close()
    for (b, c, _), (r, _m) in zip(jobs, res, strict=True):
        body = doc["banks"][b]
        m = LX.Matcher(lex, BANK_CATEGORY[b])
        seen = {e.casefold() for ents in body["entities"].values() for e in ents}
        new = []
        for e in (r or {}).get("entities", []):
            e = e.strip()
            if e.lower().startswith("the ") and b == "country_landmark":
                e = e[4:]
            k = e.casefold()
            if not e or k in seen or k in item_ents:
                continue
            if b == "country_capital" and any(k == v.casefold() for v in capitals.values()):
                continue
            if b in ("country_capital", "country_landmark") and any(h.concept for h in m.find(e)):
                continue
            seen.add(k)
            new.append(e)
        body["entities"][c] = body["entities"].get(c, []) + new
        body.setdefault("topup", {})[c] = new
        print(f"[topup] {b}/{c}: +{len(new)}")
    doc["provenance"] += (
        f"; topup (brbank-v1 'topup:' keys) for classes with < {TOPUP_TARGET} kept + extra controls {EXTRA_CONTROLS}"
    )
    ENTITY_BANKS.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    print(f"[topup] calls {op.n_calls}: {op.usage.report()}")


# ------------------------------------------------------------------ capture rows

LAYERS = [*range(20, 61, 4), 42]
STOP_TOKENS = {"the", "a", "an", "of", "in", "and", "or", "to", ""}


def class_words(lex, bank: str, classes: list[str], tok) -> dict[int, list[str]]:
    """first-token id (with a leading space) -> the bank classes it can start."""
    cat = BANK_CATEGORY[bank]
    out: dict[int, set[str]] = defaultdict(set)
    for c in classes:
        cc = lex["categories"][cat]["concepts"].get(c, {})
        for a in [cc.get("label", c), *cc.get("en", [])]:
            if not a or "(" in a:
                continue
            for form in (a, a.lower(), a[0].upper() + a[1:]):
                ids = tok(" " + form, add_special_tokens=False)["input_ids"]
                if not ids:
                    continue
                first = tok.decode([ids[0]]).strip()
                if first.lower() in STOP_TOKENS or (len(first) < 3 and len(ids) > 1):
                    continue  # a word piece ("g" of "germany"), not a word
                out[ids[0]].add(c)
    return {k: sorted(v) for k, v in out.items()}


def stage_capture_rows(args) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    lex = json.loads(LX.LEXICON_PATH.read_text())
    doc = json.loads(ENTITY_BANKS.read_text())
    items = {x["id"]: x for x in entity_items()}
    rows, words = [], {}
    for b, body in doc["banks"].items():
        cw = class_words(lex, b, body["classes"], tok)
        words[b] = {str(k): v for k, v in cw.items()}
        for c, ents in body["entities"].items():
            for e in ents:
                for t, tpl in enumerate(body["templates"]):
                    rows.append(
                        {"key": f"{b}|{c}|{e}|t{t}", "prompt": tpl.format(e=e), "gen": t == 0}
                    )
    for it in items.values():
        b = bank_of_item(it)
        p0 = doc["banks"][b]["templates"][0].format(e=it["entity"])
        if p0 != it["prompt"]:
            sys.exit(f"{it['id']}: template-0 render {p0!r} != item prompt {it['prompt']!r}")
        rows.append(
            {
                "key": f"item|{it['id']}",
                "prompt": it["prompt"],
                "expect_ids": it["input_ids"],
                "gen": True,
            }
        )
    word_ids = sorted({int(k) for w in words.values() for k in w})
    if args.exclude_runs:
        have = set()
        for r in args.exclude_runs.split(","):
            have |= {m["key"] for m in json.loads((CAPTURE / r / "meta.json").read_text())["rows"]}
            old = json.loads((CAPTURE / r / "meta.json").read_text())["word_ids"]
            word_ids = old  # keep the word-id order of the first run (word_logp columns align)
        rows = [r for r in rows if r["key"] not in have]
    spec = {"layers": LAYERS, "word_ids": word_ids, "class_words": words, "rows": rows}
    (OUT / "capture_rows.json").write_text(json.dumps(spec, ensure_ascii=False) + "\n")
    print(
        f"[capture-rows] {len(rows)} rows ({sum(r['gen'] for r in rows)} with greedy), {len(word_ids)} word ids, layers {LAYERS}"
    )
    for b, w in words.items():
        amb = {tok.decode([int(k)]): v for k, v in w.items() if len(v) > 1}
        print(f"  {b}: {len(w)} word ids; shared first tokens {amb}")


# ------------------------------------------------------------------ probes

CAPTURE = OUT / "capture"
MIN_CLASS_ENTS = 10  # a bank class needs >= 10 known entities to be trained


FILTER_EQUIV = {"pro_wrestling": {"wrestling"}}  # Ali: the word "wrestling" names both classes


def knows(
    word_logp: dict[int, float], cw: dict[str, list[str]], true: str, greedy: str | None, matcher
) -> bool:
    """Knowledge filter (Ali method, extended for multi-token names).

    Keep if (a) the most likely category word (first tokens of the bank's class names) names
    only the true class, or names several classes including it and the greedy continuation's
    first category mention agrees; or (b) the greedy continuation's first category mention is
    the true class (names whose first token is a bare piece, e.g. " c|ello", are not in the word
    list). ``FILTER_EQUIV``: the greedy "wrestling" also counts for pro_wrestling."""
    ok = {true} | FILTER_EQUIV.get(true, set())
    hits = [h.concept for h in matcher.find(greedy or "") if h.concept]
    greedy_ok = bool(hits) and hits[0] in ok
    ids = [int(k) for k in cw if int(k) in word_logp]
    if ids:
        best = max(ids, key=lambda i: word_logp[i])
        cls = cw[str(best)]
        if true in cls and (len(cls) == 1 or greedy_ok):
            return True
    return greedy_ok


def _probe_task(a):
    from hallucination_bench.evals.sibling_probe import ladder_probe as LP

    return LP.probe_task(a)


def run_probes_modal(tasks, args):
    """Fit every (bank, layer) task on Modal CPU (scripts/sibling_probe/ladder_probes_modal.py):
    the acts files are read from the ``sibling-probes`` volume (``ladder/<run>/acts.npy``)."""
    import pickle
    import subprocess
    import tempfile

    vol_tasks = []
    for a in tasks:
        runs = [("/vol/ladder/" + Path(p).parent.name + "/acts.npy", off, n) for p, off, n in a[3]]
        vol_tasks.append((*a[:3], runs, *a[4:]))
    with tempfile.TemporaryDirectory() as td:
        inp, outp = Path(td) / "tasks.pkl", Path(td) / "results.pkl"
        inp.write_bytes(pickle.dumps(vol_tasks))
        cmd = [
            "uvx",
            "--python",
            "3.12",
            "--with",
            "numpy",
            "modal@latest",
            "run",
            "--env",
            "camila-blank",
            str(REPO_ROOT / "scripts" / "sibling_probe" / "ladder_probes_modal.py"),
            "--inp",
            str(inp),
            "--out",
            str(outp),
        ]
        print("[probes] modal:", " ".join(cmd), flush=True)
        subprocess.run(
            cmd, check=True, env={**__import__("os").environ, "MODAL_PROFILE": "safety-sahan-2"}
        )
        return pickle.loads(outp.read_bytes())


def stage_probes(args) -> None:
    from concurrent.futures import ProcessPoolExecutor

    import numpy as np

    runs, metas, off = [], [], 0
    for r in args.run.split(","):
        m = json.loads((CAPTURE / r / "meta.json").read_text())
        runs.append((str(CAPTURE / r / "acts.npy"), off, len(m["rows"])))
        metas.append(m)
        off += len(m["rows"])
    meta = {
        "layers": metas[0]["layers"],
        "word_ids": metas[0]["word_ids"],
        "rows": [x for m in metas for x in m["rows"]],
    }
    assert all(m["layers"] == meta["layers"] and m["word_ids"] == meta["word_ids"] for m in metas)
    spec = json.loads((OUT / "capture_rows.json").read_text())
    doc = json.loads(ENTITY_BANKS.read_text())
    lex = json.loads(LX.LEXICON_PATH.read_text())
    items = {x["id"]: x for x in entity_items()}
    layers = meta["layers"]
    word_ids = meta["word_ids"]
    krow = {m["key"]: i for i, m in enumerate(meta["rows"])}
    tasks, counts, dropped_all, dropped_classes = [], {}, {}, {}
    for b, body in doc["banks"].items():
        cw = spec["class_words"][b]
        matcher = LX.Matcher(lex, BANK_CATEGORY[b])
        classes = body["classes"]
        kept: dict[str, list[str]] = {}
        dropped: dict[str, dict[str, str]] = {}
        for c, ents in body["entities"].items():
            kept[c] = []
            for e in ents:
                m = meta["rows"][krow[f"{b}|{c}|{e}|t0"]]
                wl = dict(zip(word_ids, m["word_logp"], strict=True))
                if knows(wl, cw, c, m["greedy"], matcher):
                    kept[c].append(e)
                else:
                    best = max((int(k) for k in cw if int(k) in wl), key=lambda i: wl[i])
                    dropped.setdefault(c, {})[e] = f"{cw[str(best)]} | {m['greedy']!r}"
        counts[b] = {
            c: {"n_bank": len(body["entities"][c]), "n_kept": len(kept[c])} for c in classes
        }
        dropped_all[b] = dropped
        rows, y, t, groups = [], [], [], []
        # classes with too few known entities cannot be trained: dropped (their units -> probe n/a)
        small = [c for c in classes if len(kept[c]) < MIN_CLASS_ENTS]
        classes = [c for c in classes if c not in small]
        dropped_classes[b] = small
        for ci, c in enumerate(classes):
            for e in kept[c]:
                for ti in range(len(body["templates"])):
                    rows.append(krow[f"{b}|{c}|{e}|t{ti}"])
                    y.append(ci)
                    t.append(ti)
                    groups.append(e)
        its = [x for x in items.values() if bank_of_item(x) == b]
        item_rows = [krow[f"item|{x['id']}"] for x in its]
        for li, L in enumerate(layers):
            tasks.append(
                (
                    b,
                    L,
                    li,
                    runs,
                    np.array(rows),
                    np.array(y),
                    np.array(t),
                    np.array(groups),
                    np.array(item_rows),
                    classes,
                    [LD.lex_gold(x["gold"]) for x in its],
                    [x["id"] for x in its],
                )
            )
    # post-filter counts BEFORE fitting
    print("[probes] post-filter entity counts per class (bank -> kept / generated):")
    for b, cc in counts.items():
        print(f"  {b}: " + ", ".join(f"{c} {v['n_kept']}/{v['n_bank']}" for c, v in cc.items()))
        few = [c for c, v in cc.items() if v["n_kept"] < LD.MIN_REFS]
        if few:
            print(
                f"    < {LD.MIN_REFS} kept (items with this gold -> probe n/a (few refs); concept classes still trained): {few}"
            )
    (OUT / "probe_counts.json").write_text(
        json.dumps({"counts": counts, "dropped": dropped_all}, ensure_ascii=False, indent=1) + "\n"
    )
    # item knowledge (diagnostic)
    for x in items.values():
        b = bank_of_item(x)
        m = meta["rows"][krow[f"item|{x['id']}"]]
        wl = dict(zip(word_ids, m["word_logp"], strict=True))
        x["_knows"] = knows(
            wl,
            spec["class_words"][b],
            LD.lex_gold(x["gold"]),
            m["greedy"],
            LX.Matcher(lex, BANK_CATEGORY[b]),
        )
    print(
        "[probes] items the model does not 'know' (filter rule):",
        [x["id"] for x in items.values() if not x["_knows"]],
    )
    if args.limit:
        tasks = tasks[: args.limit]
    out: dict[str, Any] = {
        "run": args.run,
        "layers": layers,
        "banks": {},
        "items": defaultdict(dict),
    }
    for b, body in doc["banks"].items():
        out["banks"][b] = {
            "classes": [c for c in body["classes"] if c not in dropped_classes[b]],
            "dropped_classes": dropped_classes[b],
            "controls": [c for c in body["controls"] if c not in dropped_classes[b]],
            "golds": body["golds"],
            "counts": counts[b],
            "layers": {},
        }
    if args.modal:
        results = run_probes_modal(tasks, args)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(_probe_task, tasks))
    for b, L, lay, cells in results:
        out["banks"][b]["layers"][str(L)] = lay
        for iid, cell in cells.items():
            out["items"][iid][str(L)] = cell
    out["item_knows"] = {x["id"]: x["_knows"] for x in items.values()}
    (OUT / "probes.json").write_text(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"[probes] -> {OUT / 'probes.json'}")


# ------------------------------------------------------------------ probe quality report + recipe choice


def used_classes(units, items) -> dict[str, set[str]]:
    """bank -> classes the ladder asks about (golds + concepts of committed units + controls)."""
    out: dict[str, set[str]] = defaultdict(set)
    for it in items.values():
        out[bank_of_item(it)].add(LD.lex_gold(it["gold"]))
    for u in units:
        if u["committed"]:
            out[bank_of_item(items[u["item"]])].add(u["concept"])
    return out


def stage_probe_report(args) -> None:
    import numpy as np

    from hallucination_bench.evals.sibling_probe import ladder_probe as LP

    pr = json.loads((OUT / "probes.json").read_text())
    items = {x["id"]: x for x in entity_items()}
    units = read_jsonl(OUT / "commit.jsonl")
    used = used_classes(units, items)
    layers = [str(L) for L in pr["layers"]]
    res: dict[str, Any] = {"by_probe": {}, "banks": {}}
    lines = [
        "# Plan 0005 probe quality (held-out entity folds only)",
        "",
        "Held-out template-0 AUROC per class (class vs the bank's other classes), GroupKFold(5) over entities, "
        "trained on all 3 templates. `used` = gold, a committed concept, or a control. Gate 0.85.",
        "",
    ]
    for p in LP.PROBE_TYPES:
        vals_all, vals_used = [], []
        for b, body in pr["banks"].items():
            for L in layers:
                au = body["layers"][L]["auroc"][p]
                for c in body["classes"]:
                    v = au[c]
                    vals_all.append(v)
                    if c in used[b] or c in body["controls"]:
                        vals_used.append(v)
        va, vu = np.array(vals_all, float), np.array(vals_used, float)
        res["by_probe"][p] = {
            "median_all": round(float(np.nanmedian(va)), 4),
            "median_used": round(float(np.nanmedian(vu)), 4),
            "pass_all": f"{int(np.sum(va >= LD.AUROC_GATE))}/{len(va)}",
            "pass_used": f"{int(np.sum(vu >= LD.AUROC_GATE))}/{len(vu)}",
            "min_used": round(float(np.nanmin(vu)), 4),
        }
    # recipe: the DiffMean slot of the fire pair = the higher median held-out AUROC over used (class, layer)
    best_dm = max(("dom", "wdm"), key=lambda p: res["by_probe"][p]["median_used"])
    pair = [x.strip() for x in args.fire_pair.split(",")]
    alt = ["logreg", "wdm" if pair[1] == "dom" else "dom"]
    res["fire_pair"] = pair
    res["alt_pair"] = alt
    res["auroc_best_diffmean"] = best_dm
    res["why"] = (
        f"fire pair {' + '.join(pair)} = the plan's pre-specified probes (a) logistic PCA-k and (b) one-vs-rest "
        f"difference of means; their (class, layer) cells below the 0.85 validity gate are probe n/a (gate not lowered). "
        f"The whitened DiffMean has the higher held-out AUROC ({res['by_probe']['wdm']['median_used']} vs "
        f"{res['by_probe']['dom']['median_used']} median over used cells) but is NOT the primary DiffMean: AUROC "
        f"measures separation BETWEEN classes, while the fire rule asks whether an item is an outlier AMONG its "
        f"same-gold references along concept S, and whitening by the within-class covariance shrinks exactly that "
        f"within-class variation (a priori argument, no item or lens output used). {' + '.join(alt)} is reported "
        f"as a sensitivity row"
    )
    lines += [
        "## Summary per probe type",
        "",
        "| probe | median (all) | median (used) | pass >= 0.85 (all) | pass (used) | min (used) |",
        "|---|---|---|---|---|---|",
    ]
    for p, r in res["by_probe"].items():
        lines.append(
            f"| {p} | {r['median_all']} | {r['median_used']} | {r['pass_all']} | {r['pass_used']} | {r['min_used']} |"
        )
    lines += ["", f"**Fire pair:** {' + '.join(res['fire_pair'])}. {res['why']}.", ""]
    below = []
    for b, body in pr["banks"].items():
        cnt = body["counts"]
        hyp = {L: body["layers"][L]["hyper"] for L in layers}
        res["banks"][b] = {"hyper": hyp, "counts": cnt}
        lines += [
            f"## {b}",
            "",
            "inner-CV choices per layer (5 outer folds): "
            + "; ".join(f"L{L} k={hyp[L]['k']} lam={hyp[L]['lam']}" for L in layers),
            "",
        ]
        for p in LP.PROBE_TYPES:
            lines += [
                f"### {b} / {p}",
                "",
                "| class | n kept | used | " + " | ".join(f"L{L}" for L in layers) + " |",
                "|---|---|---|" + "---|" * len(layers),
            ]
            for c in body["classes"]:
                row = [body["layers"][L]["auroc"][p][c] for L in layers]
                u = "ctrl" if c in body["controls"] else ("yes" if c in used[b] else "")
                lines.append(
                    f"| {c} | {cnt[c]['n_kept']} | {u} | "
                    + " | ".join((f"**{v:.2f}**" if v < LD.AUROC_GATE else f"{v:.2f}") for v in row)
                    + " |"
                )
                if p in res["fire_pair"] and u and min(row) < LD.AUROC_GATE:
                    below.append(
                        {
                            "bank": b,
                            "class": c,
                            "probe": p,
                            "n_kept": cnt[c]["n_kept"],
                            "best": round(max(row), 3),
                            "worst": round(min(row), 3),
                            "layers_below": [
                                L for L, v in zip(layers, row, strict=True) if v < LD.AUROC_GATE
                            ],
                        }
                    )
            lines.append("")
    res["below_gate_used"] = below
    # capital gold-control pilot (references are non-capital cities in a false-premise template)
    pilot = {}
    for iid, it in sorted(items.items()):
        if bank_of_item(it) != "country_capital":
            continue
        g = LD.lex_gold(it["gold"])
        rows = {}
        for Lyr in layers:
            cell = pr["items"][iid][Lyr]
            if cell.get("na_reason"):
                rows[Lyr] = {"na": cell["na_reason"]}
                continue
            rows[Lyr] = {
                "p_gold": {p: round(cell["p_gold"][p], 3) for p in res["fire_pair"]},
                "z_gold_vs_same": {p: round(cell["z_concept"][p][g], 2) for p in res["fire_pair"]},
                "control_pass": all(cell["p_gold"][p] <= LD.P_GATE for p in res["fire_pair"]),
            }
        pilot[iid] = rows
    res["capital_pilot"] = pilot
    lines += [
        "",
        "## Capital gold-control pilot",
        "",
        "Per capital item and layer: gold control passes (empirical p of the capital's gold score vs other-class "
        "non-capital cities <= 0.05 in both fire-pair probes) and the capital-vs-reference shift "
        "(z of the capital's gold score vs the same-country non-capital cities, "
        + "/".join(res["fire_pair"])
        + ").",
        "",
        "| item | " + " | ".join(f"L{x}" for x in layers) + " |",
        "|---|" + "---|" * len(layers),
    ]
    for iid, rows in pilot.items():
        cells_ = []
        for Lyr in layers:
            r = rows[Lyr]
            if "na" in r:
                cells_.append("n/a")
            else:
                z = "/".join(f"{r['z_gold_vs_same'][p]:+.1f}" for p in res["fire_pair"])
                cells_.append(("✓ " if r["control_pass"] else "✗ ") + z)
        lines.append(f"| {iid} | " + " | ".join(cells_) + " |")
    lines += [
        "",
        "Control pass counts per layer (capital items): "
        + ", ".join(
            f"L{Lyr} {sum(pilot[i][Lyr].get('control_pass', False) for i in pilot)}/{len(pilot)}"
            for Lyr in layers
        ),
    ]
    lines += ["## Used classes below 0.85 at some layer (fire-pair probes)", ""]
    lines += [
        f"- {r['bank']} / {r['class']} ({r['probe']}, n={r['n_kept']}): best {r['best']}, worst {r['worst']}, below at L{','.join(r['layers_below'])}"
        for r in below
    ] or ["- none"]
    (OUT / "probe_quality.json").write_text(json.dumps(res, indent=1) + "\n")
    (OUT / "probe_quality.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(res["by_probe"], indent=1))
    print("fire pair", res["fire_pair"], "| used classes below gate:", len(below))


# ------------------------------------------------------------------ labels

ABBR = {LD.L_NOT: "N", LD.L_PROBE: "P", LD.L_CROSS: "X", LD.L_HALL: "H"}
ALI_README = paths.ALI_PROBE_ROOT / "docs" / "project" / "experiments" / "ali_probe" / "README.md"
# the Ali case study's verdicts (README TL;DR), per extra sport
ALI_VERDICTS = {
    "swimming": "not represented; lens generic-athlete prior -> hallucination",
    "track_and_field": "represented (Ali > all held-out boxers L8-36), via the name; lens agreement is the prior",
    "wrestling": "combat-sport neighbour shared by all boxers; not Ali-specific (no fire vs boxers)",
    "pro_wrestling": "combat-sport neighbour shared by all boxers; not Ali-specific (no fire vs boxers)",
    "basketball": "represented for the person (Ali > >=21/22 boxers L20-60)",
    "american_football": "late-layer quiz distractor; not interpretable",
    "soccer": "late-layer quiz distractor; not interpretable",
    "cricket": "late-layer quiz distractor; not interpretable",
    "tennis": "late-layer quiz distractor; not represented",
    "figure_skating": "not represented",
    "gymnastics": "(not in the README)",
}


def stage_label(args) -> None:
    pr = json.loads((OUT / "probes.json").read_text())
    pq = json.loads((OUT / "probe_quality.json").read_text())
    pair = tuple(pq["fire_pair"])
    items = {x["id"]: x for x in entity_items()}
    units = read_jsonl(OUT / "commit.jsonl")
    group = {iid: bank_of_item(it) for iid, it in items.items()}
    n_items = Counter(group.values())
    named = LD.named_items(units, lambda i: group[i])
    cells = pr["items"]
    # control-concept fire rates per (bank, layer)
    fpr: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for b, body in pr["banks"].items():
        for L in map(str, pr["layers"]):
            st = Counter()
            for iid, it in items.items():
                if group[iid] != b:
                    continue
                for c in body["controls"]:
                    st[
                        LD.fires(cells.get(iid, {}).get(L), c, LD.lex_gold(it["gold"]), pair)[
                            "status"
                        ]
                    ] += 1
            ev = st["fires"] + st["no_fire"]
            rate = st["fires"] / ev if ev else None
            fpr[b][L] = {
                "fires": st["fires"],
                "evaluable": ev,
                "n/a": st["n/a"],
                "rate": rate,
                "weak_null": rate is not None and rate > 0.10,
            }
    by_ic: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for u in units:
        by_ic[(u["item"], u["concept"])].append(u)
    out = []
    for u in units:
        it = items[u["item"]]
        b = group[u["item"]]
        L = LD.eff_layer(u["method"], u["layer"])
        gold = u["gold"]
        f = LD.fires(cells.get(u["item"], {}).get(str(L)), u["concept"], gold, pair)
        pm = [
            LD.fires(cells.get(u["item"], {}).get(str(L2)), u["concept"], gold, pair)
            for L2 in (L - LD.GRID, L + LD.GRID)
            if str(L2) in cells.get(u["item"], {})
        ]
        fire_pm1 = f["fires"] or any(x["fires"] for x in pm)

        def prior_of(m, u=u, b=b):
            s = named.get((m, b, u["concept"]), set()) - {u["item"]}
            return len(s) / (n_items[b] - 1) if n_items[b] > 1 else 0.0

        others = [o for o in by_ic[(u["item"], u["concept"])] if o["method"] != u["method"]]
        cs = LD.cross_support(u, others, prior_of)
        c = u["committed"]
        lab = LD.ladder_label(c, f["fires"], cs["primary"]["support"])
        cell = cells.get(u["item"], {}).get(str(L)) or {}
        probe_view = {}
        if cell and not cell.get("na_reason") and u["concept"] in cell.get("classes", []):
            probe_view = {
                p: {
                    "rank": cell["rank_concept"][p][u["concept"]],
                    "p": cell["p_concept"][p][u["concept"]],
                    "auroc": cell["auroc"][p][u["concept"]],
                    "p_gold": cell["p_gold"][p],
                }
                for p in pair
            }
        weak = lab == LD.L_PROBE and fpr[b].get(str(L), {}).get("weak_null", False)
        rec = {k: v for k, v in u.items() if k != "samples"}
        rec.update(
            {
                "bank": b,
                "eff_layer": L,
                "prompt": it["prompt"],
                "probe": {**f, "pair": list(pair), "view": probe_view, "fire_pm1": fire_pm1},
                "cross": cs,
                "lens_prior_own": round(prior_of(u["method"]), 3),
                "label": lab,
                "weak_null": weak,
                "label_any_layer": LD.ladder_label(c, f["fires"], cs["any_layer"]["support"]),
                "label_no_prior": LD.ladder_label(c, f["fires"], cs["primary"]["support_no_prior"]),
                "label_probe_pm1": LD.ladder_label(c, fire_pm1, cs["primary"]["support"]),
                "label_pair_alt": LD.ladder_label(
                    c,
                    LD.fires(
                        cells.get(u["item"], {}).get(str(L)),
                        u["concept"],
                        gold,
                        tuple(pq["alt_pair"]),
                    )["fires"],
                    cs["primary"]["support"],
                ),
                "samples": [
                    {
                        "si": s["si"],
                        "committed": s["verdict"]["committed"],
                        "form": s["verdict"]["form"],
                        "quote": s["verdict"]["quote"],
                        "why": s["verdict"].get("why"),
                        "fail_closed": s["verdict"]["fail_closed"],
                    }
                    for s in u["samples"]
                ],
            }
        )
        out.append(rec)
    LO = OUT / "ladder"
    write_jsonl(LO / "units.jsonl", out)
    summ = ladder_summary(out, fpr, pr, pq)
    (LO / "summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False) + "\n")
    report = ladder_report(out, summ, items, pr, pq)
    (LO / "report.md").write_text(report)
    print(json.dumps(summ["by_method"], indent=1))
    print(f"[label] -> {LO}")


def ladder_summary(out, fpr, pr, pq) -> dict[str, Any]:
    def table(key, rows):
        t: dict[str, Counter] = defaultdict(Counter)
        for r in rows:
            t[r["method"]][r[key]] += 1
        return {m: {lab: t[m][lab] for lab in LD.LABELS} for m in LD.METHODS if m in t}

    s: dict[str, Any] = {
        "version": "ladder-v1 (plan 0005 rev 2)",
        "judge": LD.COMMIT_VERSION,
        "fire_pair": pq["fire_pair"],
        "n_units": len(out),
        "n_fail_closed_samples": sum(int(r.get("n_fail_closed", 0)) for r in out),
        "by_method": table("label", out),
        "by_method_any_layer": table("label_any_layer", out),
        "by_method_no_prior": table("label_no_prior", out),
        "by_method_probe_pm1": table("label_probe_pm1", out),
        "by_method_pair_alt": table("label_pair_alt", out),
        "weak_null_probe_validated": sum(r["weak_null"] for r in out),
        "probe_status_committed": dict(
            Counter(r["probe"]["status"] for r in out if r["committed"])
        ),
        "probe_na_reasons": dict(
            Counter(
                r["probe"]["reason"].split(" (")[0]
                for r in out
                if r["committed"] and r["probe"]["status"] == "n/a"
            )
        ),
        "control_fpr": fpr,
        "by_method_layer": {},
        "probe_validated_share": {},
    }
    for m in LD.METHODS:
        rows = [r for r in out if r["method"] == m]
        if not rows:
            continue
        s["by_method_layer"][m] = {}
        s["probe_validated_share"][m] = {}
        for L in sorted({r["eff_layer"] for r in rows}):
            rl = [r for r in rows if r["eff_layer"] == L]
            s["by_method_layer"][m][str(L)] = dict(Counter(r["label"] for r in rl))
            com = [r for r in rl if r["committed"]]
            s["probe_validated_share"][m][str(L)] = {
                "n_committed": len(com),
                "n_validated": sum(r["label"] == LD.L_PROBE for r in com),
                "n_probe_evaluable": sum(r["probe"]["status"] != "n/a" for r in com),
            }
    cal = OUT / "calib.json"
    if cal.exists():
        c = json.loads(cal.read_text())
        s["judge_calibration"] = {k: c[k]["unit"] for k in ("gold80", "ali", "all") if k in c}
        s["judge_below_target"] = c.get("below_target")
    # Ali comparison
    ali = [r for r in out if r["item"] == ALI]
    comp = {}
    for concept in sorted({r["concept"] for r in ali}):
        rs = [r for r in ali if r["concept"] == concept]
        comp[concept] = {
            "readme": ALI_VERDICTS.get(concept, "(not in the README)"),
            "labels": {
                m: sorted(f"L{r['eff_layer']}:{ABBR[r['label']]}" for r in rs if r["method"] == m)
                for m in LD.METHODS
                if any(r["method"] == m for r in rs)
            },
            "probe_fires_layers": sorted(
                {r["eff_layer"] for r in rs if r["probe"]["status"] == "fires"}
            ),
        }
    s["ali"] = comp
    return s


def _fmt_probe(r) -> str:
    v = r["probe"]["view"]
    if r["probe"]["status"] == "n/a":
        return f"L{r['eff_layer']} n/a ({r['probe']['reason']})"
    ranks = "/".join(f"{round(float(v[p]['rank']), 3):.3f}" for p in r["probe"]["pair"])
    mark = "fires" if r["probe"]["fires"] else "no"
    return f"L{r['eff_layer']} r={ranks} {mark}"


def ladder_report(out, summ, items, pr, pq) -> str:
    L = [
        "# Plan 0005: basic-readout mention ladder",
        "",
        f"Judge `{LD.COMMIT_VERSION}`; fire pair {' + '.join(pq['fire_pair'])} (probe ranks shown as "
        f"{'/'.join(pq['fire_pair'])} = share of same-gold held-out references below the item); labels: "
        "N not committed, P probe-validated, X cross-method allowed (primary +-1 grid step), H hallucination.",
        "",
    ]
    jc = summ.get("judge_calibration")
    if jc:
        g = jc["gold80"]
        tag = " (below target)" if summ.get("judge_below_target") else ""
        L += [
            f"Judge vs blind gold (80 units): accuracy {g['accuracy']} (Wilson95 {g['wilson95']}){tag}; "
            f"committed P/R {g['committed']['precision']}/{g['committed']['recall']}; "
            f"Ali anchors (60): {jc['ali']['accuracy']}.",
            "",
        ]
    L += [
        "## Counts per method x label (primary rule)",
        "",
        "| method | " + " | ".join(LD.LABELS) + " |",
        "|---|" + "---|" * len(LD.LABELS),
    ]
    for m, row in summ["by_method"].items():
        L.append(f"| {LD.METHOD_LABEL[m]} | " + " | ".join(str(row[x]) for x in LD.LABELS) + " |")
    for key, name in (
        ("by_method_any_layer", "cross-method at any layer"),
        ("by_method_no_prior", "without lens-prior-flagged allowances"),
        ("by_method_probe_pm1", "probe fires at +-1 grid step"),
        ("by_method_pair_alt", "alternative fire pair " + " + ".join(pq["alt_pair"])),
    ):
        L += [
            "",
            f"Sensitivity: {name}",
            "",
            "| method | " + " | ".join(LD.LABELS) + " |",
            "|---|" + "---|" * len(LD.LABELS),
        ]
        for m, row in summ[key].items():
            L.append(
                f"| {LD.METHOD_LABEL[m]} | " + " | ".join(str(row[x]) for x in LD.LABELS) + " |"
            )
    L += [
        "",
        f"Probe status of committed units: {summ['probe_status_committed']}; n/a reasons: {summ['probe_na_reasons']}; "
        f"probe-validated under a weak null (control FPR > 10%): {summ['weak_null_probe_validated']}.",
        "",
    ]
    L += [
        "## Control-concept fire rate (FPR) per bank x layer",
        "",
        "| bank | " + " | ".join(f"L{x}" for x in pr["layers"]) + " |",
        "|---|" + "---|" * len(pr["layers"]),
    ]
    for b, row in summ["control_fpr"].items():
        L.append(
            f"| {b} | "
            + " | ".join(
                (
                    "—"
                    if row[str(x)]["rate"] is None
                    else f"{row[str(x)]['fires']}/{row[str(x)]['evaluable']}"
                    + (" ⚠" if row[str(x)]["weak_null"] else "")
                )
                for x in pr["layers"]
            )
            + " |"
        )
    L += [
        "",
        "## Ali vs the case study (`gw-ali-probe/docs/project/experiments/ali_probe/README.md`)",
        "",
        "| concept | case-study verdict | ladder labels (per method, layer:label) | probe fires at |",
        "|---|---|---|---|",
    ]
    for c, v in summ["ali"].items():
        labs = "; ".join(f"{LD.METHOD_LABEL[m]}: {' '.join(x)}" for m, x in v["labels"].items())
        L.append(f"| {c} | {v['readme']} | {labs} | {v['probe_fires_layers'] or '—'} |")
    L += ["", "## Per-item breakdown", ""]
    for b in ("sport", "country_capital", "country_landmark", "musical_instrument"):
        L += [f"### {b}", ""]
        for iid in sorted(i for i, it in items.items() if bank_of_item(it) == b):
            rs = [r for r in out if r["item"] == iid]
            it = items[iid]
            L += [f'#### {iid}: "{it["prompt"]}" (gold {it["gold"]})', ""]
            if not rs:
                L += ["no mentions", ""]
                continue
            L += [
                "| concept | method | layers | committed / not | example quote | probe (committed units) | cross-method (primary) | labels | lens prior |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
            groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
            for r in rs:
                groups[(r["concept"], r["method"])].append(r)
            for (c, m), g in sorted(groups.items()):
                g.sort(key=lambda r: r["eff_layer"])
                com = [r for r in g if r["committed"]]
                q = next((r["quote"] for r in com if r.get("quote")), None) or next(
                    (r["quote"] for r in g if r.get("quote")), ""
                )
                q = (q or "").replace("|", "/")[:90]
                probe = "; ".join(_fmt_probe(r) for r in com) or "—"
                cross = (
                    "; ".join(
                        f"L{r['eff_layer']}: {','.join(r['cross']['primary']['methods']) or 'none'}"
                        for r in com
                    )
                    or "—"
                )
                labs = " ".join(
                    f"L{r['eff_layer']}:{ABBR[r['label']]}{'*' if r['weak_null'] else ''}"
                    for r in g
                )
                L.append(
                    f"| {c} | {LD.METHOD_LABEL[m]} | {','.join(str(r['eff_layer']) for r in g)} | {len(com)} / {len(g) - len(com)} | {q} | {probe} | {cross} | {labs} | {g[0]['lens_prior_own']:.2f} |"
                )
            L.append("")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ figures

LABEL_COLORS = {
    LD.L_NOT: "#B9C0C9",
    LD.L_PROBE: "#2E7D5B",
    LD.L_CROSS: "#E3A13B",
    LD.L_HALL: "#C4453C",
}


def stage_figures(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = read_jsonl(OUT / "ladder" / "units.jsonl")
    summ = json.loads((OUT / "ladder" / "summary.json").read_text())
    P = REPO_ROOT / "plots" / "sibling_probe"
    P.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"font.size": 14, "axes.titlesize": 16, "axes.labelsize": 15, "legend.fontsize": 13}
    )
    methods = [m for m in LD.METHODS if any(r["method"] == m for r in out)]

    # (a) share of mentions per final label, per method
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, m in enumerate(methods):
        rows = [r for r in out if r["method"] == m]
        n = len(rows)
        left = 0.0
        for lab in LD.LABELS:
            k = sum(r["label"] == lab for r in rows)
            w = k / n
            ax.barh(
                i,
                w,
                left=left,
                color=LABEL_COLORS[lab],
                edgecolor="white",
                label=lab if i == 0 else None,
            )
            if w > 0.04:
                ax.text(
                    left + w / 2, i, str(k), ha="center", va="center", fontsize=12, color="black"
                )
            left += w
        ax.text(1.01, i, f"n={n}", va="center", fontsize=12)
    ax.set_yticks(range(len(methods)), [LD.METHOD_LABEL[m] for m in methods])
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of mention units")
    ax.set_title("Ladder label of every mention unit, per method (primary rule)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.09, 1.0), frameon=False)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(P / "ladder_labels_by_method.png", dpi=150)
    plt.close(fig)

    # (b) the same by layer, one panel per method
    fig, axes = plt.subplots(
        1,
        len(methods),
        figsize=(5.2 * len(methods), 6),
        sharey=True,
        gridspec_kw={
            "width_ratios": [
                max(1, len({r["eff_layer"] for r in out if r["method"] == m})) for m in methods
            ]
        },
    )
    for ax, m in zip(axes, methods, strict=True):
        rows = [r for r in out if r["method"] == m]
        Ls = sorted({r["eff_layer"] for r in rows})
        for j, Lyr in enumerate(Ls):
            rl = [r for r in rows if r["eff_layer"] == Lyr]
            bottom = 0.0
            for lab in LD.LABELS:
                w = sum(r["label"] == lab for r in rl) / len(rl)
                ax.bar(
                    j,
                    w,
                    bottom=bottom,
                    color=LABEL_COLORS[lab],
                    edgecolor="white",
                    width=0.8,
                    label=lab if (m == methods[0] and j == 0) else None,
                )
                bottom += w
            ax.text(j, 1.01, str(len(rl)), ha="center", va="bottom", fontsize=10)
        ax.set_xticks(range(len(Ls)), [str(x) for x in Ls], rotation=90 if len(Ls) > 3 else 0)
        ax.set_title(LD.METHOD_LABEL[m], pad=18)
        ax.set_xlabel("layer")
        ax.set_ylim(0, 1.08)
    axes[0].set_ylabel("share of mention units (n on top)")
    fig.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.suptitle("Ladder labels by layer", y=1.0)
    fig.tight_layout()
    fig.savefig(P / "ladder_labels_by_layer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # (c) share of committed mentions the probe validates, per method and layer
    fig, ax = plt.subplots(figsize=(12, 5.5))
    width = 0.8 / max(1, len(methods))
    allL = sorted({r["eff_layer"] for r in out})
    xi = {Lyr: i for i, Lyr in enumerate(allL)}
    colors = {
        "s3d": "#3B6EA8",
        "mlnla-rl600": "#8C5FB0",
        "jlens": "#4E9A8E",
        "nla-rl-L42": "#B5651D",
    }
    for k, m in enumerate(methods):
        sh = summ["probe_validated_share"].get(m, {})
        labelled = False
        for Lyr, v in sh.items():
            n = v["n_committed"]
            if not n:
                continue
            x = xi[int(Lyr)] + (k - (len(methods) - 1) / 2) * width
            ax.bar(
                x,
                v["n_validated"] / n,
                width=width,
                color=colors[m],
                label=None if labelled else LD.METHOD_LABEL[m],
            )
            ax.text(
                x,
                v["n_validated"] / n + 0.01,
                f"{v['n_validated']}/{n}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )
    ax.set_xticks(range(len(allL)), [str(x) for x in allL])
    ax.set_xlabel("layer")
    ax.set_ylabel("probe-validated / committed")
    ax.set_ylim(0, 1.15)
    ax.set_title(
        "Share of committed mentions the probe validates (fire pair "
        + " + ".join(summ["fire_pair"])
        + ")"
    )
    h, lab = ax.get_legend_handles_labels()
    uniq = dict(zip(lab, h, strict=True))
    ax.legend(
        list(uniq.values()),
        list(uniq.keys()),
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(P / "ladder_probe_validation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] -> {P}/ladder_*.png")


# ------------------------------------------------------------------ main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--run", default="ladder1")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--exclude-runs", default="")
    ap.add_argument(
        "--fire-pair", default="logreg,dom", help="probe-report: the two probes of the fire rule"
    )
    ap.add_argument(
        "--modal",
        action="store_true",
        help="probes: fit on Modal CPU containers (acts on the sibling-probes volume)",
    )
    args = ap.parse_args()
    if args.stage == "lexicons":
        asyncio.run(stage_lexicons())
    elif args.stage == "mentions":
        stage_mentions()
    else:
        fn = globals().get("stage_" + args.stage.replace("-", "_"))
        if fn is None:
            sys.exit(f"unknown stage {args.stage}")
        r = fn(args)
        if asyncio.iscoroutine(r):
            asyncio.run(r)


if __name__ == "__main__":
    main()
