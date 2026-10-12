"""Set B (plan 0004): the WorkspaceBench ``basic_readout`` items as a second sibling bank.

``data/sibling_probe/basic_readout_bank.json``::

    {"version", "provenance", "counts",
     "items": [{"id" (label), "name", "subfamily": entity|processing|implicit, "prompt",
                "target", "eval_render": plain|chat_prefill, "prefill", "input_ids", "tokens",
                "pos" (the read site: the final prompt token), "response" (greedy raw
                continuation, 48 new tokens), "category", "gold", "gold_label", "categorical",
                "in_pipeline" (entity, or implicit with a categorical gold), "template",
                "template_idx"}]}

Only ``in_pipeline`` items enter the sibling pipeline; processing items are viewer-only
(plan rev 2 item 1). Three cached Opus calls (``claude-opus-5-5``, effort low), each with its
own prompt version and cache file under ``outputs/sibling/br/``:

* ``brcat-v1``: (prompt, target) -> category name + gold concept + categorical flag, per item;
* ``brinv-v1``: per category, a fixed list of 15 common members with aliases, generated
  BEFORE any claim is mapped (category-first inventory, plan rev 2 item 3 / rev 4 item 6);
* ``brnorm-v1``: a set-B claim (probe_target or the segment) -> one inventory concept of the
  item's category, ``none``, or an entity substitution (the readout swaps the prompt's entity).
* ``brtext-v1``: probe texts ending with the eval prompts' own templates (plan rev 2 item 2).
"""

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from hallucination_bench import DATA, REPO_ROOT
from hallucination_bench.llm import schema_block

WSB_BR_ITEMS = Path("/workspace/root/wsbench-pub/evals/basic_readout/items.json")
BANK_PATH = DATA / "sibling_probe" / "basic_readout_bank.json"
CONCEPTS_PATH = DATA / "sibling_probe" / "br_concepts.json"
PROBE_TEXTS_PATH = DATA / "sibling_probe" / "br_probe_texts.jsonl"
BR = REPO_ROOT / "outputs" / "sibling" / "br"
ANSWER_STEM = "My one-word answer is:"  # wsbench.readplan.ANSWER_STEM (implicit items)
RESPONSE_MAX_NEW = 48
N_MEMBERS = 15
_LABEL_BAD = re.compile(r"[^A-Za-z0-9._-]")


def label_of(name: str) -> str:
    """``wsbench.banks.label_of``: characters outside ``[A-Za-z0-9._-]`` -> ``_``."""
    return _LABEL_BAD.sub("_", name)


def load_wsb_items(path: Path = WSB_BR_ITEMS) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text())["items"]


def render_ids(tokenizer: Any, item: Mapping[str, Any]) -> list[int]:
    """The wsbench eval render of a basic_readout item (``wsbench.produce.render``): plain text
    for entity / processing; the chat template (thinking off) + the prefilled ANSWER_STEM for
    implicit items."""
    if item.get("eval_render", "plain") == "plain":
        return list(tokenizer(item["prompt"], add_special_tokens=False)["input_ids"])
    out = tokenizer.apply_chat_template(
        [{"role": "user", "content": item["prompt"]}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = list(out["input_ids"] if hasattr(out, "keys") else out)
    return ids + list(tokenizer(ANSWER_STEM, add_special_tokens=False)["input_ids"])


def alignment_problems(
    bank: Sequence[Mapping[str, Any]], rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[str]:
    """Plan 0004 Set-B check: every readout row of every arm sits at the item's read site
    (``pos == len(input_ids) - 1``) and, when the row carries ``token``, the decoded token there
    equals the prompt's last token."""
    by_id = {it["id"]: it for it in bank}
    out = []
    for arm, rows in rows_by_arm.items():
        for r in rows:
            it = by_id.get(r["id"])
            if it is None:
                out.append(f"{arm}: unknown id {r['id']}")
                continue
            if int(r["pos"]) != int(it["pos"]):
                out.append(f"{arm}: {r['id']} L{r['layer']} pos {r['pos']} != {it['pos']}")
            elif "token" in r and r["token"] is not None and r["token"] != it["tokens"][-1]:
                out.append(
                    f"{arm}: {r['id']} L{r['layer']} token {r['token']!r} != {it['tokens'][-1]!r}"
                )
    return out


def judge_prompt(item: Mapping[str, Any]) -> str:
    """The prompt shown to the claim map / verdict judge (the user turn for implicit items)."""
    return item["prompt"]


def response_text(item: Mapping[str, Any]) -> str:
    """The text after the read site: the greedy raw continuation (implicit items: the
    continuation after the prefilled answer stem)."""
    return item.get("response") or ""


def shown_prefix(item: Mapping[str, Any]) -> str:
    """Prefilled text that precedes the read site inside the reply (implicit items)."""
    return ANSWER_STEM if item.get("eval_render") == "chat_prefill" else ""


# ------------------------------------------------------------------ brcat-v1

BRCAT_VERSION = "brcat-v1"
BRCAT_SYSTEM = (
    "You name the category of the answer to a short prompt. The prompt is a sentence to complete "
    "or a question; its expected answer (TARGET) is given. Output only the JSON verdict."
)
BRCAT_USER = (
    "Prompt: {prompt}\nTARGET: {target}\n\n"
    "1. category: the kind of thing the TARGET is, as a short lowercase singular noun phrase in "
    "snake_case that names the set of alternatives a reader could have answered instead (e.g. "
    "sport, country, musical_instrument, card_game, color, flower, fruit, programming_language). "
    "Use the most natural everyday category: a country is `country` whether the prompt asks for "
    "a capital's country, a landmark's country or a favourite country.\n"
    "2. gold: the TARGET as a canonical member of that category, in snake_case (e.g. boxing, "
    "american_football if 'football' clearly means American football, united_kingdom for "
    "England only if the category is country and England is not itself a member; otherwise keep "
    "it as given).\n"
    "3. gold_label: a short display name for the gold (e.g. 'Boxing').\n"
    "4. categorical: true if the TARGET is one member of a set of named alternatives of the same "
    "kind (sports, countries, instruments, flowers, ...); false for numbers, dates, days or months "
    "computed by arithmetic, spelled-out numbers, or a free-text phrase."
)
BRCAT_SCHEMA = schema_block(
    "brcat",
    {
        "category": {"type": "string"},
        "gold": {"type": "string"},
        "gold_label": {"type": "string"},
        "categorical": {"type": "boolean"},
    },
    ["category", "gold", "gold_label", "categorical"],
)


def snake(s: str) -> str:
    s = re.sub(r"[^0-9a-z]+", "_", s.strip().lower())
    return s.strip("_")


def render_brcat(prompt: str, target: str) -> tuple[str, str]:
    return BRCAT_SYSTEM, BRCAT_USER.replace("{prompt}", prompt).replace("{target}", target)


def valid_brcat(r: Mapping[str, Any]) -> bool:
    return (
        isinstance(r.get("category"), str)
        and bool(snake(r["category"]))
        and isinstance(r.get("gold"), str)
        and bool(snake(r["gold"]))
        and isinstance(r.get("categorical"), bool)
    )


# ------------------------------------------------------------------ brinv-v1

BRINV_VERSION = "brinv-v1"
BRINV_SYSTEM = (
    "You list members of a category for an interpretability study. Output only the JSON answer."
)
BRINV_USER = (
    "Category: {category} (as in the prompts: {examples}).\n\n"
    "List the {n} most common, distinct members of this category that a reader might name as "
    "the answer. Include these members: {golds}. Each member is one specific instance (no "
    "generic or umbrella terms, no two members that are synonyms). For each member give:\n"
    "- id: snake_case canonical name;\n"
    "- label: display name;\n"
    "- aliases: the English words or short phrases that name it in running text (name, "
    "common variants, adjective/demonym if it names the member itself), plus its Chinese name. "
    "No alias may name another member."
)
BRINV_SCHEMA = schema_block(
    "brinv",
    {
        "members": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "label", "aliases"],
                "additionalProperties": False,
            },
        }
    },
    ["members"],
)


def render_brinv(category: str, golds: Sequence[str], examples: Sequence[str]) -> tuple[str, str]:
    user = (
        BRINV_USER.replace("{category}", category.replace("_", " "))
        .replace("{examples}", "; ".join(examples[:4]))
        .replace("{n}", str(N_MEMBERS))
        .replace("{golds}", ", ".join(golds))
    )
    return BRINV_SYSTEM, user


# ------------------------------------------------------------------ brnorm-v1

BRNORM_VERSION = "brnorm-v1"
BRNORM_SYSTEM = (
    "You map one claim from an interpretability-lens readout to a fixed list of category "
    "members. The lens read a language model's activation at the end of a prompt; the claim "
    "names something the readout says the model is thinking of. Output only the JSON verdict."
)
BRNORM_USER = (
    "Prompt: {prompt}\nCategory: {category}\nMembers (id: label):\n{members}\n\n"
    "Claim segment: {segment}\nClaimed target (may be empty): {target}\n\n"
    "Decide what the claim names:\n"
    '- kind "member": it names one of the listed members as the answer -> concept = that id;\n'
    '- kind "out_of_inventory": it names a specific member of the category that is not '
    "listed -> concept = a snake_case name for it;\n"
    '- kind "entity_substitution": it replaces the entity the prompt is about (the person, '
    "city, landmark, ...) with a different one (e.g. the prompt names Muhammad Ali, the claim "
    "talks about Usain Bolt) -> concept = snake_case name of the substituted entity;\n"
    '- kind "none": it names no member of the category -> concept = "none".'
)
BRNORM_SCHEMA = schema_block(
    "brnorm",
    {
        "kind": {
            "type": "string",
            "enum": ["member", "out_of_inventory", "entity_substitution", "none"],
        },
        "concept": {"type": "string"},
    },
    ["kind", "concept"],
)


def render_brnorm(
    prompt: str, category: str, members: Sequence[tuple[str, str]], segment: str, target: str
) -> tuple[str, str]:
    mem = "\n".join(f"{i}: {lab}" for i, lab in members)
    vals = {
        "prompt": prompt,
        "category": category.replace("_", " "),
        "members": mem,
        "segment": segment,
        "target": target,
    }
    user = BRNORM_USER
    for k in vals:
        user = user.replace("{" + k + "}", "\0" + k + "\0")
    for k, v in vals.items():
        user = user.replace("\0" + k + "\0", v)
    return BRNORM_SYSTEM, user


def valid_brnorm(r: Mapping[str, Any], member_ids: Sequence[str]) -> bool:
    k, c = r.get("kind"), r.get("concept")
    if not isinstance(c, str):
        return False
    if k == "member":
        return c in set(member_ids)
    return k in ("out_of_inventory", "entity_substitution", "none")


# ------------------------------------------------------------------ brtext-v1

BRTEXT_VERSION = "brtext-v1"
BRTEXT_SYSTEM = (
    "You write short plain-text passages for an interpretability study. Output only the JSON "
    "answer."
)


def render_brtext(
    category: str,
    template: str,
    kind: str,
    k: int,
    label: str | None,
    aliases: Sequence[str],
    others: Sequence[str],
) -> tuple[str, str]:
    """Probe-text prompt. Each passage is context only and must end so that the fixed template
    (written with NAME for an entity) follows naturally; the code appends the template."""
    noun = category.replace("_", " ")
    if kind == "neutral":
        rule = (
            f"The passages must have nothing to do with any {noun}: never name or hint at any "
            f"{noun} (e.g. {', '.join(others[:12])}), so that the sentence after them could be "
            "completed with any member."
        )
    elif kind == "explicit":
        rule = (
            f"Each passage must explicitly name the {noun} {label!r} (using one of: "
            f"{', '.join(aliases[:6])}) as the answer the closing sentence is about. Never name "
            f"any other {noun} (none of: {', '.join(others)})."
        )
    else:
        rule = (
            f"Each passage must make the {noun} {label!r} clearly inferable as the answer the "
            f"closing sentence is about WITHOUT naming it: never use any of: "
            f"{', '.join(aliases[:10])}; and never name any other {noun} (none of: "
            f"{', '.join(others)})."
        )
    user = (
        f"Write {k} distinct short passages of plain text (not a chat message).\n{rule}\n\n"
        "Constraints:\n"
        "- 1 to 3 sentences, 15 to 50 words, ending with '.'.\n"
        "- Each passage is context only. It will be followed by this exact closing sentence, which "
        f'you must NOT include: "{template}" where NAME is replaced by a fictional person or '
        "thing you introduce in the passage (give them a name or refer to them so the closing "
        "sentence reads naturally). The closing sentence stops mid-sentence; the passage must "
        "make its completion natural.\n"
        "- Vary the setting, register and invented names widely; no lists, no quotation marks "
        "around the whole passage.\n\n"
        'Return {"texts": [{"text": "...", "name": "..."}, ...]} where name is what replaces NAME '
        '(use "" if the closing sentence has no NAME).'
    )
    return BRTEXT_SYSTEM, user


BRTEXT_SCHEMA = schema_block(
    "brtext",
    {
        "texts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"text": {"type": "string"}, "name": {"type": "string"}},
                "required": ["text", "name"],
                "additionalProperties": False,
            },
        }
    },
    ["texts"],
)


# ------------------------------------------------------------------ templates


ENTITY_TEMPLATES: tuple[tuple[str, str], ...] = (
    # (regex on the prompt, template with NAME); the template is the prompt's own frame
    (r"^The athlete (.+) plays the sport of$", "The athlete NAME plays the sport of"),
    (r"^The city of (.+) is the capital of$", "The city of NAME is the capital of"),
    (r"^The (.+) is located in the country of$", "The NAME is located in the country of"),
    (
        r"^The musician (.+) is famous for playing the$",
        "The musician NAME is famous for playing the",
    ),
)


def template_of(item: Mapping[str, Any]) -> tuple[str, str | None]:
    """(template, entity) of a set-B item: the entity frame for entity items; the question +
    answer stem (no entity) for implicit items."""
    if item.get("eval_render") == "chat_prefill":
        return f"{item['prompt']}\n{ANSWER_STEM}", None
    for rx, tpl in ENTITY_TEMPLATES:
        m = re.match(rx, item["prompt"])
        if m:
            return tpl, m.group(1)
    return item["prompt"], None


def load_bank(path: Path = BANK_PATH) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text())["items"]


def load_br_concepts(path: Path = CONCEPTS_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


# ------------------------------------------------------------------ alias matching


def alias_regex(aliases: Sequence[str]) -> re.Pattern[str] | None:
    parts = []
    for a in sorted({a.strip() for a in aliases if a and a.strip()}, key=len, reverse=True):
        if re.search(r"[㐀-鿿]", a):
            parts.append(re.escape(a))
        else:
            parts.append(r"(?<![A-Za-z0-9])" + re.escape(a) + r"(?![A-Za-z0-9])")
    return re.compile("|".join(parts), re.IGNORECASE) if parts else None


def named_in(text: str, aliases: Sequence[str]) -> bool:
    rx = alias_regex(aliases)
    return bool(rx and rx.search(text or ""))


# ------------------------------------------------------------------ judge wiring


def token_aliases(tokenizer: Any, aliases: Sequence[str]) -> list[str]:
    """Single-token spellings (raw vocabulary form not needed: the matcher compares the decoded,
    stripped, lower-cased token) of each alias: ``alias`` / `` alias`` / capitalised."""
    out: set[str] = set()
    for a in aliases:
        a = a.strip()
        if not a:
            continue
        for v in {a, a.lower(), a.capitalize()}:
            for s in (v, " " + v):
                if len(tokenizer.encode(s, add_special_tokens=False)) == 1:
                    out.add(s.strip())
    return sorted(out)


def categories(doc: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``br_concepts.json`` -> {category: concepts.Category} (for :class:`concepts.Matcher`)."""
    from hallucination_bench.evals.sibling_probe.concepts import Category, Concept

    doc = doc or load_br_concepts()
    out = {}
    for name, d in doc["categories"].items():
        cs = tuple(
            Concept(
                id=c["id"],
                label=c["label"],
                category=name,
                gold=bool(c.get("gold")),
                aliases=tuple(c["aliases"]),
                exclude={},
                token_aliases=tuple(c.get("token_aliases") or ()),
                token_eligible=bool(c.get("token_aliases")),
            )
            for c in d["concepts"]
        )
        out[name] = Category(name=name, question="", requests=tuple(d["templates"]), concepts=cs)
    return out


def judge_items(bank: Sequence[Mapping[str, Any]], cats: Mapping[str, Any]) -> list[dict]:
    """Set-B items in the shape ``judge.process_cell`` takes: the prompt as the user turn, the
    marked response (implicit items: answer stem + marker + continuation), category, gold
    concept id, and the concepts named in the prompt / response."""
    from hallucination_bench.evals.sibling_probe.concepts import Matcher
    from hallucination_bench.judges.v5c import READ_MARK

    out = []
    for it in bank:
        if not it.get("in_pipeline"):
            continue
        cat = cats[it["category"]]
        m = Matcher(cat)
        resp = response_text(it)
        out.append(
            {
                "id": it["id"],
                "category": it["category"],
                "gold": it["gold_concept"],
                "messages": [{"role": "user", "content": judge_prompt(it)}],
                "response": resp,
                "marked_response": shown_prefix(it) + READ_MARK + resp,
                "named": sorted(m.mentions(it["prompt"])),
            }
        )
    return out
