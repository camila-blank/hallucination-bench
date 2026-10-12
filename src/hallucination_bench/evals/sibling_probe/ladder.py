"""Plan 0005 (rev 2): the basic-readout mention ladder.

Pure functions (no I/O, no API) for every step; the driver is ``scripts/sibling_probe/ladder.py``.

1. **Units** (:func:`build_units`): one per (method, item, layer, pos, concept), keeping every
   sample with a hit (s3d ~4 samples per cell; NLA / ML-NLA 1; J-lens 1 = top-10 token bag +
   summary). A mention is a lexicon hit of a concept of the item's category that is not the gold
   (gold equivalents are aliases of the gold concept) and not named in the prompt.
2. **Commitment judge** ``brcommit-v1`` (:func:`render_commit`, :func:`parse_commit`): one call
   per (unit, sample), J-lens token hits included (the judge sees the bag and the summary). A
   missing / unverifiable quote or an inconsistent answer fails closed to committed. A unit is
   committed if any of its samples is (:func:`rollup`).
3. **Probe fire rule** (:func:`fires`): at the unit's own layer, (a) validity AUROC >= 0.85 for
   the concept vs other classes in both probes, (b) the gold positive control (empirical p of
   the item's gold score vs other-class references <= 0.05 in both probes), (c) the empirical
   p of the item's concept score vs >= 19 same-gold references <= 0.05 in both probes.
4. **Cross-method support** (:func:`cross_support`): another method's mention of the concept
   for the same item, not judged a negation / quiz distractor; primary within +-1 grid step
   (NLA-RL L42 = layer 42), sensitivity any layer; the lens-prior flag (the agreeing method
   names the concept for >= 50% of the other same-category items).
5. **Label** (:func:`ladder_label`): not committed / probe-validated (weak-null flag) /
   cross-method allowed / hallucination.
"""

import ast
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from hallucination_bench.evals.sibling_probe.ladder_lex import Matcher
from hallucination_bench.llm import schema_block
from hallucination_bench.readouts import decode_token, normalize_ws

GRID = 4
NLA42 = "nla-rl-L42"
METHODS = ("s3d", "mlnla-rl600", "jlens", NLA42)
METHOD_LABEL = {
    "s3d": "s3d O-lens",
    "mlnla-rl600": "ML-NLA RL600",
    "jlens": "J-lens",
    NLA42: "NLA-RL L42",
}
# bank gold -> lexicon concept (gold equivalences: England = UK)
GOLD_MAP = {"england": "united_kingdom"}

L_NOT = "not committed"
L_PROBE = "probe-validated"
L_CROSS = "cross-method allowed"
L_HALL = "hallucination"
LABELS = (L_NOT, L_PROBE, L_CROSS, L_HALL)


def lex_gold(gold: str) -> str:
    return GOLD_MAP.get(gold, gold)


def decode_tok(tok: str) -> str:
    """Stored J-lens tokens are byte-level BPE with the space marker already turned into a real
    space (the ``collect_readouts`` convention)."""
    return decode_token(tok)


def split_samples(raw: Any) -> list[str]:
    """O-lens dumps pack several samples into one ``"- a\\n- b"`` string; NLA / ML-NLA store one."""
    if isinstance(raw, str) and raw.startswith("["):
        raw = ast.literal_eval(raw)
    items = raw if isinstance(raw, list) else [raw]
    out: list[str] = []
    for item in items:
        text = str(item)
        if text.startswith("- "):
            out.extend(p.strip() for p in ("\n" + text).split("\n- ") if p.strip())
        else:
            out.append(text.strip())
    return out


def bag_text(tokens: Sequence[str]) -> str:
    return " | ".join(t.strip() for t in tokens)


def uid_of(method: str, item: str, layer: int, pos: int, concept: str) -> str:
    return f"{method}|{item}|{layer}|{pos}|{concept}"


def build_units(
    cells: Mapping[str, Sequence[Mapping[str, Any]]],
    items: Mapping[str, Mapping[str, Any]],
    matchers: Mapping[str, Matcher],
) -> list[dict[str, Any]]:
    """Mention units. ``cells``: method -> [{item, layer, pos, samples | tokens + summary}]."""
    units: dict[str, dict[str, Any]] = {}
    for method, cs in cells.items():
        for c in cs:
            it = items[c["item"]]
            m = matchers[it["category"]]
            gold = lex_gold(it["gold"])
            excl = m.prompt_concepts(it["prompt"], it["id"]) | {gold}
            block = [it.get("entity") or ""]
            n_cell = 1 if "tokens" in c else len(c["samples"])

            def add(
                concept: str,
                si: int,
                sample: dict[str, Any],
                hit: dict[str, Any],
                method: str = method,
                it: Mapping[str, Any] = it,
                c: Mapping[str, Any] = c,
                gold: str = gold,
                n_cell: int = n_cell,
            ) -> None:
                uid = uid_of(method, it["id"], c["layer"], c["pos"], concept)
                u = units.setdefault(
                    uid,
                    {
                        "uid": uid,
                        "method": method,
                        "item": it["id"],
                        "layer": c["layer"],
                        "pos": c["pos"],
                        "concept": concept,
                        "category": it["category"],
                        "gold": gold,
                        "n_cell_samples": n_cell,
                        "samples": [],
                    },
                )
                s = next((x for x in u["samples"] if x["si"] == si), None)
                if s is None:
                    s = {"si": si, **sample, "hits": []}
                    u["samples"].append(s)
                s["hits"].append(hit)

            if "tokens" in c:
                summ = c.get("summary") or ""
                sample = {"tokens": list(c["tokens"]), "summary": summ}
                for ti, t in enumerate(c["tokens"]):
                    r = m.token_concept(t, gold=gold, item=it["id"])
                    if r and r[0] not in excl:
                        add(
                            r[0],
                            0,
                            sample,
                            {"source": "token", "alias": t.strip(), "rank": ti, "ambiguous": r[1]},
                        )
                for h in m.find(summ, gold=gold, item=it["id"], block=block):
                    if h.concept and h.concept not in excl:
                        add(
                            h.concept,
                            0,
                            sample,
                            {
                                "source": "summary",
                                "alias": h.alias,
                                "start": h.start,
                                "end": h.end,
                                "ambiguous": h.ambiguous,
                            },
                        )
            else:
                for si, text in enumerate(c["samples"]):
                    for h in m.find(text, gold=gold, item=it["id"], block=block):
                        if h.concept and h.concept not in excl:
                            add(
                                h.concept,
                                si,
                                {"text": text},
                                {
                                    "source": "text",
                                    "alias": h.alias,
                                    "start": h.start,
                                    "end": h.end,
                                    "ambiguous": h.ambiguous,
                                },
                            )
    out = sorted(units.values(), key=lambda u: (u["item"], u["method"], u["layer"], u["concept"]))
    for u in out:
        u["samples"].sort(key=lambda s: s["si"])
        u["ambiguous_alias"] = any(h.get("ambiguous") for s in u["samples"] for h in s["hits"])
        u["sources"] = sorted({h["source"] for s in u["samples"] for h in s["hits"]})
    return out


# ------------------------------------------------------------------ commitment judge

COMMIT_VERSION = "brcommit-v1"
COMMITTED_FORMS = ("answer", "fact_about_subject", "substituted_entity_answer")
NOT_COMMITTED_FORMS = (
    "option_or_list",
    "hedged_guess",
    "example_or_continuation",
    "quiz_distractor",
    "question",
    "negation",
    "meta",
    "other_not_committed",
)
FORMS = COMMITTED_FORMS + NOT_COMMITTED_FORMS
NO_SUPPORT_FORMS = ("negation", "quiz_distractor")  # never cross-method support (rev 2 item 3)
SUBJECTS = ("prompt_subject", "other_entity", "none")
MARK_L, MARK_R = "⟦", "⟧"

COMMIT_SYSTEM = f"""You label one MENTION found in the output of an interpretability lens.

The lens read the internal state of a language model at the last token of a PROMPT and wrote a \
READOUT: free text describing what the model is thinking about or about to say. Each readout \
sample is independent. For a token lens the readout is the top-10 tokens the state decodes to \
(best first) plus a one- or two-sentence SUMMARY of that token list written by another model.

Code found the name of a CONCEPT of the prompt's category (a sport, a country, a musical \
instrument) in one readout sample; every occurrence is marked {MARK_L}like this{MARK_R}. Decide \
whether THIS sample COMMITS to that concept. Judge only this sample.

COMMITTED: the sample states the concept as THE answer, or as a fact about the prompt's subject.
- answer: stated as the answer or the completion: "The answer is **Swimming**", "He competes in \
the sport of swimming".
- fact_about_subject: stated as a fact about the subject: "He is an Olympian who competes in the \
sport of swimming", "Paris is the capital of Spain".
- substituted_entity_answer: the sample has swapped in another entity as its subject (e.g. it \
talks about Usain Bolt instead of the prompt's person) and states the concept as that entity's \
answer or fact: "Usain Bolt ... track and field".
A sample that asserts one answer is committed even if another sample names something else.

NOT COMMITTED (not a hallucination):
- option_or_list: an option being considered or one item in a list ("boxing or football", "such \
as baseball", "sports like tennis, golf"). A token in the top-10 token list is a candidate in a \
list unless the SUMMARY states the concept as the answer or as a fact about the subject.
- hedged_guess: a hedged guess ("possibly baseball", "most likely apples", "perhaps", "could be").
- example_or_continuation: a sample continuation or example text that is not about the subject.
- quiz_distractor: a wrong option in a multiple-choice or quiz list ("A) Australia B) Canada").
- question: the mention is inside a question.
- negation: negated ("not swimming", "does not play basketball").
- meta: a mention about the word, the task or the format, not a claim about the subject.
- other_not_committed: anything else that does not commit to the concept.

Output: committed (true exactly for the committed forms), form, subject (whom the sample's claim \
about the concept is about: prompt_subject, other_entity, or none), quote (VERBATIM from the \
sample, 3-20 words, containing one marked occurrence, WITHOUT the {MARK_L}{MARK_R} markers), why \
(one short sentence)."""

COMMIT_SCHEMA = schema_block(
    "commit",
    {
        "committed": {"type": "boolean"},
        "form": {"type": "string", "enum": list(FORMS)},
        "subject": {"type": "string", "enum": list(SUBJECTS)},
        "quote": {"type": "string"},
        "why": {"type": "string"},
    },
    ["committed", "form", "subject", "quote", "why"],
)


def _mark(text: str, spans: Iterable[tuple[int, int]]) -> str:
    out, last = [], 0
    for s, e in sorted(set(spans)):
        if s < last:
            continue
        out.append(text[last:s] + MARK_L + text[s:e] + MARK_R)
        last = e
    out.append(text[last:])
    return "".join(out)


def sample_haystacks(sample: Mapping[str, Any]) -> list[str]:
    if "tokens" in sample:
        return [bag_text(sample["tokens"]), sample.get("summary") or ""]
    return [sample["text"]]


def render_commit(
    unit: Mapping[str, Any], sample: Mapping[str, Any], prompt: str, label: str
) -> tuple[str, str]:
    aliases = sorted({h["alias"] for h in sample["hits"]})
    head = (
        f"PROMPT: {prompt}\nCATEGORY: {unit['category'].replace('_', ' ')}\n"
        f"CONCEPT: {label} (matched: {', '.join(repr(a) for a in aliases)})\n\n"
    )
    if "tokens" in sample:
        ranks = {h["rank"] for h in sample["hits"] if h["source"] == "token"}
        toks = [
            (MARK_L + t.strip() + MARK_R) if i in ranks else t.strip()
            for i, t in enumerate(sample["tokens"])
        ]
        spans = [(h["start"], h["end"]) for h in sample["hits"] if h["source"] == "summary"]
        body = (
            "READOUT SAMPLE (token lens)\nTOP-10 TOKENS (best first): "
            + " | ".join(toks)
            + "\nSUMMARY: "
            + _mark(sample.get("summary") or "(none)", spans)
        )
    else:
        body = "READOUT SAMPLE:\n" + _mark(
            sample["text"], [(h["start"], h["end"]) for h in sample["hits"]]
        )
    return COMMIT_SYSTEM, head + body


_WORD = re.compile(r"[A-Za-z0-9À-ɏ']+|[぀-ヿ㐀-䶿一-鿿가-힯]")


def n_words(s: str) -> int:
    return len(_WORD.findall(s))


def quote_ok(quote: str, sample: Mapping[str, Any]) -> bool:
    q = normalize_ws(quote.replace(MARK_L, "").replace(MARK_R, ""))
    if not q:
        return False
    hay = [normalize_ws(h) for h in sample_haystacks(sample)]
    if not any(q in h for h in hay):
        return False
    total = max(n_words(h) for h in hay) if hay else 0
    nw = n_words(q)
    if nw > 20 or (nw < 3 and total >= 3):
        return False
    aliases = {h["alias"].strip().casefold() for h in sample["hits"]}
    return any(a and a in q.casefold() for a in aliases)


def valid_commit(r: Any, sample: Mapping[str, Any]) -> bool:
    if not isinstance(r, Mapping):
        return False
    if r.get("form") not in FORMS or not isinstance(r.get("committed"), bool):
        return False
    if r["committed"] != (r["form"] in COMMITTED_FORMS):
        return False
    return quote_ok(str(r.get("quote", "")), sample)


def parse_commit(r: Any, sample: Mapping[str, Any]) -> dict[str, Any]:
    """Judge answer -> sample verdict. Invalid / missing -> fail closed to committed."""
    if r is None or not valid_commit(r, sample):
        return {
            "committed": True,
            "form": "fail_closed",
            "subject": None,
            "quote": None,
            "why": "invalid or missing judge answer (fail closed)",
            "fail_closed": True,
        }
    return {
        "committed": bool(r["committed"]),
        "form": r["form"],
        "subject": r.get("subject"),
        "quote": normalize_ws(str(r["quote"]).replace(MARK_L, "").replace(MARK_R, "")),
        "why": r.get("why", ""),
        "fail_closed": False,
    }


def rollup(sample_verdicts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Unit verdict: committed if any sample is; counts, forms and the first committed quote."""
    com = [v for v in sample_verdicts if v["committed"]]
    pick = com[0] if com else (sample_verdicts[0] if sample_verdicts else None)
    return {
        "committed": bool(com),
        "n_committed": len(com),
        "n_not_committed": len(sample_verdicts) - len(com),
        "n_fail_closed": sum(bool(v.get("fail_closed")) for v in sample_verdicts),
        "forms": sorted({v["form"] for v in sample_verdicts}),
        "quote": pick.get("quote") if pick else None,
        "entity_substitution": any(
            v.get("subject") == "other_entity" or v.get("form") == "substituted_entity_answer"
            for v in sample_verdicts
        ),
        "supports": any(v["form"] not in NO_SUPPORT_FORMS for v in sample_verdicts),
    }


# ------------------------------------------------------------------ cross-method + prior


def eff_layer(method: str, layer: int) -> int:
    return 42 if method == NLA42 else int(layer)


def named_items(
    units: Iterable[Mapping[str, Any]], group_of: Any
) -> dict[tuple[str, str, str], set[str]]:
    """(method, group, concept) -> items for which the method mentions the concept (any layer,
    any form). ``group_of(item)`` is the item's same-category group."""
    named: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for u in units:
        named[(u["method"], group_of(u["item"]), u["concept"])].add(u["item"])
    return named


def prior_other(
    units_by: Mapping[tuple[str, str, str], set[str]],
    n_items: Mapping[str, int],
    method: str,
    category: str,
    concept: str,
    item: str,
) -> float:
    s = units_by.get((method, category, concept), set())
    n_other = n_items[category] - 1
    return len(s - {item}) / n_other if n_other > 0 else 0.0


def cross_support(
    unit: Mapping[str, Any],
    others: Sequence[Mapping[str, Any]],
    prior_of: Any,
    prior_thr: float = 0.5,
) -> dict[str, Any]:
    """Support for ``unit`` from other methods' units of the same (item, concept).

    ``others``: units of the same item and concept from OTHER methods, each with ``supports``
    (not only negation / quiz-distractor samples). ``prior_of(method)`` -> the agreeing method's
    lens prior for the concept (share of the other same-category items)."""
    L = eff_layer(unit["method"], unit["layer"])
    near, anyl = [], []
    for o in others:
        if o["method"] == unit["method"] or not o.get("supports", True):
            continue
        Lo = eff_layer(o["method"], o["layer"])
        rec = {"method": o["method"], "layer": Lo, "prior": round(float(prior_of(o["method"])), 3)}
        anyl.append(rec)
        if abs(Lo - L) <= GRID:
            near.append(rec)

    def summ(rs: list[dict[str, Any]]) -> dict[str, Any]:
        clean = [r for r in rs if r["prior"] < prior_thr]
        return {
            "support": bool(rs),
            "support_no_prior": bool(clean),
            "methods": sorted({r["method"] for r in rs}),
            "layers": sorted({(r["method"], r["layer"]) for r in rs}),
        }

    return {"primary": summ(near), "any_layer": summ(anyl)}


# ------------------------------------------------------------------ probe fire rule

AUROC_GATE = 0.85
P_GATE = 0.05
MIN_REFS = 19
PROBES = ("logreg", "dom")


def emp_p(score: float, refs: Sequence[float]) -> float:
    """(1 + #{refs >= score}) / (n + 1)."""
    n = len(refs)
    return (1 + sum(1 for r in refs if r >= score)) / (n + 1)


def fires(
    cell: Mapping[str, Any] | None, concept: str, gold: str, probes: Sequence[str] = PROBES
) -> dict[str, Any]:
    """The fire rule at one (item, layer) probe cell.

    ``cell`` (from the probe stage): {"classes": [...], "n_same": int, "auroc": {probe: {class:
    float}}, "p_concept": {probe: {class: p}}, "p_gold": {probe: p}}. Returns {fires, status,
    reason}: status ``fires`` / ``no_fire`` / ``n/a``."""
    if cell is None:
        return {"fires": False, "status": "n/a", "reason": "no probe cell"}
    if cell.get("na_reason"):
        return {"fires": False, "status": "n/a", "reason": cell["na_reason"]}
    if concept not in cell["classes"]:
        return {"fires": False, "status": "n/a", "reason": "concept not a bank class"}
    if gold not in cell["classes"]:
        return {"fires": False, "status": "n/a", "reason": "gold not a bank class"}
    if cell["n_same"] < MIN_REFS:
        return {"fires": False, "status": "n/a", "reason": f"few refs ({cell['n_same']})"}
    for p in probes:
        a = cell["auroc"][p].get(concept)
        if a is None or a < AUROC_GATE:
            return {"fires": False, "status": "n/a", "reason": f"validity gate ({p} AUROC {a})"}
    for p in probes:
        if cell["p_gold"][p] > P_GATE:
            return {
                "fires": False,
                "status": "n/a",
                "reason": f"gold control ({p} p {cell['p_gold'][p]:.3f})",
            }
    ps = {p: cell["p_concept"][p][concept] for p in probes}
    ok = all(v <= P_GATE for v in ps.values())
    return {"fires": ok, "status": "fires" if ok else "no_fire", "reason": "", "p": ps}


def ladder_label(committed: bool, fire: bool, cross: bool) -> str:
    if not committed:
        return L_NOT
    if fire:
        return L_PROBE
    return L_CROSS if cross else L_HALL
