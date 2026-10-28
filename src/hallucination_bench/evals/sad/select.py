"""Selection and read sites (plan 0018 §Selection).

From the gate survivors (``belief`` set, i.e. >= ``gate.THRESHOLD`` of 10 agree) up to
``N_PER_BELIEF`` per belief, mismatched (belief != provenance) survivors first. Read sites per
item, on the tokenisation of
``bank.plain_render``: the 4 suffix positions (``<|im_end|>``, ``\\n``, ``<|im_start|>``,
``assistant``) always, plus the body positions (tokens inside the text) -- all of them if at most
``N_BODY``, else ``N_BODY`` evenly spaced ones including the first and the last. The 3 head
positions (``<|im_start|>user\\n``) precede the text and are never read.

``selected.json`` records positions and regions only: token strings would spell the text."""

import random
from collections import Counter
from typing import Any, Protocol

from hallucination_bench.evals.sad import bank

N_PER_BELIEF = 100
N_BODY = 28
SEED = 0
SUFFIX_TOKENS = ("<|im_end|>", "\n", "<|im_start|>", "assistant")


class Tokenizer(Protocol):
    def __call__(self, text: str, **kw: Any) -> Any: ...


def tokenize(render: str, tok: Any) -> tuple[list[str], list[int]]:
    """``(decoded token strings, end char offsets)`` of ``render`` (fast tokenizer offsets). The
    decoded strings concatenate to ``render`` except where a multi-byte character is split."""
    enc = tok(render, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    ends = [int(b) for _, b in enc["offset_mapping"]]
    toks = [render[a:b] for a, b in enc["offset_mapping"]]
    if len(ids) != len(toks):
        raise ValueError("tokenizer returned misaligned offsets")
    return toks, ends


def body_positions(n_body: int, start: int) -> list[int]:
    if n_body <= N_BODY:
        return list(range(start, start + n_body))
    return sorted({start + round(i * (n_body - 1) / (N_BODY - 1)) for i in range(N_BODY)})


def sites(text: str, tok: Any) -> list[dict[str, Any]]:
    """Read sites ``[{pos, region, token, char}]`` of one item (``char`` = end offset of the
    token in ``plain_render``). Raises if the render does not end with the expected suffix."""
    render = bank.plain_render(text)
    toks, ends = tokenize(render, tok)
    head_end = len(bank.USER_HEAD)
    body_end = head_end + len(text.strip())
    if tuple(toks[-4:]) != SUFFIX_TOKENS:
        raise ValueError(f"render does not end with the suffix tokens: {toks[-4:]!r}")
    first_body = next(i for i, e in enumerate(ends) if e > head_end)
    # the head is exactly <|im_start|>, user, \n (3 tokens) unless the text merges into the \n
    n_total = len(toks)
    body = [i for i in range(first_body, n_total - 4) if ends[i] <= body_end or i == n_total - 5]
    if not body or body[0] != first_body or body[-1] != n_total - 5:
        raise ValueError("body tokens are not contiguous up to the suffix")
    picked = body_positions(len(body), first_body)
    out = [{"pos": p, "region": "body", "token": toks[p], "char": ends[p]} for p in picked]
    for p in range(n_total - 4, n_total):
        out.append({"pos": p, "region": "suffix", "token": toks[p], "char": ends[p]})
    return out


def select(gate: dict[str, dict[str, Any]], n_per_belief: int = N_PER_BELIEF) -> list[str]:
    """Up to ``n_per_belief`` survivors per belief: MISMATCHED survivors (belief differs from SAD
    provenance) first, up to half the quota (more only if the matched ones run out), then a
    seeded sample (``random.Random(SEED)`` over sorted ids) of the
    matched survivors fills the belief up. Returns sorted ids per belief, eval first."""
    rng = random.Random(SEED)
    out: list[str] = []
    for b in ("eval", "deploy"):
        surv = sorted(i for i, g in gate.items() if g["belief"] == b)
        mism = [i for i in surv if gate[i]["provenance"] != b]
        matched = [i for i in surv if gate[i]["provenance"] == b]
        rng.shuffle(mism)
        rng.shuffle(matched)
        # mismatched first, capped at half the quota so the pooled group is not one provenance
        take = mism[: max(n_per_belief // 2, n_per_belief - len(matched))]
        take += matched[: n_per_belief - len(take)]
        out += sorted(take)
    return out


GEN_SUFFIX = "\n<think>\n\n</think>\n\n"  # what the capture's chat render adds after assistant
MAX_POSITIONS = 2304  # the capture's --max-positions (strict): the full render must fit


def build_selected(
    gate: dict[str, dict[str, Any]], raw: dict[str, dict[str, Any]], tok: Any, tokenizer_id: str
) -> dict[str, Any]:
    ids = select(gate)
    items = []
    max_full = 0
    for iid in ids:
        r, g = raw[iid], gate[iid]
        s = sites(r["text"], tok)
        n_tok = len(tokenize(bank.plain_render(r["text"]), tok)[0])
        full = bank.plain_render(r["text"]) + GEN_SUFFIX
        max_full = max(max_full, len(tok(full, add_special_tokens=False)["input_ids"]))
        items.append(
            {
                "id": iid,
                "belief": g["belief"],
                "provenance": r["provenance"],
                "source": r["source"],
                "agree": g["agree"],
                "n_tokens": n_tok,
                "sites": [{"pos": x["pos"], "region": x["region"]} for x in s],
            }
        )
    if max_full > MAX_POSITIONS:
        raise ValueError(f"longest full render {max_full} > capture max_positions {MAX_POSITIONS}")
    return {
        "meta": {
            "max_full_render_tokens": max_full,
            "seed": SEED,
            "n_per_belief": N_PER_BELIEF,
            "n_body": N_BODY,
            "tokenizer": tokenizer_id,
            "n_items": len(items),
            "by_belief": dict(Counter(it["belief"] for it in items)),
            "belief_x_provenance": dict(
                Counter(f"{it['belief']}|{it['provenance']}" for it in items)
            ),
            "n_sites": sum(len(it["sites"]) for it in items),
        },
        "items": items,
    }


def capture_rows(selected: dict[str, Any], raw: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """The global-workspace capture rows (TEXT: write them under gitignored ``outputs/``)."""
    return [
        {
            "label": it["id"],
            "family": "sad",
            "user": raw[it["id"]]["text"].strip(),
            "look_for": "",
            "targets": [],
            "eval_positions": [s["pos"] for s in it["sites"]],
        }
        for it in selected["items"]
    ]


def top_up(
    prev: dict[str, Any],
    gate: dict[str, dict[str, Any]],
    raw: dict[str, dict[str, Any]],
    tok: Any,
    tokenizer_id: str,
    n_per_belief: int = 100,
) -> tuple[dict[str, Any], list[str]]:
    """Rev 5: keep every rev-4 item (already read) and fill each belief group up to
    ``n_per_belief`` with a seeded sample (``random.Random(SEED)`` over sorted ids) of the
    survivors not yet selected. Returns ``(selected, new ids)``; only the new ids need reads."""
    rng = random.Random(SEED)
    kept = {it["id"] for it in prev["items"]}
    items = list(prev["items"])
    new: list[str] = []
    for b in ("eval", "deploy"):
        have = sum(it["belief"] == b for it in items)
        pool = sorted(i for i, g in gate.items() if g["belief"] == b and i not in kept)
        rng.shuffle(pool)
        new += sorted(pool[: max(0, n_per_belief - have)])
    max_full = prev["meta"].get("max_full_render_tokens", 0)
    for iid in new:
        r, g = raw[iid], gate[iid]
        s = sites(r["text"], tok)
        full = bank.plain_render(r["text"]) + GEN_SUFFIX
        max_full = max(max_full, len(tok(full, add_special_tokens=False)["input_ids"]))
        items.append(
            {
                "id": iid,
                "belief": g["belief"],
                "provenance": r["provenance"],
                "source": r["source"],
                "agree": g["agree"],
                "n_tokens": len(tokenize(bank.plain_render(r["text"]), tok)[0]),
                "sites": [{"pos": x["pos"], "region": x["region"]} for x in s],
                "batch": "rev5",
            }
        )
    if max_full > MAX_POSITIONS:
        raise ValueError(f"longest full render {max_full} > capture max_positions {MAX_POSITIONS}")
    meta = dict(prev["meta"])
    meta.update(
        {
            "revision": "rev5 top-up: rev-4 items kept, each belief filled to n_per_belief",
            "n_per_belief": n_per_belief,
            "max_full_render_tokens": max_full,
            "tokenizer": tokenizer_id,
            "n_items": len(items),
            "n_new": len(new),
            "by_belief": dict(Counter(it["belief"] for it in items)),
            "belief_x_provenance": dict(
                Counter(f"{it['belief']}|{it['provenance']}" for it in items)
            ),
            "by_source": dict(Counter(it["source"] for it in items)),
            "n_sites": sum(len(it["sites"]) for it in items),
        }
    )
    return {"meta": meta, "items": items}, new
