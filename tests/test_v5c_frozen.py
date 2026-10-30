"""v5c span + verify stages: literals frozen, parse_verdict / tally_claims rules unchanged, and
the quote-keeping re-implementation agrees with the frozen tally count for count."""

import ast
import json
import random
import re
import subprocess
from collections import Counter
from typing import Any

import pytest

from hallucination_bench import DATA, WSBENCH_PUB_COMMIT, paths
from hallucination_bench.evals.wsbench_chat import bank
from hallucination_bench.judges import v5c

# plan 0023 (W1.6): read the pinned source with ``git show`` (WSB deletes its hallucination family)
SOURCE_REL = "src/wsbench/evals/hallucination/prompts.py"
README = DATA / "wsbench_chat" / "README.md"
_FENCE = re.compile(r"(`{3,})[^\n]*\n(.*?)\1", re.DOTALL)


def _raw(*samples: tuple[str, list[tuple[str, str]]]) -> dict[str, Any]:
    return {
        "samples": [
            {
                "idx": i,
                "kind": kind,
                "spans": [{"text": t, "type": ty, "why": "x"} for t, ty in spans],
                "rationale": "r",
            }
            for i, (kind, spans) in enumerate(samples)
        ]
    }


def _vraw(*claims: list[tuple[str, str]]) -> dict[str, Any]:
    return {
        "samples": [
            {
                "idx": i,
                "kind": "specific",
                "claims": [{"quote": q, "status": s, "why": "x"} for q, s in cl],
                "rationale": "r",
            }
            for i, cl in enumerate(claims)
        ]
    }


# ------------------------------------------------------------------ frozen literals


def test_span_prompt_is_byte_identical_to_the_frozen_json() -> None:
    frozen = json.loads((DATA / "wsbench_chat" / "judge_prompt_v5c_chat.json").read_bytes())
    assert frozen == v5c.FROZEN
    assert frozen["system"] == v5c.JUDGE_SYSTEM
    assert frozen["user"] == v5c.JUDGE_USER
    assert frozen["schema"] == v5c.JUDGE_SCHEMA
    assert frozen["version"] == v5c.PROMPT_VERSION == "v5c-chat"
    assert "{{text, type, why}}" in v5c.JUDGE_USER
    assert v5c.VERIFY_PROMPT_VERSION == "v5c-chat-verify-v1"


def test_every_prompt_literal_is_verbatim_in_the_readme() -> None:
    """Port of wsbench's ``test_family_prompts_verbatim_in_readme`` (the verify literal test)."""
    blocks = [b for _f, b in _FENCE.findall(README.read_text(encoding="utf-8"))]
    for name, text in v5c.PROMPTS.items():
        assert any(text in b for b in blocks), f"{name} is not verbatim in the data README"
    readme = README.read_text()
    assert v5c.PROMPT_VERSION in readme and v5c.VERIFY_PROMPT_VERSION in readme


def _pinned_source() -> str | None:
    r = subprocess.run(
        ["git", "-C", str(paths.WSBENCH_PUB), "show", f"{WSBENCH_PUB_COMMIT}:{SOURCE_REL}"],
        capture_output=True,
        text=True,
    )
    return None if r.returncode else r.stdout


def _source_literals() -> dict[str, str]:
    src = _pinned_source()
    assert src is not None
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        target = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target = node.target
        if target is not None:
            name = getattr(target, "id", None)
            if name in ("JUDGE_SYSTEM", "JUDGE_USER", "VERIFY_SYSTEM", "VERIFY_USER"):
                out[name] = ast.literal_eval(node.value)
            if name in ("VERIFY_STATUSES", "PROMPT_VERSION", "VERIFY_PROMPT_VERSION", "READ_MARK"):
                out[name] = ast.literal_eval(node.value)
    return out


@pytest.mark.skipif(
    _pinned_source() is None,
    reason=f"no workspace-bench clone with {WSBENCH_PUB_COMMIT} (set HBENCH_WSBENCH_PUB)",
)
def test_literals_equal_the_wsbench_pub_source() -> None:
    src = _source_literals()
    assert src["VERIFY_SYSTEM"] == v5c.VERIFY_SYSTEM
    assert src["VERIFY_USER"] == v5c.VERIFY_USER
    assert src["JUDGE_SYSTEM"] == v5c.JUDGE_SYSTEM
    assert src["JUDGE_USER"] == v5c.JUDGE_USER
    assert src["VERIFY_STATUSES"] == v5c.VERIFY_STATUSES
    assert src["READ_MARK"] == v5c.READ_MARK
    assert src["VERIFY_PROMPT_VERSION"] == v5c.VERIFY_PROMPT_VERSION


def test_verify_schema_matches_the_readme_block() -> None:
    blocks = [b for f, b in _FENCE.findall(README.read_text(encoding="utf-8"))]
    schemas = [json.loads(b) for b in blocks if b.lstrip().startswith("{")]
    assert v5c.VERIFY_SCHEMA in schemas and v5c.JUDGE_SCHEMA in schemas


# ------------------------------------------------------------------ rendering + bank


def test_judge_and_verify_prompts_mark_the_site() -> None:
    item = {"prompt": "Where's your better half?", "response": "I am an AI, my better half is X."}
    site = {"char": len("I am an AI,")}
    system, user = v5c.judge_prompt(item, site, ["a", "b"])
    assert system == v5c.JUDGE_SYSTEM
    assert f"I am an AI,{v5c.READ_MARK} my better half" in user
    assert "Below are 2 independent readouts" in user and "{{" not in user
    _s, vuser = v5c.verify_prompt(item, site, ["a b", "c"], [["a b"], []])
    assert 'Established FALSE spans of readout 0:\n  - "a b"' in vuser
    assert "Established FALSE spans of readout 1:\n  (none)" in vuser


def test_bank_is_the_verbatim_copy() -> None:
    b = bank.load_bank()
    items = b["items"]
    assert len(items) == b["meta"]["n_items"] == 149
    assert sum(len(it["sites"]) for it in items) == b["meta"]["n_sites"] == 1123
    assert Counter(it["source"] for it in items) == {"lmsys": 74, "dailydialog": 75}
    rows = {r["id"]: r for r in json.loads((DATA / "wsbench_chat/capture_rows.json").read_text())}
    for it in items:
        assert rows[it["id"]]["read_positions"] == [s["pos"] for s in it["sites"]]
        for s in it["sites"]:
            assert it["response"][: s["char"]].endswith(s["token"])


# ------------------------------------------------------------------ parse_verdict (ported)


def test_classes_wrong_beats_off_topic_and_kind() -> None:
    samples = ["alpha beta gamma", "delta epsilon zeta", "eta theta iota", "kappa lambda mu"]
    raw = _raw(
        ("specific", [("beta gamma", "off_topic"), ("alpha", "wrong")]),
        ("generic", [("epsilon", "wrong")]),
        ("empty", [("theta iota", "off_topic")]),
        ("specific", []),
    )
    v = v5c.parse_verdict(raw, samples)
    assert v is not None
    assert [x.cls for x in v] == ["hallucinated", "hallucinated", "generic", "consistent"]


def test_span_must_be_verbatim_and_unknown_type_is_wrong() -> None:
    v = v5c.parse_verdict(_raw(("specific", [("not in readout", "wrong")])), ["the readout"])
    assert v is not None and v[0].cls == "consistent" and v[0].n_unverified == 1
    v = v5c.parse_verdict(_raw(("specific", [("the readout", "mystery")])), ["the readout"])
    assert v is not None and v[0].cls == "hallucinated"


def test_partial_verdict_is_not_fully_judged() -> None:
    v = v5c.parse_verdict(_raw(("specific", [])), ["a b c", "d e f"])
    assert v is not None and [x.cls for x in v] == ["consistent", "unjudged"]
    assert not v5c.fully_judged(_raw(("specific", [])), ["a b c", "d e f"])
    assert v5c.fully_judged(_raw(("specific", [])), ["a b c"])
    assert v5c.parse_verdict(None, ["x"]) is None


def test_revocation_is_reported() -> None:
    samples = ["the Moon formed a crust and KBr", "the Moon formed a crust"]
    raw = _raw(
        ("specific", [("the Moon formed a crust", "wrong"), ("KBr", "wrong")]),
        ("specific", [("the Moon formed a crust", "wrong")]),
    )
    v = v5c.parse_verdict(raw, samples)
    assert v is not None and [x.revoked for x in v] == [False, True]


# ------------------------------------------------------------------ tally_claims (ported)


def test_tally_claims_rules() -> None:
    samples = ["The assistant names its wife, Sarah.", "A friendly exclamation."]
    wrong = [["its wife, Sarah"], []]
    raw = _vraw(
        [
            ("names its wife", "true"),
            ("The assistant", "true"),
            ("the assistant", "TRUE "),
            ("Sarah is her wife", "true"),
            ("names", "bogus"),
        ],
        [("friendly", "disputed"), ("exclamation", "off_topic"), ("A friendly", "unverifiable")],
    )
    t = v5c.tally_claims(raw, samples, wrong)
    keys = ("false", "true", "unverifiable", "disputed", "off_topic", "n_unverified")
    want = [(1, 3, 1, 0, 0, 1), (0, 0, 2, 1, 1, 0)]
    assert t == [dict(zip(keys, w, strict=True)) for w in want]
    lists = v5c.checked_claim_lists(raw, samples, wrong)
    assert lists is not None
    assert lists[0].unsupported == ["names"] and len(lists[0].true) == 3
    assert lists[1].disputed == ["friendly"] and lists[1].unsupported == ["A friendly"]


def test_tally_claims_rejects_partial_answers() -> None:
    samples = ["one readout", "another readout"]
    assert v5c.tally_claims(_vraw([]), samples, [[], []]) is None
    assert v5c.claim_lists(_vraw([]), samples, [[], []]) is None
    span_shaped = _raw(("specific", []), ("specific", []))
    assert v5c.tally_claims(span_shaped, samples, [[], []]) is None
    assert v5c.claim_lists(span_shaped, samples, [[], []]) is None


# ------------------------------------------------------------------ claim_lists == tally_claims


def test_claim_lists_fixtures_unknown_short_overlapping() -> None:
    samples = ["Joe walks to Paris with his sister Maria at noon.", "ok"]
    wrong = [["to Paris"], []]
    raw = _vraw(
        [
            ("walks to Paris", "unverifiable"),  # overlaps a wrong span (superset) -> skipped
            ("Paris", "true"),  # overlaps (subset) -> skipped
            ("Jo", "unverifiable"),  # under 3 chars -> unverified
            ("sister Maria", "weird-status"),  # unknown -> unsupported
            ("at noon", "disputed"),
            ("at noon", "unverifiable"),
            ("", "true"),
            ("his sister", "off_topic"),
        ],
        [("ok", "true")],  # 2 chars -> unverified
    )
    lists = v5c.claim_lists(raw, samples, wrong)
    tally = v5c.tally_claims(raw, samples, wrong)
    assert lists is not None and tally is not None
    for cl, t in zip(lists, tally, strict=True):
        assert len(cl.unsupported) + len(cl.disputed) == t["unverifiable"]
        for k in ("true", "off_topic"):
            assert len(getattr(cl, k)) == t[k]
        assert cl.false == t["false"] and cl.n_unverified == t["n_unverified"]
    assert lists[0].unsupported == ["sister Maria", "at noon"]
    assert lists[0].disputed == ["at noon"]
    assert lists[0].n_unverified == 2 and lists[1].n_unverified == 1


def test_claim_lists_agree_with_tally_on_random_answers() -> None:
    rng = random.Random(0)
    words = "alpha beta gamma delta eps zeta eta theta io kappa".split()
    statuses = ["true", "unverifiable", "disputed", "off_topic", "bogus", "", "TRUE"]
    for _ in range(400):
        samples = [" ".join(rng.sample(words, 6)) for _ in range(rng.randint(1, 3))]
        wrong = [[" ".join(s.split()[:2])] if rng.random() < 0.5 else [] for s in samples]
        claims = []
        for s in samples:
            toks = s.split()
            cl = []
            for _ in range(rng.randint(0, 5)):
                a = rng.randrange(len(toks))
                q = " ".join(toks[a : a + rng.randint(1, 3)])
                if rng.random() < 0.2:
                    q = q[:2]
                if rng.random() < 0.1:
                    q = "not there"
                cl.append((q, rng.choice(statuses)))
            claims.append(cl)
        raw = _vraw(*claims)
        lists = v5c.checked_claim_lists(raw, samples, wrong)  # raises on any disagreement
        assert lists is not None


def test_classify_quotes_dedupe_offtopic_and_disputed_precedence() -> None:
    cl = v5c.ClaimLists(
        unsupported=["the teacher", "The  teacher", "stray code", "at noon"],
        disputed=["at noon", "AT NOON", "code block"],
    )
    unsup, disp, n_off = v5c.classify_quotes(cl, ["stray code block"])
    assert unsup == ["the teacher"]  # deduped; off-topic overlap dropped; disputed wins
    assert disp == ["at noon"]
    assert n_off == 2  # "stray code" and "code block"
