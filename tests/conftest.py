"""Shared fixtures: a fake LLM (no network) and toy readouts on two real bank items."""

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from hallucination_bench import summarizer
from hallucination_bench.evals.wsbench_chat import bank
from hallucination_bench.judges import taxonomy_v1 as tv1
from hallucination_bench.judges import v5c
from hallucination_bench.llm import CallResult, Usage

TOY_ITEMS = ["chat-dailydialog-0000", "chat-dailydialog-0001"]
_READOUT = re.compile(r'<readout idx="(\d+)">\n(.*?)\n</readout>', re.DOTALL)
_SPAN = re.compile(r'^(S\d+) \| readout (\d+) \| grounding: (\w+) \| (".*")$', re.MULTILINE)


def readout_texts(user: str) -> list[str]:
    return [m.group(2) for m in _READOUT.finditer(user)]


def span_lines(user: str) -> list[tuple[str, int, str, str]]:
    return [
        (m.group(1), int(m.group(2)), m.group(3), json.loads(m.group(4)))
        for m in _SPAN.finditer(user)
    ]


def stage_of(system: str) -> str:
    if system == summarizer.INTERP_SYSTEM:
        return "summary"
    if system == v5c.JUDGE_SYSTEM:
        return "span"
    if system == v5c.VERIFY_SYSTEM:
        return "verify"
    if system == tv1.SYSTEM:
        return "classify"
    return "other"


def default_span(user: str) -> dict[str, Any]:
    """Every readout containing 'Paris' gets that word flagged wrong."""
    return {
        "samples": [
            {
                "idx": j,
                "kind": "specific",
                "spans": [{"text": "Paris", "type": "wrong", "why": "x"}] if "Paris" in t else [],
                "rationale": "",
            }
            for j, t in enumerate(readout_texts(user))
        ]
    }


def default_verify(user: str) -> dict[str, Any]:
    """'Paris' readouts: 'the teacher' is unverifiable; others: 'friendly' is true."""
    out = []
    for j, t in enumerate(readout_texts(user)):
        claims = []
        if "the teacher" in t:
            claims.append({"quote": "the teacher", "status": "unverifiable", "why": "x"})
        if "friendly" in t:
            claims.append({"quote": "friendly", "status": "true", "why": "x"})
        out.append({"idx": j, "kind": "specific", "claims": claims, "rationale": ""})
    return {"samples": out}


def default_classify(user: str) -> dict[str, Any]:
    spans = []
    for sid, _ri, g, _text in span_lines(user):
        spans.append(
            {
                "span_id": sid,
                "type": "concept" if g == "contradicted" else "detail",
                "target": "entity",
                "severity": 3 if g == "contradicted" else 1,
                "unsupported_kind": "inference" if g == "unsupported" else None,
                "why": "w",
            }
        )
    return {"spans": spans, "rationale": "r"}


class FakeLLM:
    """Stands in for :class:`hallucination_bench.llm.LLM`; records every call."""

    def __init__(self, responders: dict[str, Callable[[str], dict[str, Any] | None]] | None = None):
        self.responders = {
            "summary": lambda u: {"interpretation": "A friendly note about Paris and the teacher."},
            "span": default_span,
            "verify": default_verify,
            "classify": default_classify,
            **(responders or {}),
        }
        self.calls: list[tuple[str, str, str, str]] = []  # (stage, model, system, user)
        self.preflights: list[str] = []

    async def call(self, *, model: str, system: str, user: str, schema: dict) -> CallResult:
        st = stage_of(system)
        self.calls.append((st, model, system, user))
        return CallResult(self.responders[st](user), Usage(calls=1, usd=0.01))

    async def preflight(self, model: str) -> None:
        self.preflights.append(model)

    async def aclose(self) -> None:
        pass

    def n(self, stage: str) -> int:
        return sum(c[0] == stage for c in self.calls)


@pytest.fixture
def fake() -> FakeLLM:
    return FakeLLM()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return path


def toy_sites() -> list[tuple[str, int]]:
    items = {it["id"]: it for it in bank.load_items()}
    return [(i, int(s["pos"])) for i in TOY_ITEMS for s in items[i]["sites"]]


@pytest.fixture
def prose_readouts(tmp_path: Path) -> Path:
    """Toy prose arm over the two items' sites at layer 44: site 0 says Paris (wrong) + the
    teacher (unverifiable), the rest are friendly; the last site is empty."""
    sites = toy_sites()
    rows = []
    for n, (iid, pos) in enumerate(sites):
        if n == 0:
            samples = ["It is about Paris and the teacher.", "A friendly reply.", ""]
        elif n == len(sites) - 1:
            samples = ["", " "]
        else:
            samples = ["A friendly reply.", "Something friendly here."]
        rows.append({"id": iid, "layer": 44, "pos": pos, "samples": samples})
    return write_jsonl(tmp_path / "prose.jsonl", rows)


@pytest.fixture
def token_readouts(tmp_path: Path) -> Path:
    sites = toy_sites()[:3]
    rows = [
        {"id": i, "layer": 44, "pos": p, "tokens": ["ĠParis", "Ġteacher"], "scores": [2.0, 1.0]}
        for i, p in sites
    ]
    return write_jsonl(tmp_path / "tokens.jsonl", rows)
