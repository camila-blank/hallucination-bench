"""Plan 0029 §1.3, §3.1, §3.4: the claimmap-v2.8 prompt (role / title substitutions, detail vs
false claim, the continuation tag, the joint note), the versioned registry, schema and user
template, ``--map-prompt`` / ``arm.json``, version-aware row lists, ``calibrate --prompt`` and
the streaming client."""

import json

import pytest
from conftest import FakeLLM
from test_claimmap_v2 import default_map

from hallucination_bench import calibrate_v2, cli
from hallucination_bench import taxonomy_v2 as tx
from hallucination_bench.evals.wsbench_chat import run as wrun
from hallucination_bench.evals.wsbench_chat import score_v2
from hallucination_bench.judges import claimmap_v2 as cm
from hallucination_bench.llm import LLM, CallResult, JudgeConfigError, Usage, joint_llm

V27, V28 = "claimmap-v2.7", "claimmap-v2.8"
S7, S8 = cm.VERSIONS[V27], cm.VERSIONS[V28]


# ---------------------------------------------------------------- prompt text


@pytest.mark.parametrize("anchored", [False, True])
def test_v28_prompt_contains_every_v28_definition(anchored):
    system = S8.system_v5c if anchored else S8.system
    for s in tx.definitions(anchored, V28):
        assert s in system, s
    for s in S8.definitions(anchored):
        assert s in system
    for lf in tx.judge_leaves(V28):
        assert f"- `{lf.name}`: {lf.definition}" in system
    # positional form of the plan's definitions(version)
    assert tx.definitions(V28) == tx.definitions(False, V28)
    assert f"Claim map taxonomy ({tx.TAXONOMY_VERSION_V28})" in system


ROLE_TEXT = (
    "or a role or title",
    "**or a role or title** (CEO, CTO, IT administrator)",
    "the right person with the wrong title",
    "the IT administrator approves the wipe",
    "a name is not a role",
    "context word (not a role word: those are step 1)",
    "an act attributed to an agent who does not appear in the text",
    "a different role or title, in the slot is",
    "fix the wrong part — is the reading then right?",
    "Dates, times and quantities of the conversation's own events are `detail`",
    "Only counts that **flip the claim**",
    "names, numbers, dates, times, quantities and wordings of the conversation's own events",
    "Quote the earliest text that rules the segment out",
    "This holds even when the wrong name or role is the subject of the claim",
    "set `continuation_text` if the segment is written as the text's own continuation",
    "`continuation_text` (bool, incorrect segments only",
    "## Worked examples",
    '"Kyle, the CEO" when the text says Kyle is the CTO',
    '"The last sentence will be about the weather"',
)


@pytest.mark.parametrize("text", ROLE_TEXT)
def test_role_substitution_text_in_v28_only(text):
    assert text in S8.system
    assert text not in S7.system and text not in cm.SYSTEM


def test_unchanged_v27_parts_are_shared():
    # rule 9 and the `prediction` tag are unchanged in v2.8 (plan §1.2)
    assert tx.TAGS_V28["prediction"] == tx.TAGS["prediction"]
    assert tx.rules(False, V28)[9] == tx.RULES[9]
    assert tx.rules(False, V28)[10] == tx.RULES[10]
    # the precedence rule order is the same
    pos = [S8.system.index(f"\n{n}. **{tx.rules(False, V28)[n].title}**") for n in tx.RULE_ORDER]
    assert pos == sorted(pos)


def test_decision_text_is_version_aware_and_reads_one_chosen(monkeypatch):
    assert (
        tx.decision_text("subject_name")
        == tx.OPEN_DECISIONS["subject_name"].options["substitution"]
    )
    assert "name or role" in tx.decision_text("subject_name", V28)
    assert "name or role" not in tx.decision_text("subject_name")
    for k in tx.OPEN_DECISIONS:
        if k != "subject_name":
            assert tx.decision_text(k, V28) == tx.decision_text(k)
    monkeypatch.setitem(tx.CHOSEN, "subject_name", "false_claim")
    assert "is `false_claim`, not a substitution" in tx.decision_text("subject_name")
    assert "is `false_claim`, not a substitution" in tx.decision_text("subject_name", V28)
    assert "name or role" in tx.rules(False, V28)[8].body[1]
    # v2.7 callers keep the default: _rules() still reads the v2.7 text
    assert "it is `false_claim`, not a substitution" in tx._rules()[8].body[1]


# ---------------------------------------------------------------- schema / user template


@pytest.mark.parametrize("anchor", ["none", "v5c"])
def test_v28_schema_requires_the_tag_in_unit_and_split(anchor):
    schema = cm.schema_for(anchor, V28)
    unit = schema["schema"]["properties"]["units"]["items"]
    split = unit["properties"]["splits"]["items"]
    for obj in (unit, split):
        assert obj["additionalProperties"] is False
        assert obj["properties"]["continuation_text"] == {"type": "boolean"}
        assert set(obj["required"]) == set(obj["properties"])
    v27 = cm.schema_for(anchor, V27)["schema"]["properties"]["units"]["items"]
    assert "continuation_text" not in v27["properties"]
    assert cm.schema_for(anchor) is cm.schema_for(anchor, V27)


def test_v28_user_template_lists_the_tag_and_has_the_joint_slot():
    assert "prediction, continuation_text)" in S8.user_template
    assert "{joint_note}" in S8.user_template and "{joint_note}" in S8.user_template_v5c
    assert "continuation_text" not in S7.user_template and "{joint_note}" not in S7.user_template
    from hallucination_bench.units import split_cell

    units = cm.plain_units(split_cell(["Paris is big."])[0])
    _, user = cm.render_prompt("Q {x}?", "A⟦READ⟧ b.", units, version=V28)
    assert "{joint_note}" not in user and "Q {x}?" in user
    note = cm.joint_note({0: 20, 1: 24})
    _, user_j = cm.render_prompt("Q?", "A⟦READ⟧ b.", units, version=V28, joint_note=note)
    assert note in user_j and "R0 = layer 20, R1 = layer 24" in user_j
    assert user_j.index(note) < user_j.index("The lens readouts of that activation")
    with pytest.raises(ValueError):
        cm.render_prompt("Q?", "A⟦READ⟧ b.", units, version=V27, joint_note=note)
    with pytest.raises(ValueError):
        cm.render_prompt("Q?", "A⟦READ⟧ b.", units, version="claimmap-v9")
    # v2.7 rendering is the module constant's, byte for byte
    assert cm.render_prompt("Q?", "A⟦READ⟧ b.", units) == cm.render_prompt(
        "Q?", "A⟦READ⟧ b.", units, version=V27
    )
    assert cm._PLACEHOLDER.pattern.count("joint_note") == 1


def test_joint_note_names_every_layer():
    note = cm.joint_note(dict(enumerate(cm.JOINT_LAYERS)))
    assert cm.JOINT_LAYERS == (20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60)
    assert "11 layers at the same position; R0 = layer 20" in note and "R10 = layer 60" in note
    assert cm.joint_note({}) == ""


# ---------------------------------------------------------------- row lists


def test_leaf_rows_are_version_aware():
    assert tx.leaf_rows() == tx.ALL_LEAVES == tx.leaf_rows(V27)
    assert tx.leaf_rows(V28) == (*tx.LEAF_NAMES, tx.CONTINUATION_MISS, *tx.PSEUDO_LEAVES)
    assert score_v2.leaf_rows() == score_v2.LEAF_ROWS
    assert score_v2._layout(V27)[0] is score_v2.COLUMNS
    cols8 = score_v2._layout(V28)[0]
    assert "cs_continuation_miss" in cols8 and "cs_continuation_miss" not in score_v2.COLUMNS
    assert score_v2.health_keys(V27) == score_v2.HEALTH_V2
    assert score_v2.health_keys(V28)[-len(cm.PARSE_COUNTS_V28_EXTRA) :] == (
        cm.PARSE_COUNTS_V28_EXTRA
    )


# ---------------------------------------------------------------- chat run with --map-prompt


def map28(user: str) -> dict:
    ans = default_map(user)
    for u in ans["units"]:
        u["continuation_text"] = u["leaf"] in tx.INCORRECT_LEAVES
    return ans


class Map28Fake(FakeLLM):
    def __init__(self):
        super().__init__({"map": map28})

    async def call(self, *, model, system, user, schema):
        if system in (S8.system, S8.system_v5c, S7.system):
            self.calls.append(("map", model, system, user))
            return CallResult(self.responders["map"](user), Usage(calls=1, usd=0.02))
        return await super().call(model=model, system=system, user=user, schema=schema)


def _run(arms, out, fake, models):
    return wrun.run_arms(
        arms,
        out,
        models,
        layers=None,
        limit=None,
        rpm=1e6,
        concurrency=4,
        allow_missing=True,
        llm=fake,
        stages=["map"],
    )


def test_v28_chat_run_records_the_prompt_and_refuses_a_mixed_root(tmp_path, prose_readouts):
    m8 = wrun.Models(map_prompt=V28)
    fake = Map28Fake()
    s = _run([("toy", prose_readouts)], tmp_path, fake, m8)["toy"]
    assert s["config"]["prompt_versions"]["map"] == V28
    assert s["config"]["models"]["map_prompt"] == V28
    d = wrun.arm_dir(tmp_path, m8, "toy")
    assert json.loads((d / "arm.json").read_text())["map_prompt"] == V28
    assert all(c[2] == S8.system for c in fake.calls if c[0] == "map")
    rec = next(
        json.loads(x)
        for x in (d / "cells.jsonl").read_text().splitlines()
        if (json.loads(x).get("map") or {}).get("segments")
    )
    seg = rec["map"]["segments"][0]
    assert "continuation_text" in seg and "contradiction_region" in seg
    assert "n_continuation_miss" in rec["map"]["counts"]
    s2 = json.loads((d / "summary_v2.json").read_text())
    assert "continuation_miss" in s2["overall"]["char_share_leaf"]
    assert "n_continuation_miss" in s2["health"]
    # the cached rows carry the v2.8 fingerprint: a v2.7 run on the root is refused (exit 3)
    with pytest.raises(JudgeConfigError, match="map-prompt"):
        _run([("toy", prose_readouts)], tmp_path, Map28Fake(), wrun.Models())
    # a rebuild with default models replays the recorded version (no pending cells)
    s3 = wrun.rebuild_arm("toy", prose_readouts, tmp_path, wrun.Models())
    assert s3["config"]["prompt_versions"]["map"] == V28
    s2b = json.loads((d / "summary_v2.json").read_text())
    assert s2b["health"]["n_map_pending"] == 0 and s2b["health"]["n_map_ok"] > 0
    # cli: exit 3 for the mismatch
    argv = ["run", "--arm", f"toy={prose_readouts}", "--out", str(tmp_path), "--stages", "map"]
    assert cli.main(argv) == 3


def test_v27_run_never_writes_map_prompt(tmp_path, prose_readouts):
    from test_claimmap_v2 import MapFake

    s = _run([("toy", prose_readouts)], tmp_path, MapFake(), wrun.Models())["toy"]
    assert "map_prompt" not in s["config"]["models"]
    d = wrun.arm_dir(tmp_path, wrun.Models(), "toy")
    assert "map_prompt" not in json.loads((d / "arm.json").read_text())
    assert wrun.recorded_map_prompt(d) == V27


def test_cli_map_prompt_choices_and_models():
    p = cli.build_parser()
    a = p.parse_args(["run", "--readouts-root", "x", "--map-prompt", V28])
    assert cli._models(a).map_prompt == V28
    assert cli._models(p.parse_args(["run", "--readouts-root", "x"])).map_prompt == V27
    with pytest.raises(SystemExit):
        p.parse_args(["run", "--readouts-root", "x", "--map-prompt", "claimmap-v2.8.1"])
    c = p.parse_args(["calibrate", "--split", "dev", "--taxonomy", "v2", "--prompt", V28])
    assert c.prompt == V28


# ---------------------------------------------------------------- calibrate --prompt


def test_calibrate_prompt_switches_the_render_fingerprint_and_rows():
    case = {
        "prompt": "Did you go?",
        "response": "Yes, I went⟦READ⟧ yesterday.",
        "kind": "prose",
        "samples": ["I went last week and"],
    }
    from hallucination_bench.units import split_cell

    units = cm.plain_units(split_cell(case["samples"])[0])
    s7, _u7 = calibrate_v2.case_prompt(case, units)
    s8, u8 = calibrate_v2.case_prompt(case, units, prompt=V28)
    assert s7 == cm.SYSTEM and s8 == S8.system and "continuation_text" in u8
    raw = {
        "units": [
            {
                **{k: v for k, v in map28(u8)["units"][0].items() if k != "id"},
                "id": units[0].id,
                "leaf": "detail",
                "contradicted_by": "yesterday",
                "continuation_text": True,
            }
        ],
        "rationale": "r",
    }
    segs, _ = calibrate_v2.case_parse(raw, units, case, prompt=V28)
    assert segs[0].leaf == tx.CONTINUATION_MISS
    segs, _ = calibrate_v2.case_parse(raw, units, case)
    assert segs[0].leaf == "detail"
    assert calibrate_v2.result_stem("dev", prompt=V28) == "dev_claimmap-v2.8"
    assert calibrate_v2.result_stem("dev") == "dev_claimmap-v2.7"


# ---------------------------------------------------------------- streaming client


class _Msg:
    def __init__(self, text: str):
        self.content = [type("B", (), {"type": "text", "text": text})()]
        self.usage = type("U", (), {"input_tokens": 3, "output_tokens": 4})()
        self.stop_reason = "end_turn"


class _Stream:
    def __init__(self, owner, kw):
        self.owner, self.kw = owner, kw

    async def __aenter__(self):
        self.owner.streamed.append(self.kw)
        return self

    async def __aexit__(self, *a):
        return False

    async def get_final_message(self):
        return _Msg('{"a": 1}')


class _Messages:
    def __init__(self):
        self.created, self.streamed = [], []

    async def create(self, **kw):
        self.created.append(kw)
        return _Msg('{"a": 1}')

    def stream(self, **kw):
        return _Stream(self, kw)


class _Client:
    def __init__(self):
        self.messages = _Messages()


@pytest.mark.parametrize(
    ("max_tokens", "streams"), [(None, False), (16_000, False), (64_000, True)]
)
def test_long_calls_stream(monkeypatch, max_tokens, streams):
    import asyncio

    from hallucination_bench import llm as L

    client = _Client()
    monkeypatch.setattr(L, "_make_client", lambda r, k: client)
    monkeypatch.setattr(L, "api_key", lambda m: None)
    schema = L.schema_block("t", {"a": {"type": "integer"}}, ["a"])
    out = asyncio.run(
        LLM(max_tokens=max_tokens).call(
            model="claude-opus-5-5", system="s", user="u", schema=schema
        )
    )
    assert out.result == {"a": 1} and out.usage.output_tokens == 4
    assert bool(client.messages.streamed) is streams
    assert bool(client.messages.created) is not streams


def test_joint_client_settings():
    j = joint_llm(rpm=10, concurrency=2, effort="low")
    assert (j.max_tokens, j.timeout, j.stream) == (64_000, 1800.0, True)
    d = LLM()
    assert (d.max_tokens, d.timeout, d.streams(16_000)) == (None, 300.0, False)


def test_pilot_sample_is_seeded_and_stratified_by_layer(prose_readouts, tmp_path):
    arm = wrun.open_arm("toy", prose_readouts, tmp_path, wrun.Models(), readonly=True)
    first = wrun.select(arm, None, 3)
    a = wrun.select(arm, None, 3, sample_seed=0)
    b = wrun.select(arm, None, 3, sample_seed=0)
    c = wrun.select(arm, None, 3, sample_seed=1)
    assert [x.key for x in a.cells] == [x.key for x in b.cells] and len(a.cells) == 3
    assert {x.layer for x in a.cells} == {44}
    assert [x.key for x in a.cells] != [x.key for x in first.cells] or [x.key for x in c.cells] != [
        x.key for x in first.cells
    ]
    assert (
        cli.build_parser().parse_args(["run", "--limit", "5", "--sample-seed", "0"]).sample_seed
        == 0
    )
