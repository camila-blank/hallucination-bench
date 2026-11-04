"""Plan 0029 §3.2, §3.7, §2.4: the joint 11-layer unit -- ids, key and sentinel, the per-layer
source precedence, the joint client, rebuilds, and the overrefusal / benign_roleplay joint
exports read by the index."""

import json
from pathlib import Path

import pytest
from conftest import TOY_ITEMS, write_jsonl
from test_claimmap_v28 import Map28Fake
from test_index_v7 import ITEMS as BR_ITEMS
from test_index_v7 import _summary

from hallucination_bench import cli
from hallucination_bench import index as ix
from hallucination_bench.evals.benign_roleplay import index_export as bx
from hallucination_bench.evals.overrefusal import offtopic_export as ox
from hallucination_bench.evals.overrefusal import sj_metrics, sjudge
from hallucination_bench.evals.wsbench_chat import bank
from hallucination_bench.evals.wsbench_chat import run as wrun
from hallucination_bench.judges import claimmap_v2 as cm
from hallucination_bench.llm import JudgeConfigError
from hallucination_bench.readouts import Cell

V28 = "claimmap-v2.8"
M8 = wrun.Models(map_prompt=V28)
LAYERS = (20, 24, 28)


def _readouts(path: Path, layers, tag: str) -> Path:
    items = {it["id"]: it for it in bank.load_items()}
    rows = []
    for iid in TOY_ITEMS:
        for n, s in enumerate(items[iid]["sites"]):
            for la in layers:
                samples = [f"{tag} L{la} reads Paris here.", "second sample"] if n < 3 else [""]
                rows.append({"id": iid, "layer": la, "pos": int(s["pos"]), "samples": samples})
    return write_jsonl(path, rows)


def test_joint_key_ids_sentinel_and_precedence(tmp_path):
    a = _readouts(tmp_path / "a.jsonl", (20, 28), "A")  # the first-listed arm wins its layers
    b = _readouts(tmp_path / "b.jsonl", (20, 24, 28), "B")
    fake = Map28Fake()
    s = wrun.run_joint(
        "a", [("a", a), ("b", b)], tmp_path, M8, layers=LAYERS, llm=fake, concurrency=2
    )
    d = wrun.arm_dir(tmp_path, M8, "a")
    recs = [json.loads(x) for x in (d / wrun.JOINT_CELLS).read_text().splitlines()]
    it0 = next(it for it in bank.load_items() if it["id"] == TOY_ITEMS[0])
    p0 = int(it0["sites"][0]["pos"])
    r = next(x for x in recs if x["key"] == f"{it0['id']}__Lall__p{p0}")
    assert wrun.joint_key(it0["id"], p0) == r["key"] and r["layer"] == "all"
    assert r["joint_readouts"] == {"0": 20, "1": 24, "2": 28}
    assert r["joint_sources"] == {"0": "a", "1": "b", "2": "a"}
    assert r["samples"] == [
        "A L20 reads Paris here.",
        "B L24 reads Paris here.",
        "A L28 reads Paris here.",
    ]  # sample 0 of each layer only
    assert r["stages"]["map"] == "ok" and r["status"] == "judged"
    unit_ids = {u["id"] for u in r["map"]["units"]}
    assert {i.split(".")[0] for i in unit_ids} == {"R0", "R1", "R2"}
    assert all("v28" not in sg for sg in r["map"]["segments"])
    assert "continuation_text" in r["map"]["segments"][0]
    users = [c[3] for c in fake.calls if c[0] == "map"]
    assert users and all("R0 = layer 20, R1 = layer 24, R2 = layer 28" in u for u in users)
    assert sum(x["status"] == "empty" for x in recs) > 0
    assert s["config"]["unit"] == "joint" and s["overall"]["n_items"] == 2
    assert set(s["by_layer"]) == {"all"}
    j = json.loads((d / wrun.JOINT_JSON).read_text())
    assert j["sources"] == [["a", str(a)], ["b", str(b)]] and j["layers"] == list(LAYERS)
    # cache keys and the per-layer machinery do not mix
    keys = [json.loads(x)["key"] for x in (d / "cache.jsonl").open()]
    assert keys and all("__Lall__" in k and k.endswith(":M") for k in keys)
    # a rebuild (also through rebuild_arm, which refreshes the joint cells) is identical
    before = (d / wrun.JOINT_CELLS).read_text()
    wrun.rebuild_arm("a", a, tmp_path, wrun.Models())
    assert (d / wrun.JOINT_CELLS).read_text() == before
    # the index reads the joint record under the JOINT sentinel, rendered "all"
    mats = ix.claim_rows(d / wrun.JOINT_CELLS, ix.MAIN)
    assert set(mats) == {ix.JOINT} and ix.JOINT == -1
    # a resume makes no call; other sources are refused
    fake2 = Map28Fake()
    wrun.run_joint("a", [("a", a), ("b", b)], tmp_path, M8, layers=LAYERS, llm=fake2)
    assert fake2.calls == []
    with pytest.raises(JudgeConfigError, match="other sources"):
        wrun.run_joint("a", [("b", b)], tmp_path, M8, layers=LAYERS, llm=Map28Fake())


def test_joint_needs_v28_single_call_and_every_layer(tmp_path):
    a = _readouts(tmp_path / "a.jsonl", (20,), "A")
    with pytest.raises(JudgeConfigError, match=r"claimmap-v2\.8"):
        wrun.run_joint("a", [("a", a)], tmp_path, wrun.Models(), layers=(20,), llm=Map28Fake())
    with pytest.raises(JudgeConfigError, match="single-call"):
        wrun.run_joint(
            "a",
            [("a", a)],
            tmp_path,
            wrun.Models(map_prompt=V28, map_anchor="v5c"),
            layers=(20,),
            llm=Map28Fake(),
        )
    with pytest.raises(SystemExit, match="no source has layers"):
        wrun.run_joint("a", [("a", a)], tmp_path, M8, layers=(20, 24), llm=Map28Fake())


def test_token_joint_uses_the_per_layer_summaries(tmp_path):
    items = {it["id"]: it for it in bank.load_items()}
    rows = []
    for la in (20, 24):
        for s in items[TOY_ITEMS[0]]["sites"][:2]:
            rows.append(
                {
                    "id": TOY_ITEMS[0],
                    "layer": la,
                    "pos": int(s["pos"]),
                    "tokens": ["ĠParis", "Ġteacher"],
                    "scores": [2.0, 1.0],
                }
            )
    t = write_jsonl(tmp_path / "t.jsonl", rows)
    fake = Map28Fake()
    wrun.run_joint("t", [("t", t)], tmp_path, M8, layers=(20, 24), llm=fake)
    assert fake.n("summary") == 4  # one per (layer, site), keyed like the per-layer summaries
    d = wrun.arm_dir(tmp_path, M8, "t")
    keys = {json.loads(x)["key"] for x in (d / "cache.jsonl").open()}
    assert any(k.startswith("summ:") and "__L020__" in k for k in keys)
    rec = next(
        json.loads(x)
        for x in (d / wrun.JOINT_CELLS).open()
        if '"map"' in x and json.loads(x)["map"]
    )
    assert rec["samples"][0] == rec["samples"][1] == "A friendly note about Paris and the teacher."
    with pytest.raises(SystemExit, match="token arm"):
        wrun.run_joint("t", [("t", t), ("u", t)], tmp_path / "x", M8, layers=(20,), llm=fake)


def test_cli_joint_flags(tmp_path):
    a = _readouts(tmp_path / "a.jsonl", (20, 24), "A")
    p = cli.build_parser()
    args = p.parse_args(["run", "--arm", f"a={a}", "--joint", "--joint-from", "a"])
    assert args.joint and args.joint_from == "a"
    # without --map-prompt claimmap-v2.8 the joint run is refused with exit 3
    rc = cli.main(
        ["run", "--arm", f"a={a}", "--joint", "--layers", "20,24", "--out", str(tmp_path / "o")]
    )
    assert rc == 3


# ---------------------------------------------------------------- safety joint exports


def test_benign_roleplay_joint_export(tmp_path):
    flags = {20: {"s0"}, 24: {"s3", "h1"}}
    sp = _summary(tmp_path / "s.json", BR_ITEMS, {"jlens": flags})
    meta7 = bx.export([sp], tmp_path / "r7", BR_ITEMS)
    assert "joint" not in meta7 and not (tmp_path / "r7" / bx.COUNTS_JOINT).exists()
    meta = bx.export([sp], tmp_path / "r8", BR_ITEMS, joint=True)
    rows = [json.loads(x) for x in (tmp_path / "r8" / bx.COUNTS_JOINT).read_text().splitlines()]
    assert {r["layer"] for r in rows} == {"all"} and len(rows) == 12
    assert {r["item"] for r in rows if r["n_contradicted"]} == {"s0", "s3", "h1"}
    assert meta["joint"]["n_rows"] == 12
    # the per-layer file is the v7 one, byte for byte
    assert (tmp_path / "r8" / bx.COUNTS).read_bytes() == (tmp_path / "r7" / bx.COUNTS).read_bytes()
    jr, _ = ix.safety_rows(tmp_path / "r8", ix.BENIGN_ROLEPLAY, "v8", joint=True)
    got, _ = ix.safety_metric(jr["jlens"][ix.JOINT])
    u = json.loads(sp.read_text())["rates"]["jlens"]["units"]["all"]
    assert got["v"] == u["false_recognition_rate"] and got["ci"] == u["ci95"]
    args = cli.build_parser().parse_args(
        ["benign-roleplay", "index-export", "--summary", str(sp), "--dest", "x", "--joint"]
    )
    assert args.joint


def _rec(reader: str, item: str, layer, dl, off) -> dict:
    return {
        "reader": reader,
        "item": item,
        "population": "never_refused",
        "layer": layer,
        "summary_truncated": item == "i0",
        "digest": {"status": "ok"},
        "verdict": "refusal" if dl else "no_refusal",
        "decline_lines": dl,
        **(
            {
                "off": {
                    "topics": {str(k): v for k, v in zip(dl, off, strict=True)},
                    "any_off_topic": "off_topic" in off,
                }
            }
            if dl
            else {}
        ),
    }


def test_overrefusal_joint_units_and_export(tmp_path, monkeypatch):
    item = {"id": "i0", "sites": [{"pos": 0}, {"pos": 1}], "population": "never_refused"}
    cells = [Cell("i0", la, p, ("x",), None, None) for la in (20, 24) for p in (0, 1)]
    cells.append(Cell("i0", 28, 0, ("x",), None, None))
    us = sjudge.joint_units("jlens", [item], cells, [20, 24])
    assert len(us) == 1 and us[0].layer == sjudge.JOINT_LAYER and len(us[0].cells) == 4
    assert us[0].n_layers == 2
    b = sjudge.make_bundle(us[0].cells, len(item["sites"]) * us[0].n_layers)
    assert b.n_missing == 0
    # export
    items = [{"id": f"i{k}", "population": "never_refused"} for k in range(3)]
    monkeypatch.setattr(ox.data, "load_items", lambda: items)
    root = tmp_path / "overrefusal" / "m"
    monkeypatch.setattr(ox.run, "eval_root", lambda out, model: root)
    recs = [
        _rec("jlens", "i0", "all", [1, 2], ["off_topic", "on_topic"]),
        _rec("jlens", "i1", "all", [], []),
        _rec("jlens", "i2", "all", [3], ["on_topic"]),
    ]
    root.mkdir(parents=True)
    write_jsonl(root / ox.RECORDS_JOINT, recs)
    res = ox.export_joint(tmp_path, "m")
    rows = [json.loads(x) for x in (root / ox.OFFTOPIC_DIR / ox.JOINT_COUNTS).open()]
    assert [r["layer"] for r in rows] == ["all"] * 3
    assert sum(r["n_claims"] for r in rows) == 3 and sum(r["n_contradicted"] for r in rows) == 1
    assert res["truncation"]["jlens"] == {"n": 3, "n_truncated": 1, "share": 1 / 3}
    meta = json.loads((root / ox.OFFTOPIC_DIR / ox.JOINT_META).read_text())
    assert meta["unit"] == ix.UNIT_REFUSAL_CLAIMS and meta["layer"] == "all"
    jr, _ = ix.safety_rows(root / ox.OFFTOPIC_DIR, ix.OVERREFUSAL, "v8", joint=True)
    got, _ = ix.safety_metric(jr["jlens"][ix.JOINT])
    assert got["v"] == pytest.approx(1 / 3)
    assert sj_metrics.truncation_share(recs)["jlens"]["n_truncated"] == 1
    # a per-layer record file is refused as joint input
    write_jsonl(root / ox.RECORDS_JOINT, [_rec("jlens", "i0", 20, [], [])])
    with pytest.raises(ValueError, match="not a joint record file"):
        ox.export_joint(tmp_path, "m")
    p = cli.build_parser()
    assert p.parse_args(["overrefusal", "offtopic-export", "--joint"]).joint


def test_joint_records_sort_after_the_layers(tmp_path):
    recs = [{"reader": "a", "layer": "all", "item": "x"}, {"reader": "a", "layer": 20, "item": "x"}]
    sjudge.write_records(recs, tmp_path / "r.jsonl")
    assert [json.loads(x)["layer"] for x in (tmp_path / "r.jsonl").open()] == [20, "all"]


def test_joint_layers_constant():
    assert cm.JOINT_LAYERS == tuple(20 + 4 * i for i in range(11))
