"""Plan 0002 rev 19 (after parsing, no re-judging): hedged option lists are scored unverifiable
(same_slot_alternative), and model_state gets its own verdict, thinky."""

import json

from test_claimmap_v2 import MapFake, _run, e, prompt_quote, unit_lines

from hallucination_bench import consistency_v2 as cons
from hallucination_bench import taxonomy_v2 as tx
from hallucination_bench.cache import Cache, fingerprint
from hallucination_bench.evals.wsbench_chat import bank, score_v2
from hallucination_bench.evals.wsbench_chat import run as wrun
from hallucination_bench.judges import claimmap_v2 as cm
from hallucination_bench.units import split_cell

CONV = cm.conversation_text("Where is the Eiffel Tower?", "It is in Paris⟦READ⟧, France.")


def test_six_verdicts_with_thinky():
    assert tuple(tx.VERDICTS) == (
        "hallucination",
        "probe_candidate",
        "thinky",
        "correct",
        "unresolved",
        "junk",
    )
    assert tx.VERDICTS["thinky"] == ("model_state",)
    assert "model_state" not in tx.PROBE_CANDIDATE_LEAVES
    assert tx.verdict_of("model_state") == tx.V_THINKY


def test_hedged_segments_are_unverifiable_whatever_the_judge_said():
    units = split_cell(
        ["Germany or France or UK. Rome or Paris. Paris or Lyon. A or B. Berlin. Maybe so."]
    )[0]
    raw = {
        "units": [
            e("R0.U1", "false_claim", hedged=True, contradicted_by="Paris"),
            e(
                "R0.U2",
                "substitution_sibling",
                hedged=True,
                contradicted_by="Paris",
                pair={"said": "Rome", "actual": "Paris"},
                probe_target="Rome",
            ),
            e("R0.U3", "correct", hedged=True, source="after", supported_by="Paris"),
            e("R0.U4", "unverifiable", hedged=True),  # already unverifiable: tagged, not counted
            e(
                "R0.U5",
                "substitution_sibling",
                contradicted_by="Paris",
                probe_target="Berlin",
                pair={"said": "Berlin", "actual": "Paris"},
            ),  # a single wrong entity: unchanged
            e("R0.U6", "detail", hedged=True),  # no quote: quote check first, then hedged
        ],
        "rationale": "",
    }
    segs, c = cm.parse(raw, units, CONV)
    got = [(s.leaf, s.raw_leaf, s.same_slot_alternative, s.hedged) for s in segs]
    assert got == [
        ("unverifiable", "false_claim", True, True),
        ("unverifiable", "substitution_sibling", True, True),
        ("unverifiable", "correct", True, True),
        ("unverifiable", "unverifiable", True, True),
        ("substitution_sibling", "substitution_sibling", False, False),
        ("unverifiable", "unverifiable", True, True),  # judge_leaf detail, quote check
    ]
    assert segs[5].judge_leaf == "detail" and segs[5].unquoted_contradiction
    assert c["n_hedged_remapped"] == 3
    assert c["n_probe_candidates"] == 1  # only the single substitution stays a probe candidate
    assert segs[1].probe_target is None and segs[4].probe_target == "Berlin"
    assert [score_v2.hedged_origin(s.to_dict()) for s in segs] == [
        "hallucination",
        "substitution",
        "correct",
        None,
        None,
        None,
    ]


def _judge(user):
    q = prompt_quote(user)
    out = []
    for uid, _est, text in unit_lines(user):
        if "Paris" in text:
            out.append(e(uid, "false_claim", hedged=True, contradicted_by=q))
        else:
            out.append(e(uid, "model_state"))
    return {"units": out, "rationale": "r"}


def test_rebuild_rescored_without_calls_and_counts_origins(tmp_path, prose_readouts):
    out = tmp_path / "out"
    _run([("toy", prose_readouts)], out, MapFake({"map": _judge}), stages=["map"])
    d = wrun.arm_dir(out, wrun.Models(), "toy")
    n_lines = len((d / "cache.jsonl").read_text().splitlines())
    wrun.rebuild_arm("toy", prose_readouts, out, wrun.Models())  # cache only
    assert len((d / "cache.jsonl").read_text().splitlines()) == n_lines
    s2 = json.loads((d / "summary_v2.json").read_text())
    h = s2["health"]
    assert h["n_hedged_remapped"] == 1 and h["n_hedged_from_hallucination"] == 1
    b = s2["overall"]
    assert b["rates"]["verdict_hallucination"]["all"]["value"] == 0
    assert b["rates"]["verdict_thinky"]["all"]["value"] > 0
    assert b["char_share_verdict"]["probe_candidate"]["value"] == 0


def test_consistency_rescore_reparses_cached_answers(tmp_path, prose_readouts):
    # a stored consistency cell (pre-rev-19 leaves) plus its two cached answers
    items = {it["id"]: it for it in bank.load_items()}
    sites = bank.sites_by_item(list(items.values()))
    row = json.loads(prose_readouts.read_text().splitlines()[0])
    key = wrun.cell_key(row["id"], row["layer"], row["pos"])
    samples = [x for x in row["samples"] if x.strip()]
    units = cm.plain_units(split_cell(samples, [[] for _ in samples])[0])
    item, site = items[row["id"]], sites[row["id"]][row["pos"]]
    system, user = cm.render_prompt(item["prompt"], bank.marked_response(item, site), units)
    d = tmp_path / "consistency" / "claude-opus-5-5"
    d.mkdir(parents=True)
    with Cache(d / "cache.jsonl") as cache:
        for r in range(2):
            fp = fingerprint(cm.PROMPT_VERSION, "claude-opus-5-5", "low", f"r{r}", system, user)
            cache.put(cons.cons_key("toy", key, r), fp, {"result": _judge(user), "meta": {}})
    stored = {
        "arm": "toy",
        "key": key,
        "units": [{"id": u.id, "text": u.text} for u in units],
        "r0": {u.id: "false_claim" for u in units},
        "r1": {u.id: "false_claim" for u in units},
    }
    stem = f"consistency_{cm.PROMPT_VERSION}"
    (d / f"{stem}_cells.jsonl").write_text(json.dumps(stored) + "\n")
    (d / f"{stem}.json").write_text(json.dumps({"arms": "toy:44"}))
    res = cons.rescore(tmp_path, "claude-opus-5-5")
    assert res["n_cells_fallback"] == 0
    new = json.loads((d / f"{stem}_cells.jsonl").read_text().splitlines()[0])
    paris = [u["id"] for u in new["units"] if "Paris" in u["text"]]
    assert paris and all(new["r0"][i] == "unverifiable" for i in paris)  # hedged remap
    others = [u["id"] for u in new["units"] if "Paris" not in u["text"]]
    assert all(new["r1"][i] == "model_state" for i in others)
