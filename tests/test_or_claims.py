"""Plan 0020 rev 7: orclaim-v1 parser, claim dedup / provenance, ratio CI, consistency, index."""

from typing import Any

from hallucination_bench.evals.overrefusal import index_export, sclaim


def _raw(*lines: list[tuple[str, str]]) -> dict[str, Any]:
    return {
        "lines": [
            {"line": i + 1, "statements": [{"label": lab, "text": t} for lab, t in sts]}
            for i, sts in enumerate(lines)
        ]
    }


def test_parse_cl_fails_closed() -> None:
    lines = ["[few] - I can't help. - Here is how", "[one] topic"]
    ok = _raw(
        [("refusal_claim", "I can't help."), ("answer_claim", "Here is how")], [("neither", "")]
    )
    got = sclaim.parse_cl(ok, lines)
    assert got is not None and [s["label"] for s in got[0]] == ["refusal_claim", "answer_claim"]
    assert sclaim.parse_cl(_raw([("neither", "")]), lines) is None  # a line missing
    assert sclaim.parse_cl(_raw([("refusal_claim", " ")], [("neither", "")]), lines) is None
    assert sclaim.parse_cl(_raw([], [("neither", "")]), lines) is None  # empty statements
    bad = _raw([("maybe", "x")], [("neither", "")])
    assert sclaim.parse_cl(bad, lines) is None


def test_parse_cl_token_lines() -> None:
    lines = ["[one] a | 我无法 | b"]
    got = sclaim.parse_cl(_raw([("refusal_claim", "我无法")]), lines, tokens=True)
    assert got == [[{"label": "refusal_claim", "text": lines[0]}]]  # the whole line is the claim
    assert sclaim.parse_cl(_raw([("answer_claim", "a")]), lines, tokens=True) is None
    two = _raw([("neither", ""), ("neither", "")])
    assert sclaim.parse_cl(two, lines, tokens=True) is None
    # a prose line that merely contains " | " is not a token line
    assert sclaim.parse_cl(two, lines) is not None


def test_claims_dedup_and_provenance_union() -> None:
    labels = [
        [{"label": "refusal_claim", "text": "I can't  help."}, {"label": "neither", "text": ""}],
        [
            {"label": "refusal_claim", "text": "i can't help."},
            {"label": "answer_claim", "text": "Ok"},
        ],
    ]
    cl = sclaim.claims_of(labels, [[3, 4], [9]])
    ref = [c for c in cl if c["label"] == "refusal_claim"]
    assert len(cl) == 2 and len(ref) == 1 and ref[0]["cells"] == [3, 4, 9]


def _row(item: str, n: int, ref: int, hal: int, verdict: str = "no_refusal") -> dict[str, Any]:
    return {
        "reader": "s3d-rl600",
        "layer": 44,
        "item": item,
        "verdict": verdict,
        "claims": [{}] * n,
        "n_claims": n,
        "n_refusal": ref,
        "n_hallucinated": hal,
        "n_unprobed": 0,
        "n_backed": ref - hal,
    }


def test_metrics_pooled_ratio_and_absent_layer() -> None:
    rows = [
        _row("a", 8, 2, 1, "refusal"),
        _row("b", 2, 0, 0),
        {**_row("c", 0, 0, 0), "claims": None},
    ]
    m = sclaim.metrics(rows)["s3d-rl600"]["44"]
    assert m["claim_validated"]["mean"] == 0.1 and m["claim_raw"]["mean"] == 0.2
    assert m["claims_per_item"] == 5.0 and m["health"]["n_unjudged"] == 1
    assert sclaim.ratio_ci([0], [0])["mean"] is None  # 0 claims -> absent
    assert sclaim.consistency(rows) == {
        "claim_and_refusal": 1,
        "claim_only": 0,
        "refusal_only": 0,
        "neither": 1,
    }


def test_index_rows_claim_columns() -> None:
    items = [
        {"id": "a", "population": "never_refused", "sites": []},
        {"id": "b", "population": "never_refused", "sites": []},
    ]
    rows = sclaim.index_rows([_row("a", 8, 2, 1)], items)
    assert [
        (r["item"], r["n_claims"], r["n_false"], r["n_contradicted"], r["n_failed_cells"])
        for r in rows
    ] == [("a", 8, 2, 1, 0), ("b", 0, 0, 0, 1)]
    assert set(rows[0]) == set(index_export.COLUMNS)
