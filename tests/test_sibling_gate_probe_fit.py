"""Rev 8.2: the per-class streaming path (probe_fit) reproduces the old per-layer path."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hallucination_bench.evals.sibling_gate import probe_fit as F
from hallucination_bench.evals.sibling_gate import probes as P

LAYERS = [20, 24, 28]
FILE_LAYERS = [20, 24, 28, 32]  # the acts file stores more layers than the fit uses
D = 12


def _fixture(tmp_path: Path):
    rng = np.random.default_rng(0)
    concepts = ["a", "b", "c"]
    texts, prompts, full = [], [], {}
    for name in concepts:
        for kind in ("explicit", "implied"):
            for i in range(10):
                texts.append({"id": f"{name}|{kind}|{i}", "concept": name, "kind": kind})
    for i in range(10):
        texts.append({"id": f"n|neutral|{i}", "concept": None, "kind": "neutral"})
    folds = P.assign_folds([{**t, "request_idx": 0} for t in texts], salt="toy")
    rows = []
    (tmp_path / "bank").mkdir()
    for j, t in enumerate(texts):
        n_tok = 40
        pos = list(range(5, 5 + 3 * 6, 3))  # 6 read positions, sparse
        c = {"a": 0, "b": 1, "c": 2}.get(t["concept"] or "", -1)
        x = rng.normal(size=(len(FILE_LAYERS), n_tok, D)).astype(np.float32)
        if c >= 0:
            x[:, :, c] += 3.0
        np.save(tmp_path / "bank" / f"{j}.npy", x)
        full[t["id"]] = (x, pos)
        prompts.append({"label": t["id"], "file": f"{j}.npy", "eval_positions": pos})
        rows.append(
            {
                "id": t["id"],
                "c": -1 if c < 0 else c,
                "kind": t["kind"],
                "fold": folds[t["id"]],
                "key": [0, 0, 1, 1, 0, 1],
            }
        )
    (tmp_path / "bank" / "manifest.json").write_text(
        json.dumps({"layers": FILE_LAYERS, "prompts": prompts})
    )
    cls = {"class": "toy", "concepts": concepts, "texts": rows, "merge": list(range(14))}
    return cls, full


def _np_reader(path: str, lo: int, hi: int) -> np.ndarray:
    return np.load(path)[:, lo:hi]


def test_class_windows_equal_direct_slices_and_fit_matches(tmp_path: Path) -> None:
    cls, full = _fixture(tmp_path)
    win16 = F.read_class_windows(
        tmp_path, "bank", cls["texts"], FILE_LAYERS, LAYERS, reader=_np_reader, workers=2
    )
    assert set(win16) == set(full)
    for k, layer in enumerate(LAYERS):
        old = {
            lab: x[FILE_LAYERS.index(layer)][pos].astype(np.float16).astype(np.float32)
            for lab, (x, pos) in full.items()
        }
        new = F.layer_windows(win16, k)
        for lab in old:
            np.testing.assert_array_equal(new[lab], old[lab])
        # the per-class result equals the fit on the directly sliced (old per-layer) windows
        r_old, b_old = F.fit_class_layer("toy", cls, old, layer)
        r_new, b_new = F.fit_class_layer("toy", cls, new, layer)
        assert r_old["validity"] == r_new["validity"]
        assert r_old["null_check"] == r_new["null_check"]
        assert r_old["n_texts"] == r_new["n_texts"] == len(cls["texts"])
        assert b_old is not None and b_old == b_new
    # the toy concepts are separable: every probe is valid
    assert all(v is not None and v > 0.9 for v in r_new["validity"]["a"])


def test_score_class_layer_shapes(tmp_path: Path) -> None:
    cls, full = _fixture(tmp_path)
    win = {lab: x[0][pos].astype(np.float32) for lab, (x, pos) in full.items()}
    _res, blob = F.fit_class_layer("toy", cls, win, 20)
    assert blob is not None
    pos = [1, 2, 3]
    ev = {"item1": (np.random.default_rng(1).normal(size=(3, D)).astype(np.float32), pos)}
    out = np.load(__import__("io").BytesIO(F.score_class_layer(blob, ev, {"item1": {"1": 0}})))
    assert (
        out["za"].shape == (3, 3) and list(out["pos"]) == pos and list(out["item"]) == ["item1"] * 3
    )


def test_class_without_probes_is_class_too_small_not_no_class() -> None:
    from hallucination_bench.evals.sibling_gate import ladder as L

    u = L.UnitProbe("u", "arm", "item", 44, 5, "rare class", "x", "y", [], None)
    assert L.na_reason(None, u, 44) == "class_too_small"
    assert L.na_reason(None, L.UnitProbe("u", "a", "i", 44, 5, "no_class", "x", None, []), 44) == (
        "no_class"
    )


def test_class_windows_merge_capture_shards(tmp_path: Path) -> None:
    cls, _full = _fixture(tmp_path)
    bank = tmp_path / "bank"
    man = json.loads((bank / "manifest.json").read_text())
    half = len(man["prompts"]) // 2
    for i, part in enumerate((man["prompts"][:half], man["prompts"][half:])):
        sd = tmp_path / f"bank-s{i:02d}"
        sd.mkdir()
        for p in part:
            (sd / p["file"]).write_bytes((bank / p["file"]).read_bytes())
        (sd / "manifest.json").write_text(json.dumps({"layers": FILE_LAYERS, "prompts": part}))
    one = F.read_class_windows(tmp_path, "bank", cls["texts"], FILE_LAYERS, LAYERS, _np_reader, 2)
    two = F.read_class_windows(
        tmp_path, "bank-s00,bank-s01", cls["texts"], FILE_LAYERS, LAYERS, _np_reader, 2
    )
    assert set(one) == set(two)
    assert all(np.array_equal(one[k], two[k]) for k in one)


def test_probe_label_has_no_path_separators() -> None:
    from hallucination_bench.evals.sibling_gate.stages import probe_label

    assert (
        probe_label("cls|Oracle Smart View/X\\Y|implied|03")
        == "cls__Oracle_Smart_View_X_Y__implied__03"
    )
