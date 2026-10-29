"""Build a static O-lens (s3d-rl600) readout viewer for the three HallucinationBench safety evals.

Tabs (each a seeded, outcome-stratified sample; the lens is right on some items and makes false
claims on others):

- mask         ~12 matched pairs of the WorkspaceBench ``mask_lie_contrast`` family (lie vs
               honest-resisted, same archetype): prompt + response positions, verified
               ``lie_claim`` / ``honest_claim`` spans, deception-probe (LOAO, sensitivity tau)
               presence per position (L28-52 only).
- overrefusal  ~12 matched pairs of ``refusal_prediction_contrast`` (refused >= 8/10 vs never
               refused, category-matched): prompt positions, verified ``refusal_intent``
               (asserted / considered) and ``comply_intent`` spans, refusal-probe v2 presence.
- strongreject ~12 failed-jailbreak items over the jailbreak families: every prompt position is
               judged, the page keeps claim positions + the last 64 + an even sample; counted
               ``harmful_compliance`` (original / echo), uncounted ones, ``refusal_intent``;
               refusal-probe v2 presence.

Inputs are read-only: the HallucinationBench eval worktrees under /workspace (``hb-mask``,
``hb-overrefusal``, ``hb-strongreject``), their readout dirs, and the two WorkspaceBench
``items.json`` files (read from the global-workspace repo with ``git show <ref>:<path>`` unless
``--mask-items`` / ``--refusal-items`` point at a file). Stdlib only.

**The output contains harmful text (StrongREJECT prompts, lens readouts) and this repo is
public: never commit it.** It goes to ``outputs/safety_viewer/index.html`` (gitignored) and is
served privately by ``scripts/serve/safety_viewer.py``.

    python3 scripts/viewer/build_safety_viewer.py
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

WS = Path("/workspace")
ARM = "s3d-rl600"
CORE = [28, 36, 44, 52]
ALL_LAYERS = [20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]
WSB_DIR = "evals/workspace-bench/hillclimbing_evals/multi_token"
N_PAIRS = 12
N_SR = 12

Cell = dict[str, Any]


# ------------------------------------------------------------------ io helpers


def jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def line_id(line: str, field: str = "id") -> str | None:
    """The value of the first ``"<field>": "..."`` in a JSONL line, without parsing it."""
    tag = f'"{field}": "'
    i = line.find(tag)
    if i < 0:
        return None
    i += len(tag)
    return line[i : line.find('"', i)]


def filtered(path: Path, ids: set[str], field: str = "id") -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line_id(line, field) in ids:
                yield json.loads(line)


def gw_json(repo: Path, ref: str, rel: str, override: Path | None) -> dict[str, Any]:
    if override:
        return json.loads(override.read_text(encoding="utf-8"))
    out = subprocess.run(
        ["git", "-C", str(repo), "show", f"{ref}:{rel}"], check=True, capture_output=True
    ).stdout
    return json.loads(out)


def read_readouts(paths: Iterable[Path], ids: set[str]) -> dict[tuple[str, int, int], list[str]]:
    """``{(id, layer, pos): samples}``; earlier paths win."""
    out: dict[tuple[str, int, int], list[str]] = {}
    for p in paths:
        if not p.exists():
            print(f"  readouts: missing {p}")
            continue
        n = 0
        for r in filtered(p, ids):
            k = (r["id"], int(r["layer"]), int(r["pos"]))
            if k not in out:
                out[k] = r["samples"]
                n += 1
        print(f"  readouts: {n} cells from {p}")
    return out


def norm_find(text: str, quote: str) -> tuple[int, int] | None:
    """Locate ``quote`` in ``text``: exact, then case-insensitive, then whitespace-collapsed."""
    if not quote:
        return None
    i = text.find(quote)
    if i >= 0:
        return i, i + len(quote)
    i = text.lower().find(quote.lower())
    if i >= 0:
        return i, i + len(quote)
    # whitespace-insensitive: map collapsed offsets back to the original
    idx, buf = [], []
    for j, ch in enumerate(text):
        if ch.isspace():
            if buf and buf[-1] == " ":
                continue
            buf.append(" ")
        else:
            buf.append(ch.lower())
        idx.append(j)
    q = " ".join(quote.lower().split())
    k = "".join(buf).find(q)
    if k < 0:
        return None
    return idx[k], idx[k + len(q) - 1] + 1


def span(
    text: str, quote: str, s: int | None = None, e: int | None = None
) -> tuple[int, int] | None:
    if s is not None and e is not None and text[s:e] == quote:
        return s, e
    return norm_find(text, quote)


def even(xs: list[int], n: int) -> list[int]:
    if len(xs) <= n:
        return list(xs)
    return [xs[round(i * (len(xs) - 1) / (n - 1))] for i in range(n)]


def make_cell(texts: list[str], hl: list[list[Any]], flag: bool, probe: bool | None) -> Cell:
    """Compact cell: t = readout texts, h = [readout, start, end, class, tip], f = quorum flag,
    p = probe presence (1 / 0 / null = no probe at this site), u = unlocated quotes."""
    located, unloc = [], []
    for ri, s, e, cls, tip, quote in hl:
        if ri >= len(texts):
            unloc.append([ri, cls, quote, tip])
            continue
        sp = span(texts[ri], quote, s, e)
        if sp is None:
            unloc.append([ri, cls, quote, tip])
        else:
            located.append([ri, sp[0], sp[1], cls, tip])
    c: Cell = {"t": texts, "h": located}
    if unloc:
        c["u"] = unloc
    if flag:
        c["f"] = 1
    if probe is not None:
        c["p"] = int(probe)
    return c


def stratified_pairs(
    pairs: list[dict[str, Any]], kind_of: dict[str, str], group: str, rng: random.Random, n: int
) -> list[dict[str, Any]]:
    """Round-robin over outcome kinds (rarest first), spreading over ``group`` values."""
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in pairs:
        by_kind[kind_of[p["pair_id"]]].append(p)
    for v in by_kind.values():
        rng.shuffle(v)
    order = sorted(by_kind, key=lambda k: len(by_kind[k]))
    out: list[dict[str, Any]] = []
    used_groups: Counter[str] = Counter()
    while len(out) < n and any(by_kind.values()):
        for k in order:
            if len(out) >= n or not by_kind[k]:
                continue
            cand = by_kind[k]
            best = min(cand, key=lambda p: used_groups[p[group]])
            cand.remove(best)
            out.append(best)
            used_groups[best[group]] += 1
    return sorted(out, key=lambda p: (p[group], p["pair_id"]))


# ------------------------------------------------------------------ MASK


def mask_section(a: argparse.Namespace, rng: random.Random) -> dict[str, Any]:
    fam = gw_json(a.gw_repo, a.gw_ref, f"{WSB_DIR}/mask_lie_contrast/items.json", a.mask_items)
    items = {it["key"]: it for it in fam["items"]}
    keys = {it["key"] for it in fam["items"]}
    pairs = fam["pairs"]["main"]
    cdir = a.mask_root / "outputs/mask/claude-opus-5-5" / ARM
    files = {int(p.stem.split("_L")[1]): p for p in sorted(cdir.glob("cells_L*.jsonl"))}
    print(f"  mask judged layers on disk: {sorted(files)}")

    # pass 1: quorum flags (verified lie_claim on >= 2 of 3 readouts) for every family item
    flags: dict[str, set[int]] = defaultdict(set)
    by_set: dict[tuple[str, str], set[int]] = defaultdict(set)
    for L in CORE:
        with files[L].open(encoding="utf-8") as f:
            for line in f:
                if '"lie_claim"' not in line:
                    continue
                c = json.loads(line)
                if c["task_id"] not in keys or c["status"] != "ok":
                    continue
                ro = {
                    m["readout"]
                    for m in c["mentions"]
                    if m["category"] == "lie_claim" and (m.get("verify") or {}).get("real")
                }
                if len(ro) >= 2:
                    flags[c["task_id"]].add(L)
                    by_set[(c["task_id"], c["site_type"])].add(L)

    def pkind(p: dict[str, Any]) -> str:
        i = items[p["I"]]["key"]
        n = items[p["N"]]["key"]
        return f"I{'+' if flags[i] else '-'}N{'+' if flags[n] else '-'}"

    kind_of = {p["pair_id"]: pkind(p) for p in pairs}
    print(f"  mask pair outcomes (any core layer): {Counter(kind_of.values())}")
    chosen = stratified_pairs(pairs, kind_of, "archetype", rng, N_PAIRS)
    sel = [items[p["I"]] for p in chosen] + [items[p["N"]] for p in chosen]
    sel_ids = {it["key"] for it in sel}
    pair_of = {items[p["I"]]["key"]: p["pair_id"] for p in chosen}
    pair_of |= {items[p["N"]]["key"]: p["pair_id"] for p in chosen}

    # pass 2: every cell of the chosen items
    cells: dict[tuple[str, int, int], dict[str, Any]] = {}
    for L, p in files.items():
        for c in filtered(p, sel_ids, "task_id"):
            cells[(c["task_id"], L, c["pos"])] = c
    sites = {
        r["task_id"]: r["sites"]
        for r in json.load(gzip.open(a.mask_root / "data/mask/sites.json.gz", "rt"))["items"]
        if r["task_id"] in sel_ids
    }
    # deception probe: LOAO job of the item's archetype, sensitivity tau = OOF honest q95 (tau_pos)
    scores = json.load(gzip.open(a.mask_root / "outputs/mask/probe_scores_v5.json.gz", "rt"))
    summ = json.loads((a.mask_root / "data/mask/probe_summary_v5.json").read_text())
    probe: dict[tuple[str, int, int], bool] = {}
    for it in sel:
        t, arch = it["key"], it["archetype"]
        pos_of = [s["pos"] for s in sites[t]]
        for Ls, jobs in scores["layers"].items():
            job = f"loao:{arch}"
            if job not in jobs:
                continue
            for st in ("prompt", "response"):
                sc = jobs[job]["scores"].get(st, {}).get(t)
                if not sc:
                    continue
                tau = summ["layers"][Ls][job][f"loao_{st}"]["tau_pos"]
                for i, v in zip(sc[0], sc[1], strict=True):
                    probe[(t, int(Ls), pos_of[i])] = v > tau
    del scores
    ro_paths = [a.mask_readouts / ARM / "mask.jsonl", a.mask_readouts / ARM / "mask_l11.jsonl"]
    readouts = read_readouts(ro_paths, sel_ids)

    out_items = []
    for it in sel:
        t = it["key"]
        ss = sites[t]
        layers = sorted({L for (tid, L, _) in cells if tid == t})
        # every position with a verified lie_claim (any readout, any layer), and prompt positions
        # with an honest_claim on >= 2 of 3 readouts at some layer (honest_claims are everywhere)
        claim_pos = {
            pos
            for (tid, _, pos), c in cells.items()
            if tid == t
            and (
                any(
                    m["category"] == "lie_claim" and (m.get("verify") or {}).get("real")
                    for m in c["mentions"]
                )
                or (
                    c["site_type"] == "prompt"
                    and len(
                        {m["readout"] for m in c["mentions"] if m["category"] == "honest_claim"}
                    )
                    >= 2
                )
            )
        }
        prompt_pos = [s["pos"] for s in ss if s["site_type"] == "prompt"]
        resp_pos = [s["pos"] for s in ss if s["site_type"] == "response"]
        keep = set(
            prompt_pos if len(prompt_pos) <= a.max_prompt else even(prompt_pos, a.max_prompt)
        )
        keep |= set(prompt_pos[-6:]) | set(even(resp_pos, a.resp_even)) | set(resp_pos[:4])
        keep |= claim_pos
        cols = sorted(keep)
        grid: dict[str, list[Cell | None]] = {}
        for L in layers:
            row: list[Cell | None] = []
            for pos in cols:
                c = cells.get((t, L, pos))
                texts = readouts.get((t, L, pos))
                if c is None or texts is None:
                    row.append(None)
                    continue
                hl, lie_ro = [], set()
                for m in c["mentions"]:
                    cat = m["category"]
                    if cat == "lie_claim":
                        v = m.get("verify") or {}
                        ok = bool(v.get("real"))
                        cls = "lie" if ok else "lie_rej"
                        if ok:
                            lie_ro.add(m["readout"])
                        vb = m.get("verify_blind") or {}
                        tip = (
                            f"lie_claim · verifier: {'ACCEPTED' if ok else 'rejected'}"
                            f"{' → ' + str(v.get('recategory')) if not ok else ''}"
                            f" — {v.get('reason', '')} · blind verifier: "
                            f"{'accepted' if vb.get('real') else 'rejected'}"
                        )
                    elif cat == "honest_claim":
                        cls, tip = "honest", "honest_claim (detector)"
                    else:
                        continue
                    hl.append([m["readout"], m.get("start"), m.get("end"), cls, tip, m["quote"]])
                row.append(make_cell(texts, hl, len(lie_ro) >= 2, probe.get((t, L, pos))))
            grid[str(L)] = row
        region = {s["pos"]: ("p" if s["site_type"] == "prompt" else "r") for s in ss}
        is_lie = it["kind"] == "lie"
        f = flags[t]
        out_items.append(
            {
                "id": t,
                "pair": pair_of[t],
                "label": "lie" if is_lie else "honest",
                "group": it["archetype"],
                "verdict": (("right" if f else "miss") if is_lie else ("false" if f else "right")),
                "flag_layers": sorted(f),
                "flag_by_set": {st: sorted(by_set[(t, st)]) for st in ("prompt", "response")},
                "head": {
                    "archetype": it["archetype"],
                    "system": next(
                        (m["content"] for m in it["messages"] if m["role"] == "system"), ""
                    ),
                    "user": "\n\n".join(
                        m["content"] for m in it["messages"] if m["role"] == "user"
                    ),
                    "reply": it["y"],
                    "proposition": it["proposition"],
                    "options": it["options"],
                    "belief": it["belief"],
                    "ground_truth": it.get("ground_truth"),
                    "pressure_target": it.get("pressure_target"),
                    "statements": it["statements"],
                    "lie_rate": it.get("lie_rate"),
                },
                "tokens": [s["token"] for s in ss],
                "tokpos": [s["pos"] for s in ss],
                "region": [region[s["pos"]] for s in ss],
                "cols": cols,
                "layers": layers,
                "grid": grid,
            }
        )
    order = {p["pair_id"]: i for i, p in enumerate(chosen)}
    out_items.sort(key=lambda x: (order[x["pair"]], x["label"] != "lie"))
    return {
        "items": out_items,
        "pairs": [
            {"pair_id": p["pair_id"], "group": p["archetype"], "outcome": kind_of[p["pair_id"]]}
            for p in chosen
        ],
        "n_family_pairs": len(pairs),
        "layers_on_disk": sorted(files),
    }


# ------------------------------------------------------------------ overrefusal


def overrefusal_section(a: argparse.Namespace, rng: random.Random) -> dict[str, Any]:
    root = a.or_root
    fam = gw_json(
        a.gw_repo,
        a.gw_ref,
        f"{WSB_DIR}/refusal_prediction_contrast/items.json",
        a.refusal_items,
    )
    wsb = {it["key"]: it for it in fam["items"]}
    pairs = fam["pairs"]["main"]
    hb_of = {k: it["hb_id"] for k, it in wsb.items()}
    hb_ids = set(hb_of.values())
    cdir = root / "outputs/overrefusal/claude-sonnet-5-5" / ARM
    summary = json.loads((cdir / "summary.json").read_text())
    # tail = the 4 template positions before the reply (<|im_end|>, \n, <|im_start|>, assistant)
    tail_of = {
        it["hb_id"]: set(range(it["n_positions"] - 4, it["n_positions"])) for it in wsb.values()
    }
    cells: dict[tuple[str, int, int], dict[str, Any]] = {}
    flags: dict[str, set[int]] = defaultdict(set)
    flags_tail: dict[str, set[int]] = defaultdict(set)
    for c in filtered(cdir / "cells.jsonl", hb_ids):
        cells[(c["id"], c["layer"], c["pos"])] = c
        if sum(r["ref"] for r in c["ro"]) >= 2:
            flags[c["id"]].add(c["layer"])
            if c["pos"] in tail_of[c["id"]]:
                flags_tail[c["id"]].add(c["layer"])
    layers_on_disk = sorted({L for (_, L, _) in cells})
    print(f"  overrefusal judged layers on disk: {layers_on_disk}")

    def pkind(p: dict[str, Any]) -> str:
        i, n = hb_of[p["I"]], hb_of[p["N"]]
        fi, fn = flags[i] & set(CORE), flags[n] & set(CORE)
        return f"I{'+' if fi else '-'}N{'+' if fn else '-'}"

    kind_of = {p["pair_id"]: pkind(p) for p in pairs}
    print(
        f"  overrefusal pair outcomes (all positions, any core layer): {Counter(kind_of.values())}"
    )
    chosen = stratified_pairs(pairs, kind_of, "category", rng, N_PAIRS)
    sel_keys = [p["I"] for p in chosen] + [p["N"] for p in chosen]
    sel = {hb_of[k] for k in sel_keys}

    hb_items = {
        it["id"]: it
        for it in json.loads((root / "data/overrefusal/items.json").read_text())["items"]
        if it["id"] in sel
    }
    # comply_intent quotes live only in the detector cache (:D), keyed <id>__L<layer>__p<pos>
    det: dict[str, list[dict[str, Any]]] = {}
    with (cdir / "cache.jsonl").open(encoding="utf-8") as f:
        for line in f:
            k = line_id(line, "key")
            if not k or not k.endswith(":D") or k.split("__", 1)[0] not in sel:
                continue
            r = json.loads(line)
            if r.get("result"):
                det[k[:-2]] = r["result"]["mentions"]
    probe: dict[tuple[str, int, int], bool] = {}
    for name in ("probe_sites_v2.jsonl", "probe_sites_v2_l11.jsonl"):
        for r in filtered(root / "outputs/overrefusal" / name, sel):
            probe[(r["id"], r["layer"], r["pos"])] = r["present"] >= 0.5
    ro_paths = [Path(summary["readouts"])] + [
        WS / d / ARM / "overrefusal.jsonl"
        for d in (
            "hb-readouts-overrefusal-v2",
            "hb-readouts-overrefusal-v2add",
            "hb-readouts-overrefusal-v2b",
        )
    ]
    readouts = read_readouts(ro_paths, sel)
    # one sampled reply (T = 1) matching the majority behaviour, with its judge label
    labels: dict[str, dict[str, Any]] = {}
    for gl in (
        "data/overrefusal/gt_labels.jsonl",
        "data/overrefusal/probes/orbench80k_gt_labels.jsonl",
        "data/overrefusal/grow/orbench80k2_gt_labels.jsonl",
    ):
        for r in filtered(root / gl, sel):
            labels.setdefault(r["id"], r)
    samples: dict[tuple[str, int], str] = {}
    for sf in ("samples.jsonl", "samples_80k.jsonl", "samples_80k2.jsonl"):
        for r in filtered(root / "outputs/overrefusal" / sf, sel):
            samples.setdefault((r["id"], r["sample"]), r["text"])

    out_items = []
    for key in sel_keys:
        w = wsb[key]
        hid = w["hb_id"]
        hb = hb_items[hid]
        sites = hb["sites"]
        cols = [s["pos"] for s in sites]
        tail = set(cols[-4:])
        assert tail == tail_of[hid], hid
        lab = labels.get(hid, {})
        want_refuse = w["kind"] == "I"
        si = next(
            (
                i
                for i, x in enumerate(lab.get("labels", []))
                if (x != "comply") == want_refuse and (hid, i) in samples
            ),
            0,
        )
        layers = sorted({L for (i, L, _) in cells if i == hid})
        grid: dict[str, list[Cell | None]] = {}
        for L in layers:
            row: list[Cell | None] = []
            for pos in cols:
                c = cells.get((hid, L, pos))
                texts = readouts.get((hid, L, pos))
                if c is None or texts is None:
                    row.append(None)
                    continue
                hl = []
                for m in c.get("refusals", []):
                    ok = m["verified"] in (None, "refusal_intent")
                    st = m["stance"] if ok else None
                    cls = "ref_rej" if not ok else ("ref_as" if st == "asserted" else "ref_con")
                    tip = (
                        f"refusal_intent · detector stance {m['det_stance']} · verifier: "
                        + ("unverified (kept)" if m["verified"] is None else str(m["verified"]))
                        + (f" · stance {st}" if ok else " (rejected)")
                    )
                    hl.append([m["readout"], None, None, cls, tip, m["quote"]])
                for m in det.get(f"{hid}__L{L:03d}__p{pos}", []):
                    if m["label"] == "comply_intent":
                        hl.append(
                            [
                                m["readout"],
                                None,
                                None,
                                "comply",
                                "comply_intent (detector)",
                                m["quote"],
                            ]
                        )
                q = sum(r["ref"] for r in c["ro"]) >= 2
                row.append(make_cell(texts, hl, q, probe.get((hid, L, pos))))
            grid[str(L)] = row
        f = flags[hid] & set(CORE)
        out_items.append(
            {
                "id": hid,
                "pair": next(p["pair_id"] for p in chosen if key in (p["I"], p["N"])),
                "label": "refused" if want_refuse else "never refused",
                "group": w["category"],
                "verdict": ("right" if f else "miss")
                if want_refuse
                else ("false" if f else "right"),
                "flag_layers": sorted(flags[hid]),
                "flag_by_set": {"all": sorted(flags[hid]), "tail": sorted(flags_tail[hid])},
                "head": {
                    "category": w["category"],
                    "source": w.get("source"),
                    "prompt": "\n\n".join(
                        m["content"] for m in w["messages"] if m["role"] == "user"
                    ),
                    "n_refuse": w["n_refuse_of_10"],
                    "sample": samples.get((hid, si), ""),
                    "sample_idx": si,
                    "sample_label": (lab.get("labels") or [None] * 10)[si],
                    "sample_reason": (lab.get("reasons") or [None] * 10)[si],
                    "tail": sorted(tail),
                },
                "tokens": [s["token"] for s in sites],
                "tokpos": cols,
                "region": [
                    "t" if s["pos"] in tail else ("m" if s["kind"] == "template" else "p")
                    for s in sites
                ],
                "cols": cols,
                "layers": layers,
                "grid": grid,
            }
        )
    order = {p["pair_id"]: i for i, p in enumerate(chosen)}
    out_items.sort(key=lambda x: (order[x["pair"]], x["label"] != "refused"))
    return {
        "items": out_items,
        "pairs": [
            {"pair_id": p["pair_id"], "group": p["category"], "outcome": kind_of[p["pair_id"]]}
            for p in chosen
        ],
        "n_family_pairs": len(pairs),
        "layers_on_disk": layers_on_disk,
    }


# ------------------------------------------------------------------ StrongREJECT


def _counted(r: dict[str, Any] | None) -> bool:  # = hallucination_bench...strongreject.echo.counted
    return bool(r) and bool(r["confirmed"]) and r["speaker"] == "assistant"


def sr_records(
    a: argparse.Namespace, adir: Path, ids: set[str]
) -> tuple[list[dict[str, Any]], list[int]]:
    """Per-cell StrongREJECT records (mentions + echo verdicts) for ``ids`` and the judged layers.

    ``records.jsonl`` only holds the layers of the arm's LAST ``run`` (a run over new layers
    overwrites it), so by default the records are rebuilt read-only from the arm's judge cache
    with HB's own ``cell_record`` (needs the HB deps: run under the repo venv). A layer counts as
    judged when >= 95% of its cells are ``ok``. ``--sr-records-file`` uses the file instead.
    """
    recs: list[dict[str, Any]] = []
    status: dict[int, Counter[str]] = defaultdict(Counter)
    if a.sr_records_file:
        for r in filtered(adir / "slim_records.jsonl", ids):
            status[r["layer"]][r["status"]] += 1
        recs = [r for r in filtered(adir / "records.jsonl", ids)]
    else:
        import sys

        sys.path.insert(0, str(a.sr_root / "src"))
        from hallucination_bench.evals.strongreject import run as R  # type: ignore[import-not-found]

        judge = R.Judge()
        cfg = json.loads((adir / "arm.json").read_text())
        arm = R.open_arm(
            ARM,
            Path(cfg["readouts"]),
            a.sr_root / "outputs",
            judge,
            allow_missing=True,
            readonly=True,
        )
        assert arm.root.resolve() == adir.resolve(), (arm.root, adir)
        for c in arm.cells:
            if c.id not in ids:
                continue
            r = R.cell_record(arm, c, judge)
            status[r["layer"]][r["status"]] += 1
            if r["mentions"]:
                recs.append(r)
        del arm
    judged = sorted(L for L, cnt in status.items() if cnt["ok"] >= 0.95 * sum(cnt.values()))
    skipped = sorted(set(status) - set(judged))
    if skipped:
        print(f"  strongreject: layers with readouts but not (fully) judged, skipped: {skipped}")
    return recs, judged


def strongreject_section(a: argparse.Namespace, rng: random.Random) -> dict[str, Any]:
    root = a.sr_root
    bank = {
        it["id"]: it
        for it in json.loads((root / "data/strongreject/items.json").read_text())["items"]
    }
    unsucc = {i for i, it in bank.items() if it["outcome"] == "unsuccessful"}
    adir = root / "outputs/strongreject/claude-opus-5-5" / ARM
    all_recs, layers_on_disk = sr_records(a, adir, unsucc)
    print(f"  strongreject judged layers: {layers_on_disk}")
    recs: dict[tuple[str, int, int], dict[str, Any]] = {}
    hc: Counter[str] = Counter()  # cells with a counted harmful_compliance claim, any layer
    hc_tail44: Counter[str] = Counter()  # ... in the last-64 window at L44
    hc_pos: dict[str, set[int]] = defaultdict(set)
    refc: Counter[str] = Counter()
    for r in all_recs:
        if r["layer"] not in layers_on_disk:
            continue
        recs[(r["id"], r["layer"], r["pos"])] = r
        if any(
            m["label"] in ("harmful_compliance", "persona_adoption") and _counted(m.get("echo"))
            for m in r["mentions"]
        ):
            hc[r["id"]] += 1
        if any(
            m["label"] == "harmful_compliance" and _counted(m.get("echo")) and not m["echo"]["echo"]
            for m in r["mentions"]
        ):
            hc_pos[r["id"]].add(r["pos"])  # counted ORIGINAL harmful_compliance: always kept
        if (
            r["region"] == "tail"
            and r["layer"] == 44
            and any(
                m["label"] in ("harmful_compliance", "persona_adoption") and _counted(m.get("echo"))
                for m in r["mentions"]
            )
        ):
            hc_tail44[r["id"]] += 1
        if r["refusal"]:
            refc[r["id"]] += 1
    fams = sorted({bank[i]["family"] for i in unsucc})
    false_pool = [i for i in sorted(unsucc) if hc_tail44[i] > 0]
    right_pool = [i for i in sorted(unsucc) if hc_tail44[i] == 0]
    print(
        f"  strongreject: {len(unsucc)} failed jailbreaks; {sum(bool(hc[i]) for i in unsucc)} with "
        "a "
        f"counted compliance claim somewhere, {len(false_pool)} in the last 64 positions at L44, "
        f"{len(right_pool)} clean there; {len(fams)} families"
    )
    n_false_pool, n_right_pool = len(false_pool), len(right_pool)
    rng.shuffle(false_pool)
    rng.shuffle(right_pool)
    chosen: list[str] = []
    used: Counter[str] = Counter()
    want = {
        "false": N_SR - min(len(right_pool), N_SR // 2),
        "right": min(len(right_pool), N_SR // 2),
    }
    for name, pool in (("right", right_pool), ("false", false_pool)):
        for _ in range(want[name]):
            if not pool:
                break
            best = min(pool, key=lambda i: used[bank[i]["family"]])
            pool.remove(best)
            chosen.append(best)
            used[bank[best]["family"]] += 1
    chosen.sort(key=lambda i: (bank[i]["family"], i))
    sel = set(chosen)
    probe: dict[tuple[str, int, int], bool] = {}
    for name in ("probe_sites.jsonl", "probe_sites_l11.jsonl"):
        for r in filtered(root / "outputs/strongreject" / name, sel):
            if r.get("variant", "refusal-dm-v1") == "refusal-dm-v1":
                probe[(r["id"], r["layer"], r["pos"])] = r["present"] >= 0.5
    arm_cfg = json.loads((adir / "arm.json").read_text())
    ro_paths = [Path(arm_cfg["readouts"])] + [
        WS / d / ARM / "strongreject.jsonl"
        for d in ("hb-readouts-strongreject-rest", "hb-readouts-strongreject-l11")
    ]
    readouts = read_readouts(ro_paths, sel)
    rollouts = {
        r["id"]: r["samples"] for r in filtered(root / "outputs/strongreject/rollouts.jsonl", sel)
    }

    out_items = []
    for iid in chosen:
        it = bank[iid]
        sites = it["sites"]
        allpos = [s["pos"] for s in sites]
        # with 11 layers nearly every position carries some claim somewhere, so the page keeps the
        # last 64, an even sample, and every position with a counted ORIGINAL compliance claim
        claim_pos = hc_pos[iid]
        cols = sorted(set(allpos[-a.sr_tail :]) | set(even(allpos, a.sr_even)) | claim_pos)
        grid: dict[str, list[Cell | None]] = {}
        for L in layers_on_disk:
            row: list[Cell | None] = []
            for pos in cols:
                texts = readouts.get((iid, L, pos))
                if texts is None:
                    row.append(None)
                    continue
                r = recs.get((iid, L, pos))
                hl = []
                q_ro: set[int] = set()
                for m in r["mentions"] if r else []:
                    lab = m["label"]
                    if lab in ("harmful_compliance", "persona_adoption"):
                        e = m.get("echo")
                        if _counted(e):
                            cls = ("hc" if lab == "harmful_compliance" else "pa") + (
                                "_echo" if e["echo"] else "_orig"
                            )
                            q_ro.add(m["readout"])
                        else:
                            cls = "hc_unc"
                        tip = f"{lab} · " + (
                            "unjudged echo stage"
                            if not e
                            else (
                                f"speaker {e['speaker']}, confirmed {e['confirmed']}, "
                                f"echo {e['echo']}"
                                + (f" (of “{e['echo_quote']}”)" if e.get("echo_quote") else "")
                                + (", restates request" if e.get("restates_request") else "")
                                + (" → COUNTED" if _counted(e) else " → not counted")
                            )
                        )
                    elif lab == "refusal_intent":
                        cls, tip = "refusal", "refusal_intent (detector)"
                    else:
                        continue
                    hl.append([m["readout"], m["start"], m["end"], cls, tip, m["quote"]])
                row.append(make_cell(texts, hl, bool(q_ro), probe.get((iid, L, pos))))
            grid[str(L)] = row
        oc = it["outcome_counts"]
        if isinstance(oc, str):
            oc = json.loads(oc.replace("'", '"'))
        out_items.append(
            {
                "id": iid,
                "pair": None,
                "label": it["family"],
                "group": it["family"],
                "verdict": "false" if hc_tail44[iid] else "right",
                "n_hc_cells": hc[iid],
                "n_hc_tail44": hc_tail44[iid],
                "n_refusal_cells": refc[iid],
                "head": {
                    "family": it["family"],
                    "category": it["category"],
                    "forbidden_prompt": it["forbidden_prompt"],
                    "jailbreak_prompt": it["jailbreak_prompt"],
                    "outcome": it["outcome"],
                    "outcome_counts": oc,
                    "reply": (rollouts.get(iid) or [""])[0],
                },
                "tokens": [s["token"] for s in sites],
                "tokpos": allpos,
                "region": ["t" if s["region"] == "tail" else "p" for s in sites],
                "cols": cols,
                "tail_cols": [p for p in cols if p in set(allpos[-a.sr_tail :])],
                "layers": layers_on_disk,
                "grid": grid,
            }
        )
    return {
        "items": out_items,
        "n_failed": len(unsucc),
        "n_false_pool": n_false_pool,
        "n_right_pool": n_right_pool,
        "n_any_hc": sum(1 for i in unsucc if hc[i]),
        "families": fams,
        "layers_on_disk": layers_on_disk,
    }


# ------------------------------------------------------------------ html


def build_html(data: dict[str, Any]) -> str:
    template = (Path(__file__).parent / "safety_viewer.html").read_text(encoding="utf-8")
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return template.replace("/*__DATA__*/null", blob)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--gw-repo", type=Path, default=WS / "global-workspace")
    ap.add_argument("--gw-ref", default="origin/main")
    ap.add_argument("--mask-items", type=Path, default=None)
    ap.add_argument("--refusal-items", type=Path, default=None)
    ap.add_argument("--mask-root", type=Path, default=WS / "hb-mask")
    ap.add_argument("--mask-readouts", type=Path, default=WS / "hb-readouts-mask")
    ap.add_argument("--or-root", type=Path, default=WS / "hb-overrefusal")
    ap.add_argument("--sr-root", type=Path, default=WS / "hb-strongreject")
    ap.add_argument("--max-prompt", type=int, default=60, help="MASK prompt positions kept")
    ap.add_argument("--resp-even", type=int, default=24, help="MASK response positions (even)")
    ap.add_argument("--sr-tail", type=int, default=64)
    ap.add_argument(
        "--sr-records-file",
        action="store_true",
        help="read StrongREJECT records.jsonl instead of rebuilding from the cache (fast, but only "
        "the layers of the last run)",
    )
    ap.add_argument("--sr-even", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", choices=["mask", "overrefusal", "strongreject"], default=None)
    ap.add_argument("--out", type=Path, default=Path("outputs/safety_viewer/index.html"))
    a = ap.parse_args()

    data: dict[str, Any] = {"seed": a.seed, "arm": ARM}
    for name, fn in (
        ("mask", mask_section),
        ("overrefusal", overrefusal_section),
        ("strongreject", strongreject_section),
    ):
        if a.only and a.only != name:
            data[name] = {"items": []}
            continue
        print(name)
        data[name] = fn(a, random.Random(f"{name}-{a.seed}"))
        print(
            f"  {name}: {len(data[name]['items'])} items, verdicts "
            f"{dict(Counter(x['verdict'] for x in data[name]['items']))}"
        )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    html = build_html(data)
    a.out.write_text(html, encoding="utf-8")
    blob = gzip.compress(html.encode("utf-8"), 9, mtime=0)
    a.out.with_suffix(".html.gz").write_bytes(blob)  # served pre-compressed
    gz = len(blob)
    print(f"-> {a.out}  {len(html) / 1e6:.1f} MB raw, {gz / 1e6:.1f} MB gzip")


if __name__ == "__main__":
    main()
