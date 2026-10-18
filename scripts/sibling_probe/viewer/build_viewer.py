# ruff: noqa: E501
"""Build the plan-0004 sibling-verdict readout viewer (Agam's diagnostic-dashboard style).

Based on ``gw-mlnla/scripts/mlnla/build_wsb_viewer.py`` + ``wsb_readout_viewer.html`` (copied and
adapted, never imported): every item of set A (sibling bank) and set B (basic_readout), the
exact tokenised prompt with clickable read positions (←/→ position, [ ] item), each method's
readouts at all 11 layers (NLA-RL at L42), and the claim badges (validated / consensus /
option / hallucination / unclear / probe n/a / entity substitution) with tooltips (probe z,
detected, gold detected; the judge's per-segment form, quote and why; the rule label).
Filters: set, category, method, layer, items with claims only, Camila's human-label set.

One JSON per item under ``<out>/data/items/`` (lazy-loaded) + ``<out>/data/index.json``.

    uv run python scripts/sibling_probe/viewer/build_viewer.py      # -> outputs/sibling/viewer/
    python3 -m http.server 8767 -d outputs/sibling/viewer
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PAGE = HERE / "readout_viewer.html"
SIB = ROOT / "outputs" / "sibling"
METHODS = {
    "s3d": "O-lens (s3d-rl600)",
    "nla-rl-L42": "NLA-RL (L42)",
    "jlens": "J-lens (top-10)",
    "mlnla-rl600": "ML-NLA RL600 (side)",
}
A_ARMS = {
    "s3d": "s3d-rl600-all11",
    "jlens": "jlens-all11",
    "nla-rl-L42": "nla-rl-L42",
    "mlnla-rl600": "mlnla-rl600-all11",
}
B_ARMS = {
    "s3d": "s3d-rl600",
    "jlens": "jlens",
    "nla-rl-L42": "nla-rl-L42",
    "mlnla-rl600": "mlnla-rl600",
}


def byte_decoder() -> dict[str, int]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs, strict=True)}


class Detok:
    """Token pieces of a prompt from tokenizer.json (no transformers needed)."""

    def __init__(self, path: Path) -> None:
        tj = json.loads(path.read_text())
        self.vocab = {i: s for s, i in tj["model"]["vocab"].items()}
        self.special = {t["id"]: t["content"] for t in tj.get("added_tokens", [])}
        self.bd = byte_decoder()

    def piece_bytes(self, i: int) -> bytes:
        if i in self.special:
            return self.special[i].encode()
        return bytes(self.bd.get(ch, ord("?")) for ch in self.vocab[i])

    def text_of_piece(self, s: str) -> str:
        raw = b"".join(bytes([self.bd[ch]]) if ch in self.bd else ch.encode() for ch in s)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return s

    def spans(self, ids: list[int]) -> tuple[str, list[tuple[int, int]]]:
        pieces = [self.piece_bytes(i) for i in ids]
        whole = b"".join(pieces)
        text = whole.decode("utf-8", errors="replace")
        out, at = [], 0
        for p in pieces:
            s = len(whole[:at].decode("utf-8", errors="ignore"))
            at += len(p)
            e = len(whole[:at].decode("utf-8", errors="ignore"))
            out.append((s, max(e, s)))
        return text, out


def find_tokenizer() -> Path:
    snaps = sorted(
        Path("/root/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots").glob(
            "*/tokenizer.json"
        )
    )
    if not snaps:
        raise SystemExit("Qwen3.6-27B tokenizer.json not in the HF cache")
    return snaps[-1]


def read_jsonl(p: Path):
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


# ------------------------------------------------------------------ the pure writer (tested)


def badge(u: dict, human_uids: set[str]) -> dict[str, Any]:
    j = u.get("judge") or {}
    segs = []
    for k, s in enumerate(u.get("segments") or ()):
        segs.append(
            {
                "readout": s.get("readout"),
                "start": s.get("start"),
                "end": s.get("end"),
                "text": s.get("text"),
                "leaf": s.get("leaf"),
                "form": (j.get("seg_forms") or [None] * (k + 1))[k] if j else None,
                "quote": (j.get("quotes") or [None] * (k + 1))[k] if j else None,
            }
        )
    p = u.get("probe") or {}
    return {
        "uid": u["uid"],
        "method": u["method"],
        "layer": u["layer"],
        "pos": u["pos"],
        "concept": u["concept"],
        "kind": u.get("kind"),
        "category": u.get("category"),
        "form_label": u.get("form_label"),
        "rule": u.get("rule"),
        "consensus": u.get("consensus"),
        "consensus_mlnla": u.get("consensus_mlnla"),
        "probe": {
            k: p.get(k)
            for k in (
                "status",
                "z",
                "det_cat",
                "detected",
                "gold_det",
                "na_reason",
                "validity",
                "transfer",
            )
        },
        "probe_v0": {k: (u.get("probe_v0") or {}).get(k) for k in ("status", "detected", "z")},
        "why": j.get("why") if j else None,
        "segments": segs,
        "human": u["uid"] in human_uids,
    }


def ladder_badge(u: dict, offsets: dict | None = None) -> dict[str, Any]:
    """Plan 0005 ladder unit -> viewer badge. ``offsets``: si -> char offset of the sample in
    the stored (packed) readout string, to underline the mention spans."""
    pv = (u.get("probe") or {}).get("view") or {}
    spans = []
    for h in u.get("hits") or ():
        if h.get("start") is None:
            continue
        off = (offsets or {}).get(h["si"], 0) if h["source"] == "text" else 0
        spans.append({"src": h["source"], "start": h["start"] + off, "end": h["end"] + off})
    cs = u.get("cross") or {}
    return {
        "uid": u["uid"],
        "method": u["method"],
        "layer": u["layer"],
        "eff_layer": u.get("eff_layer", u["layer"]),
        "pos": u["pos"],
        "concept": u["concept"],
        "label": u["label"],
        "weak_null": u.get("weak_null", False),
        "label_any_layer": u.get("label_any_layer"),
        "label_no_prior": u.get("label_no_prior"),
        "label_probe_pm1": u.get("label_probe_pm1"),
        "committed": u.get("committed"),
        "n_committed": u.get("n_committed"),
        "n_not_committed": u.get("n_not_committed"),
        "entity_substitution": u.get("entity_substitution"),
        "quote": u.get("quote"),
        "samples": u.get("samples") or [],
        "probe": {
            "status": (u.get("probe") or {}).get("status"),
            "reason": (u.get("probe") or {}).get("reason"),
            "pair": (u.get("probe") or {}).get("pair"),
            "fire_pm1": (u.get("probe") or {}).get("fire_pm1"),
            "view": pv,
        },
        "cross_primary": (cs.get("primary") or {}).get("layers"),
        "cross_any": (cs.get("any_layer") or {}).get("layers"),
        "cross_no_prior": (cs.get("primary") or {}).get("support_no_prior"),
        "lens_prior": u.get("lens_prior_own"),
        "spans": spans,
    }


def write_viewer(
    items: list[dict],
    readouts: dict,
    units: list[dict],
    data_dir: Path,
    human_uids: set[str] = frozenset(),
    ladder: dict[str, list[dict]] | None = None,
) -> dict:
    """``items``: [{set, id, prompt, prompt_tokens | (text + spans), positions, gold, response,
    category}]; ``readouts``: (set, id) -> method -> pos -> layer -> value; writes
    ``<data_dir>/items/<set>__<id>.json`` and ``<data_dir>/index.json``; returns the index.
    ``ladder`` (plan 0005, optional): set-B item id -> ladder badges (:func:`ladder_badge`)."""
    data_dir = Path(data_dir)
    (data_dir / "items").mkdir(parents=True, exist_ok=True)
    by_item: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for u in units:
        by_item[(u["set"], u["item"])].append(u)
    index: dict[str, Any] = {"methods": METHODS, "items": []}
    for it in items:
        key = f"{it['set']}__{it['id']}"
        bs = [
            badge(u, human_uids)
            for u in sorted(
                by_item.get((it["set"], it["id"]), []),
                key=lambda u: (u["pos"], u["method"], u["layer"]),
            )
        ]
        doc = {k: v for k, v in it.items()} | {
            "key": key,
            "readouts": readouts.get((it["set"], it["id"]), {}),
            "badges": bs,
        }
        lad = (ladder or {}).get(it["id"]) if it["set"] == "B" else None
        if lad is not None:
            doc["ladder"] = lad
        (data_dir / "items" / f"{key}.json").write_text(
            json.dumps(doc, ensure_ascii=False, separators=(",", ":"))
        )
        index["items"].append(
            {
                "key": key,
                "set": it["set"],
                "id": it["id"],
                "category": it.get("category"),
                "gold": it.get("gold"),
                "n_badges": len(bs),
                "cats": dict(Counter(b["category"] for b in bs)),
                "methods": sorted({b["method"] for b in bs}),
                "layers": sorted({b["layer"] for b in bs}),
                "human": any(b["human"] for b in bs),
                "subfamily": it.get("subfamily"),
            }
        )
        if lad is not None:
            index["items"][-1]["ladder"] = dict(Counter(b["label"] for b in lad))
    if ladder is not None:
        index["ladder"] = True
    (data_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False))
    return index


# ------------------------------------------------------------------ inputs


def load_inputs(dt: Detok) -> tuple[list[dict], dict]:
    sys.path.insert(0, str(ROOT / "src"))
    from hallucination_bench.evals.sibling_probe import bank as SB
    from hallucination_bench.evals.sibling_probe import br_bank as BB

    items, readouts = [], defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    # ---- set A
    rows = {
        p["label"]: p
        for p in json.loads((SIB / "render_eval/prompts/L28-36-42-44-52.json").read_text())
    }
    meta = {}
    for f in sorted((SIB / "render_eval/meta").glob("*.json")):
        meta.update(json.loads(f.read_text()))
    for it in SB.load_items():
        p = rows.get(it["id"])
        if p is None:
            continue
        text, spans = dt.spans(p["input_ids"])
        pos = [int(x) for x in meta[it["id"]]["positions_abs"]]
        items.append(
            {
                "set": "A",
                "id": it["id"],
                "category": it["category"],
                "gold": it["gold"],
                "prompt": SB.user_text(it["messages"]),
                "response": it.get("response") or "",
                "text": text,
                "pos": [[q, *spans[q]] for q in pos],
                "subfamily": it.get("source"),
            }
        )
    for m, arm in A_ARMS.items():
        _load_arm(
            readouts,
            "A",
            m,
            sorted((SIB / "readouts" / arm).glob("*.jsonl")),
            dt,
            SIB / "judge" / arm / "cells.jsonl",
        )
    # ---- set B (every item; processing items have no claims)
    for it in BB.load_bank():
        text, spans = dt.spans(it["input_ids"])
        items.append(
            {
                "set": "B",
                "id": it["id"],
                "category": it["category"]
                if it.get("in_pipeline")
                else f"(viewer only) {it['subfamily']}",
                "gold": it.get("gold_concept") or it["target"],
                "prompt": it["prompt"],
                "response": (BB.shown_prefix(it) + " ‸ " if it.get("prefill") else "‸ ")
                + (it.get("response") or ""),
                "text": text,
                "pos": [[it["pos"], *spans[it["pos"]]]],
                "subfamily": it["subfamily"],
            }
        )
    for m, arm in B_ARMS.items():
        _load_arm(
            readouts,
            "B",
            m,
            [SIB / "br" / "readouts" / arm / "basic_readout.jsonl"],
            dt,
            SIB / "judge" / f"br-{arm}" / "cells.jsonl",
        )
    return items, readouts


def _load_arm(readouts, set_, method, files, dt, cells_path: Path) -> None:
    summ = {}
    if method == "jlens" and cells_path.exists():
        for r in read_jsonl(cells_path):
            if r.get("summary"):
                summ[r["key"]] = r["summary"]
    for f in files:
        if not f.exists():
            continue
        for r in read_jsonl(f):
            if "tokens" in r:
                key = f"{r['id']}__L{int(r['layer']):03d}__p{int(r['pos'])}"
                v: Any = {
                    "tokens": [
                        [dt.text_of_piece(t), round(float(s), 2)]
                        for t, s in zip(r["tokens"], r["scores"], strict=False)
                    ],
                    "summary": summ.get(key),
                }
            else:
                v = [s for s in r.get("samples") or [] if s.strip()]
            readouts[(set_, r["id"])][method][str(r["pos"])][str(r["layer"])] = v


def load_ladder(readouts) -> dict[str, list[dict]] | None:
    """Plan 0005 ladder units (+ mention spans from ``mentions.jsonl``) per set-B item."""
    lp = SIB / "ladder" / "ladder" / "units.jsonl"
    mp = SIB / "ladder" / "mentions.jsonl"
    if not lp.exists():
        return None
    sys.path.insert(0, str(ROOT / "src"))
    from hallucination_bench.evals.sibling_probe.ladder import split_samples

    hits = {}
    if mp.exists():
        for m in read_jsonl(mp):
            hits[m["uid"]] = [{"si": s["si"], **h} for s in m["samples"] for h in s["hits"]]
    out: dict[str, list[dict]] = defaultdict(list)
    for u in read_jsonl(lp):
        offs = None
        v = (
            readouts.get(("B", u["item"]), {})
            .get(u["method"], {})
            .get(str(u["pos"]), {})
            .get(str(u["layer"]))
        )
        if isinstance(v, list) and v:
            packed, offs, at = v[0], {}, 0
            for si, t in enumerate(split_samples([packed]) if len(v) == 1 else v):
                k = packed.find(t, at)
                if k >= 0:
                    offs[si] = k
                    at = k + len(t)
        out[u["item"]].append(ladder_badge({**u, "hits": hits.get(u["uid"], [])}, offs))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=SIB / "viewer")
    a = ap.parse_args()
    dt = Detok(find_tokenizer())
    items, readouts = load_inputs(dt)
    up = SIB / "verdict" / "units_labeled.jsonl"
    if not up.exists():
        up = SIB / "verdict" / "units.jsonl"
    units = list(read_jsonl(up)) if up.exists() else []
    hp = ROOT / "data" / "sibling_probe" / "verdict_human_todo.jsonl"
    human = {r["uid"] for r in read_jsonl(hp)} if hp.exists() else set()
    ladder = load_ladder(readouts)
    idx = write_viewer(
        items,
        {k: {m: dict(v) for m, v in d.items()} for k, d in readouts.items()},
        units,
        a.out / "data",
        human,
        ladder,
    )
    (a.out / "index.html").write_text(PAGE.read_text())
    mb = sum(f.stat().st_size for f in (a.out / "data").rglob("*.json")) / 1e6
    nl = sum(len(v) for v in (ladder or {}).values())
    print(
        f"[viewer] {len(idx['items'])} items, {len(units)} badges ({up.name}), {nl} ladder units, {mb:.1f} MB -> {a.out}/index.html"
    )


if __name__ == "__main__":
    main()
