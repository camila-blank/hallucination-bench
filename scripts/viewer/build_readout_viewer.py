"""Build a static readout viewer: ~10 judged examples from each HallucinationBench eval.

Evals and where their judged outputs live (each on its own worktree / branch), under
`outputs/` of the given root:

- chat   wsbench_chat, claim map v2.7     wsbench_chat/claude-opus-5-5/<arm>/cells.jsonl
- story  story (plan 0007), claim map v2.7 story/claude-opus-5-5/<arm>/cells.jsonl
- um     user-modelling ladder (0006/7)  sibling/um[/<domain>]/{label/units_labeled,commit,mentions}
- am     agentic misalignment (0008)  agentic_misalignment/claude-opus-5-5/<arm>/{cells,mentions}

Stdlib only; read-only on every input. Selection is seeded and stratified by judge outcome so
each tab shows a mix of hallucinated, correct and other verdicts. Writes one self-contained
HTML file (data embedded) — open it directly or serve the dir with `python -m http.server`.

    uv run --no-sync python scripts/viewer/build_readout_viewer.py
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

WS = Path("/workspace")
JUDGE = "claude-opus-5-5"
HALLUC_LEAVES = {"false_claim", "detail", "substitution_category"}
CLAIMMAP_ARMS = ["s3d-rl600-all11", "mlnla-rl600-all11", "nla-rl-L42", "jlens"]
# per-arm quota for the two claim-map evals (sums to 10)
CLAIMMAP_QUOTA = {"s3d-rl600-all11": 4, "mlnla-rl600-all11": 2, "nla-rl-L42": 2, "jlens": 2}
AM_ARMS = ["s3d-rl600", "nla-rl-L42", "jlens"]
# (verdict wanted, arm preferred) for the AM tab
AM_PLAN = [
    ("hallucination", "s3d-rl600"),
    ("hallucination", "nla-rl-L42"),
    ("hallucination", "jlens"),
    ("correct", "s3d-rl600"),
    ("correct", "jlens"),
    ("unresolved", "s3d-rl600"),
    ("unresolved", "nla-rl-L42"),
    ("thinky", "s3d-rl600"),
    ("probe_candidate", "s3d-rl600"),
    ("junk", "nla-rl-L42"),
]
# (domain, label wanted, method preferred) for the UM tab; domain "country" lives at um/label
UM_PLAN = [
    ("country", "hallucination", "s3d"),
    ("country", "probe-faithful (same layer)", "s3d"),
    ("country", "probe-faithful (nearby layer)", "mlnla-rl600"),
    ("country", "not committed", "s3d"),
    ("capital_city", "hallucination", "mlnla-rl600"),
    ("capital_city", "cross-method allowed (adjudicable)", "s3d"),
    ("occupation", "hallucination", "s3d"),
    ("occupation", "not committed", "nla-rl-L42"),
    ("holiday", "probe-faithful (same layer)", "s3d"),
    ("holiday", "cross-method allowed (adjudicable)", "mlnla-rl600"),
]
RESERVOIR = 300


def jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_items(path: Path) -> dict[str, dict[str, Any]]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    items = doc["items"] if isinstance(doc, dict) else doc
    return {it["id"]: it for it in items}


class Reservoir:
    def __init__(self, cap: int, rng: random.Random) -> None:
        self.cap, self.rng, self.n, self.rows = cap, rng, 0, []  # type: ignore[var-annotated]

    def add(self, row: Any) -> None:
        self.n += 1
        if len(self.rows) < self.cap:
            self.rows.append(row)
        else:
            j = self.rng.randrange(self.n)
            if j < self.cap:
                self.rows[j] = row


def site_char(item: dict[str, Any], pos: int) -> tuple[int | None, str | None, str | None]:
    for s in item.get("sites", []):
        if s["pos"] == pos:
            return s.get("char"), s.get("token"), s.get("kind")
    return None, None, None


def readouts_from(
    samples: list[str] | None, segments: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Readout texts plus their judged segments. Text comes from `samples` when present (prose
    arms); otherwise it is rebuilt from the segments' own offsets (AM cells carry no samples)."""
    by_r: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for s in segments:
        by_r[int(s["readout"])].append(s)
    n = max([len(samples or [])] + [r + 1 for r in by_r])
    out = []
    for r in range(n):
        segs = sorted(by_r.get(r, []), key=lambda s: s["start"])
        if samples and r < len(samples):
            text = samples[r]
        else:
            buf = [" "] * max([s["end"] for s in segs] + [0])
            for s in segs:
                buf[s["start"] : s["end"]] = list(s["text"])[: s["end"] - s["start"]]
            text = "".join(buf)
        out.append(
            {
                "text": text,
                "segs": [
                    {
                        "s": s["start"],
                        "e": s["end"],
                        "leaf": s["leaf"],
                        "why": s.get("why"),
                        "reason": s.get("reason"),
                        "source": s.get("source"),
                        "contradicted_by": s.get("contradicted_by"),
                        "supported_by": s.get("supported_by"),
                    }
                    for s in segs
                ],
            }
        )
    return out


# ------------------------------------------------------------------ chat / story (claim map)


def claimmap_examples(
    root: Path, items: dict[str, dict[str, Any]], rng: random.Random
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    used_items: set[str] = set()
    for arm in CLAIMMAP_ARMS:
        path = root / arm / "cells.jsonl"
        if not path.exists():
            print(f"  skip {path} (missing)")
            continue
        pools = {True: Reservoir(RESERVOIR, rng), False: Reservoir(RESERVOIR, rng)}
        for c in jsonl(path):
            m = c.get("map")
            if not m or not m.get("segments") or any(s["leaf"] == "failed" for s in m["segments"]):
                continue
            pools[any(s["leaf"] in HALLUC_LEAVES for s in m["segments"])].add(c)
        quota = CLAIMMAP_QUOTA[arm]
        want = [True, False] * quota
        for wanted in want[:quota]:
            # fall back to the other pool when one is empty (NLA-RL story: ~every cell hallucinates)
            flag = wanted if pools[wanted].rows else not wanted
            taken = {e["key"] for e in out}
            rows = [c for c in pools[flag].rows if c["key"] not in taken]
            rows = [c for c in rows if c["id"] not in used_items] or rows
            if not rows:
                continue
            c = rng.choice(rows)
            used_items.add(c["id"])
            item = items[c["id"]]
            ch, tok, kind = site_char(item, c["pos"])
            out.append(
                {
                    "arm": arm,
                    "key": c["key"],
                    "id": c["id"],
                    "layer": c["layer"],
                    "pos": c["pos"],
                    "site_token": tok,
                    "site_kind": kind or c.get("site_kind"),
                    "prompt": item["prompt"],
                    "response": item["response"],
                    "marker": ch,
                    "jlens_tokens": c.get("tokens"),
                    "readouts": readouts_from(c["samples"], c["map"]["segments"]),
                    "rationale": c["map"].get("rationale"),
                    "has_halluc": flag,
                }
            )
    return out


# ------------------------------------------------------------------ agentic misalignment


def am_examples(
    root: Path, items: dict[str, dict[str, Any]], rng: random.Random, readouts_root: Path
) -> list[dict[str, Any]]:
    cells: dict[str, dict[str, dict[str, Any]]] = {}
    mentions: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for arm in AM_ARMS:
        res = Reservoir(RESERVOIR, rng)
        for c in jsonl(root / arm / "cells.jsonl"):
            m = c.get("map")
            if c.get("n_mentions") and m and m.get("status") == "ok" and m.get("segments"):
                res.add(c)
        cells[arm] = {c["key"]: c for c in res.rows}
        mentions[arm] = defaultdict(list)
        for mn in jsonl(root / arm / "mentions.jsonl"):
            if mn["key"] in cells[arm]:
                mentions[arm][mn["key"]].append(mn)
        print(f"  am {arm}: {res.n} mapped cells with mentions, kept {len(res.rows)}")

    picks: list[tuple[str, str, str]] = []
    used: set[tuple[str, str]] = set()
    used_items: set[str] = set()
    for verdict, pref in AM_PLAN:
        for arm in [pref] + [a for a in AM_ARMS if a != pref]:
            cand = [
                k
                for k, ms in mentions[arm].items()
                if (arm, k) not in used and any(x["verdict"] == verdict for x in ms)
            ]
            fresh = [k for k in cand if cells[arm][k]["id"] not in used_items]
            if cand:
                k = rng.choice(fresh or cand)
                break
        else:
            print(f"  am: no cell with verdict {verdict}")
            continue
        used.add((arm, k))
        used_items.add(cells[arm][k]["id"])
        picks.append((verdict, arm, k))

    # real readout text (newlines intact) for the prose arms; J-lens readouts are the summaries
    # the map itself segmented, so those are rebuilt from the segments
    raw: dict[tuple[str, str], list[str]] = {}
    for arm in {a for _, a, _ in picks if a != "jlens"}:
        want = {k for _, a, k in picks if a == arm}
        path = readouts_root / arm / "agentic_misalignment.jsonl"
        if path.exists():
            for r in jsonl(path):
                k = f"{r['id']}__L{r['layer']:03d}__p{r['pos']}"
                if k in want:
                    raw[(arm, k)] = r["samples"]

    out: list[dict[str, Any]] = []
    for verdict, arm, k in picks:
        c = cells[arm][k]
        item = items[c["id"]]
        samples = raw.get((arm, k))
        if samples and not all(
            int(sg["readout"]) < len(samples)
            and samples[int(sg["readout"])][sg["start"] : sg["end"]] == sg["text"]
            for sg in c["map"]["segments"]
        ):
            print(f"  am {arm} {k}: raw readout disagrees with segment offsets; rebuilding")
            samples = None
        ch, tok, _ = site_char(item, c["pos"])
        render = item.get("prompt_render", "")
        lo = max(0, (ch or 0) - 700)
        out.append(
            {
                "arm": arm,
                "key": k,
                "id": c["id"],
                "layer": c["layer"],
                "pos": c["pos"],
                "site_token": tok,
                "category": item.get("category"),
                "item_arm": item.get("arm"),
                "persona": item.get("persona"),
                "system": item.get("system"),
                "user": item.get("text"),
                "rollout": item.get("prefill") or item.get("response"),
                "window": render[lo : ch or 0],
                "window_cut": lo > 0,
                "readouts": readouts_from(samples, c["map"]["segments"]),
                "rationale": c["map"].get("rationale"),
                "wanted": verdict,
                "mentions": [
                    {
                        f: x.get(f)
                        for f in (
                            "readout",
                            "start",
                            "end",
                            "quote",
                            "kind",
                            "secondary_kind",
                            "actor",
                            "actor_name",
                            "stance",
                            "headline",
                            "verdict",
                            "leaf",
                            "why",
                            "tier",
                            "matched_kind",
                            "set_up",
                        )
                    }
                    for x in sorted(mentions[arm][k], key=lambda x: (x["readout"], x["start"]))
                ],
            }
        )
    return out


# ------------------------------------------------------------------ user modelling ladder


def um_examples(
    root: Path, items: dict[str, dict[str, Any]], rng: random.Random
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cache: dict[str, tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]] = {}
    used_items: set[str] = set()
    for domain, label, pref in UM_PLAN:
        d = root / ("um" if domain == "country" else f"um/{domain}")
        if domain not in cache:
            units = list(jsonl(d / "label" / "units_labeled.jsonl"))
            commit = {r["uid"]: r for r in jsonl(d / "commit.jsonl")}
            ments = {r["uid"]: r for r in jsonl(d / "mentions.jsonl")}
            cache[domain] = (units, commit, ments)
        units, commit, ments = cache[domain]
        pool = [u for u in units if u["label"] == label and u["uid"] in ments]
        pool_pref = [u for u in pool if u["method"] == pref] or pool
        fresh = [u for u in pool_pref if u["item"] not in used_items]
        if not pool_pref:
            print(f"  um: no {domain} unit labelled {label}")
            continue
        u = rng.choice(fresh or pool_pref)
        used_items.add(u["item"])
        item = items.get(u["item"], {})
        msgs = item.get("messages") or []
        cm = commit.get(u["uid"], {})
        judged = {s["si"]: s for s in cm.get("samples", [])}
        samples = []
        for s in ments[u["uid"]]["samples"]:
            j = judged.get(s["si"], {})
            samples.append(
                {
                    "si": s["si"],
                    "text": s["text"],
                    "hits": [h for h in s["hits"] if h["concept"] == u["concept"]],
                    "committed": j.get("committed"),
                    "form": j.get("form"),
                    "hedged": j.get("hedged"),
                    "why": j.get("why"),
                    "quote": j.get("quote"),
                }
            )
        out.append(
            {
                "domain": domain,
                "method": u["method"],
                "uid": u["uid"],
                "id": u["item"],
                "layer": u["layer"],
                "pos": u["pos"],
                "gold": u["gold"],
                "concept": u["concept"],
                "label": u["label"],
                "committed": u["committed"],
                "via_summary": u.get("via_summary"),
                "probe": u.get("probe"),
                "cross": u.get("cross"),
                "cross_any_layer": u.get("cross_any_layer"),
                "system": next((m["content"] for m in msgs if m["role"] == "system"), None),
                "user": next((m["content"] for m in msgs if m["role"] == "user"), item.get("cue")),
                "response": item.get("response"),
                "samples": samples,
            }
        )
    return out


# ------------------------------------------------------------------ html


def build_html(data: dict[str, Any]) -> str:
    template = (Path(__file__).parent / "readout_viewer.html").read_text(encoding="utf-8")
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return template.replace("/*__DATA__*/null", blob)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--chat-root", type=Path, default=WS / "hb-taxonomy-v2")
    ap.add_argument("--story-root", type=Path, default=WS / "hb-story")
    ap.add_argument("--um-root", type=Path, default=WS / "hb-um-ladder")
    ap.add_argument("--am-root", type=Path, default=WS / "hb-am-halluc")
    ap.add_argument("--readouts-root", type=Path, default=WS / "wsbench-paper-data/readouts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("outputs/readout_viewer/index.html"))
    a = ap.parse_args()

    data: dict[str, Any] = {"seed": a.seed, "sources": {}}
    print("chat")
    data["chat"] = claimmap_examples(
        a.chat_root / "outputs/wsbench_chat" / JUDGE,
        load_items(a.chat_root / "data/wsbench_chat/items.json"),
        random.Random(f"chat-{a.seed}"),
    )
    print("story")
    data["story"] = claimmap_examples(
        a.story_root / "outputs/story" / JUDGE,
        load_items(a.story_root / "data/story/items.json"),
        random.Random(f"story-{a.seed}"),
    )
    print("um")
    data["um"] = um_examples(
        a.um_root / "outputs/sibling",
        load_items(a.um_root / "data/sibling_probe/items.json"),
        random.Random(f"um-{a.seed}"),
    )
    print("am")
    data["am"] = am_examples(
        a.am_root / "outputs/agentic_misalignment" / JUDGE,
        load_items(a.am_root / "data/agentic_misalignment/items.json"),
        random.Random(f"am-{a.seed}"),
        a.readouts_root,
    )
    for k, root in [
        ("chat", a.chat_root),
        ("story", a.story_root),
        ("um", a.um_root),
        ("am", a.am_root),
    ]:
        data["sources"][k] = str(root)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(build_html(data), encoding="utf-8")
    print({k: len(data[k]) for k in ("chat", "story", "um", "am")}, "->", a.out)


if __name__ == "__main__":
    main()
