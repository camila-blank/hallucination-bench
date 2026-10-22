"""Build the DeepSeek-V4-Flash agentic_misalignment bank (same schema as
``data/agentic_misalignment/items.json``, plan 0008) for ``HBENCH_AM_ITEMS``.

The V4 copy of WorkspaceBench (``wsbench-pub-v4-opus``) keeps the 32 AM items byte for byte
(``system``, ``text``, ``prefill``, ``descriptor``, ``misalignment_rate``, ``arm``: the pinned
rollouts are still **Qwen3.6-27B's**; the bench never re-ran the scenarios on V4). What changes is
the render the subject read and so the read sites:

- ``prompt_render`` = V4's chat encoding (``encode_messages(thinking_mode="chat")``):
  ``<｜begin▁of▁sentence｜>{system}<｜User｜>{text}<｜Assistant｜></think>`` (no system: BOS +
  ``<｜User｜>...``);
- ``sites`` = every prompt position of that render, from the V4 J-lens readouts' layer-21
  ``(id, pos, token)`` columns (contiguous from 0, ending ``<｜Assistant｜>``, ``</think>``),
  aligned to the render exactly as ``build_am_bank.align`` does (U+FFFD split characters).

Checks (fail loudly): the source items equal the Qwen bank's fields (so the hand-audited intent
overrides still describe the same rollouts); every J-lens layer has the same columns; the
alignment; the persona counts; and, with ``--check`` arms, those readouts' columns.

    uv run --no-sync python scripts/build_am_bank_v4.py --out outputs/v4/am_bank/items.json \
        [--check-readouts <root>/<arm>/agentic_misalignment.jsonl ...]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import build_am_bank as qb  # noqa: E402

from hallucination_bench import paths  # noqa: E402

V4_PUB = paths.WSBENCH_PUB_V4
# V4 special tokens (bars = U+FF5C FULLWIDTH VERTICAL LINE, spaces = U+2581), as escapes
BOS = "<\uff5cbegin\u2581of\u2581sentence\uff5c>"
USER, ASSISTANT, THINK_END = "<\uff5cUser\uff5c>", "<\uff5cAssistant\uff5c>", "</think>"
SITE_LAYER = 21


def render_prompt_v4(system: str | None, text: str) -> str:
    return f"{BOS}{system or ''}{USER}{text}{ASSISTANT}{THINK_END}"


def stream_all(path: Path) -> dict[int, dict[str, list[tuple[int, str | None]]]]:
    """``{layer: {id: [(pos, token)]}}`` of a readouts file (columns only), streamed."""
    out: dict[int, dict[str, list[tuple[int, str | None]]]] = defaultdict(lambda: defaultdict(list))
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            out[int(r["layer"])][r["id"]].append((int(r["pos"]), r.get("token")))
    for by in out.values():
        for v in by.values():
            v.sort()
    return {la: dict(by) for la, by in out.items()}


def build(items_path: Path, jlens_path: Path, checks: list[Path]) -> dict[str, Any]:
    src = json.loads(items_path.read_text(encoding="utf-8"))["items"]
    qwen = {it["id"]: it for it in json.loads((qb.OUT / "items.json").read_text())["items"]}
    if len(src) != qb.N_ITEMS or {it["id"] for it in src} != set(qwen):
        raise SystemExit("V4 source ids differ from the Qwen bank")
    for it in src:
        for k in (*qb.ITEM_KEYS, "system", "text", "prefill"):
            if it.get(k) != qwen[it["id"]].get(k):
                raise SystemExit(f"{it['id']}: field {k} differs from the Qwen bank")
    print(f"streaming {jlens_path} ...", flush=True)
    jl = stream_all(jlens_path)
    ref = jl[SITE_LAYER]
    for la, by in jl.items():
        if by != ref:
            raise SystemExit(f"J-lens layer {la} columns differ from layer {SITE_LAYER}")
    items: list[dict[str, Any]] = []
    personas: dict[str | None, int] = defaultdict(int)
    for it in src:
        iid = it["id"]
        col = ref.get(iid, [])
        if [p for p, _ in col] != list(range(len(col))) or not col:
            raise SystemExit(f"{iid}: positions are not contiguous from 0")
        toks = [t for _, t in col]
        if any(t is None for t in toks):
            raise SystemExit(f"{iid}: rows without a token")
        if toks[0] != BOS or toks[-2:] != [ASSISTANT, THINK_END]:
            raise SystemExit(f"{iid}: positions run {toks[:1]} .. {toks[-2:]}")
        render = render_prompt_v4(it["system"], it["text"])
        sites = qb.align(toks, render, iid)
        per = qb.persona(it["system"])
        personas[per] += 1
        items.append(
            {
                **{k: it[k] for k in qb.ITEM_KEYS},
                "persona": per,
                "system": it["system"],
                "text": it["text"],
                "prefill": it["prefill"],
                "prompt_render": render,
                "response": it["prefill"],
                "sites": sites,
            }
        )
    if dict(personas) != qb.PERSONA_COUNTS:
        raise SystemExit(f"persona counts {dict(personas)} != {qb.PERSONA_COUNTS}")
    cols = {it["id"]: [(s["pos"], s["token"]) for s in it["sites"]] for it in items}
    for p in checks:
        print(f"checking {p} ...", flush=True)
        for la, by in stream_all(p).items():
            if by != cols:
                bad = sorted(i for i in set(by) | set(cols) if by.get(i) != cols.get(i))
                raise SystemExit(f"{p} layer {la}: columns differ from the bank on {bad[:5]}")
    n_sites = sum(len(i["sites"]) for i in items)
    meta = {
        "eval": "agentic_misalignment",
        "plan": "plans/0008-agentic-misalignment-claim-map.md",
        "subject": "DeepSeek-V4-Flash (port: scripts/build_am_bank_v4.py)",
        "source_repo": "camilablank/wsbench-pub (V4 copy wsbench-pub-v4-opus)",
        "source_commit": qb.git_head(items_path.parents[2]),
        "source_path": qb.SRC_REL,
        "source_sha256": {
            "items.json": qb.sha256(items_path),
            "v4-jlens-std-n1000/agentic_misalignment.jsonl": qb.sha256(jlens_path),
        },
        "site_source": f"v4-jlens-std-n1000/agentic_misalignment.jsonl layer {SITE_LAYER} "
        "(id, pos, token); every layer's columns equal",
        "render": "V4 chat (encode_messages thinking_mode=chat): <｜begin▁of▁sentence｜>{system}"
        "<｜User｜>{text}<｜Assistant｜></think>",
        "model": "deepseek-ai/DeepSeek-V4-Flash",
        "rollouts": "prefill / response = the Qwen3.6-27B pinned rollouts (unchanged in the V4 "
        "bench copy); arm / misalignment_rate are Qwen's",
        "n_items": len(items),
        "n_sites": n_sites,
        "n_split_char_sites": sum(bool(s.get("split_char")) for i in items for s in i["sites"]),
        "persona_counts": {str(k): v for k, v in sorted(personas.items(), key=str)},
    }
    return {"meta": meta, "items": items}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--items", type=Path, default=V4_PUB / qb.SRC_REL)
    ap.add_argument(
        "--jlens",
        type=Path,
        default=V4_PUB / "outputs/readouts/v4-jlens-std-n1000/agentic_misalignment.jsonl",
    )
    ap.add_argument("--check-readouts", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    bank = build(args.items, args.jlens, args.check_readouts)
    text = qb.dumps(bank)
    if json.loads(text) != bank:
        raise SystemExit("serialisation round-trip failed")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    m = bank["meta"]
    print(
        f"wrote {args.out}: {m['n_items']} items, {m['n_sites']} sites, "
        f"{m['n_split_char_sites']} split_char sites, personas {m['persona_counts']}"
    )


if __name__ == "__main__":
    main()
