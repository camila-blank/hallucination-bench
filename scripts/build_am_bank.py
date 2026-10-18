"""Build ``data/agentic_misalignment/`` (plan 0008) from WorkspaceBench's frozen AM items.

Reads the wsbench-pub ``evals/agentic_misalignment/items.json`` and the layer-20
``(id, pos, token)`` columns of the ``s3d-rl600`` readouts (streamed line by line: the readout
files are 50-180 MB). Per item it writes the bank fields, the item's ``persona``, the Qwen chat
render of the prompt (``prompt_render``) and one site per prompt position, ``{pos, token,
char[, split_char]}``, where ``char`` is the end offset of token ``pos`` in ``prompt_render``.

The decoded token strings concatenate to ``prompt_render`` except where a multi-byte character
is split across tokens (each piece decodes to U+FFFD). ``difflib`` aligns the two; every
mismatch must be a ``replace`` of a U+FFFD-only run by FFFD-free text. A token whose text is part
of such a run is marked ``split_char`` and its ``char`` moves to the end of the run when it ends
inside it, so the marker never splits a character.

Checks (fail loudly): 32 items; contiguous positions ending ``<|im_start|>``, ``assistant``; the
alignment; ``prompt_render[:char]`` ends with the token for every non-split site; the J-lens
layer-20 ``(id, pos, token)`` columns and the NLA positions equal s3d's; the persona counts.

    uv run --no-sync python scripts/build_am_bank.py [--items PATH] [--readouts-root DIR]
"""

import argparse
import difflib
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hallucination_bench.readouts import decode_token  # noqa: E402

OUT = REPO / "data" / "agentic_misalignment"
WSBENCH_PUB = Path("/workspace/root/wsbench-pub")
SRC_REL = "evals/agentic_misalignment/items.json"
DEFAULT_ITEMS = WSBENCH_PUB / SRC_REL
DEFAULT_READOUTS = Path("/workspace/wsbench-paper-data/readouts")
READOUTS_FILE = "agentic_misalignment.jsonl"
N_ITEMS = 32
N_SITES = 50_452
BANK_LAYER = 20
FFFD = "�"
ITEM_KEYS = ("id", "category", "arm", "group", "descriptor", "misalignment_rate")
# the name after "You are" in the system prompt, checked by hand for all 32 items (plan 0008)
PERSONA_COUNTS = {
    "Alex": 19,
    "Atlas": 2,
    "Argus": 1,
    "Devin": 1,
    "Hypatia": 1,
    "Kepler": 1,
    "Meridian": 1,
    None: 6,
}
_PERSONA = re.compile(r"^You are (?!(?:an?|the)\b)([A-Z][A-Za-z0-9-]*)")
_LAYER = re.compile(r'"layer":\s*(\d+)')


def render_prompt(system: str, text: str) -> str:
    head = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
    return head + f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant"


def persona(system: str) -> str | None:
    m = _PERSONA.match(system or "")
    return m.group(1) if m else None


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(d: Path) -> str:
    out = subprocess.run(
        ["git", "-C", str(d), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def git_dirty(d: Path, rel: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(d), "status", "--porcelain", "--", rel],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def stream_columns(path: Path, layer: int | None) -> dict[str, list[tuple[int, str | None]]]:
    """``{id: [(pos, token), ...]}`` of one layer (``None``: every row), streamed."""
    out: dict[str, list[tuple[int, str | None]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            if layer is not None:
                m = _LAYER.search(line, 0, 400)
                if m is None or int(m.group(1)) != layer:
                    continue
            row = json.loads(line)
            if layer is not None and row["layer"] != layer:
                continue
            out[row["id"]].append((int(row["pos"]), row.get("token")))
    for v in out.values():
        v.sort()
    return dict(out)


def align(tokens: list[str], render: str, iid: str) -> list[dict[str, Any]]:
    """Sites ``{pos, token, char[, split_char]}`` from the token stream and the render."""
    concat = "".join(tokens)
    sm = difflib.SequenceMatcher(None, concat, render, autojunk=False)
    ops = sm.get_opcodes()
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            continue
        if tag != "replace":
            raise SystemExit(
                f"{iid}: alignment has a {tag!r} block {concat[i1:i2]!r}->{render[j1:j2]!r}"
            )
        if set(concat[i1:i2]) != {FFFD} or FFFD in render[j1:j2]:
            raise SystemExit(
                f"{iid}: non-U+FFFD mismatch {concat[i1:i2]!r} -> {render[j1:j2]!r} at {i1}"
            )
    sites: list[dict[str, Any]] = []
    start = 0
    k = 0  # opcode cursor (token ends are monotone)
    for pos, tok in enumerate(tokens):
        end = start + len(tok)
        while k < len(ops) - 1 and ops[k][2] < end:
            k += 1
        tag, i1, i2, j1, j2 = ops[k]
        if tag == "equal":
            char = j1 + (end - i1)
        else:  # the token ends inside (or at the end of) a U+FFFD run
            char = j2
        split = any(
            t == "replace" and a < end and start < b for t, a, b, _c, _d in ops
        )  # the token carries a piece of a split character
        if end == start:  # an empty token: maps like its neighbour
            split = False
        site: dict[str, Any] = {"pos": pos, "token": tok, "char": char}
        if split:
            site["split_char"] = True
        elif not render[:char].endswith(tok):
            raise SystemExit(f"{iid}: pos {pos} char {char} does not end with {tok!r}")
        sites.append(site)
        start = end
    if sites[-1]["char"] != len(render):
        raise SystemExit(f"{iid}: last site char {sites[-1]['char']} != len(render) {len(render)}")
    return sites


def build(items_path: Path, readouts_root: Path, *, allow_dirty: bool = False) -> dict[str, Any]:
    dirty = git_dirty(WSBENCH_PUB, SRC_REL) if items_path == DEFAULT_ITEMS else ""
    if dirty and not allow_dirty:
        raise SystemExit(f"source items have uncommitted changes:\n{dirty}")
    src = json.loads(items_path.read_text(encoding="utf-8"))
    src_items = src["items"]
    if len(src_items) != N_ITEMS:
        raise SystemExit(f"expected {N_ITEMS} items, got {len(src_items)}")
    paths = {a: readouts_root / a / READOUTS_FILE for a in ("s3d-rl600", "jlens", "nla-rl-L42")}
    print("streaming s3d-rl600 (layer 20) ...", flush=True)
    s3d = stream_columns(paths["s3d-rl600"], BANK_LAYER)
    print("streaming jlens (layer 20) ...", flush=True)
    jl = stream_columns(paths["jlens"], BANK_LAYER)
    print("streaming nla-rl-L42 (positions) ...", flush=True)
    nla = stream_columns(paths["nla-rl-L42"], None)

    ids = [it["id"] for it in src_items]
    if set(s3d) != set(ids):
        raise SystemExit(f"s3d ids differ from the bank: {sorted(set(s3d) ^ set(ids))}")
    items: list[dict[str, Any]] = []
    personas: Counter = Counter()
    for it in src_items:
        iid = it["id"]
        col = s3d[iid]
        if [p for p, _ in col] != list(range(len(col))):
            raise SystemExit(f"{iid}: s3d positions are not contiguous from 0")
        toks = [t for _, t in col]
        if any(t is None for t in toks):
            raise SystemExit(f"{iid}: s3d rows without a token")
        if toks[-2:] != ["<|im_start|>", "assistant"]:
            raise SystemExit(f"{iid}: positions end with {toks[-2:]!r}, not <|im_start|> assistant")
        jcol = jl.get(iid, [])
        if [p for p, _ in jcol] != [p for p, _ in col]:
            raise SystemExit(f"{iid}: jlens layer-20 positions differ from s3d")
        for (p, a), (_, b) in zip(col, jcol, strict=True):
            if b is None or (b != a and decode_token(b) != a):
                raise SystemExit(f"{iid}: jlens token at pos {p} {b!r} != s3d {a!r}")
        npos = sorted({p for p, _ in nla.get(iid, [])})
        if npos != [p for p, _ in col]:
            raise SystemExit(f"{iid}: nla-rl-L42 positions differ from s3d")
        render = render_prompt(it["system"], it["text"])
        sites = align(toks, render, iid)
        per = persona(it["system"])
        personas[per] += 1
        items.append(
            {
                **{k: it[k] for k in ITEM_KEYS},
                "persona": per,
                "system": it["system"],
                "text": it["text"],
                "prefill": it["prefill"],
                "prompt_render": render,
                "response": it["prefill"],
                "sites": sites,
            }
        )
    if dict(personas) != PERSONA_COUNTS:
        raise SystemExit(f"persona counts {dict(personas)} != the hand-checked {PERSONA_COUNTS}")
    n_sites = sum(len(i["sites"]) for i in items)
    if n_sites != N_SITES:
        raise SystemExit(f"{n_sites} sites, expected {N_SITES}")
    n_split = sum(bool(s.get("split_char")) for i in items for s in i["sites"])
    commit = git_head(WSBENCH_PUB) if items_path == DEFAULT_ITEMS else None
    meta = {
        "eval": "agentic_misalignment",
        "plan": "plans/0008-agentic-misalignment-claim-map.md",
        "source_repo": "camilablank/wsbench-pub",
        "source_commit": commit or "unknown",
        "source_path": SRC_REL,
        "source_sha256": {
            "items.json": sha256(items_path),
            **{f"{a}/{READOUTS_FILE}": sha256(p) for a, p in paths.items()},
        },
        "site_source": f"s3d-rl600/{READOUTS_FILE} layer {BANK_LAYER} (id, pos, token)",
        "render": "Qwen chat: <|im_start|>system\\n{system}<|im_end|>\\n (if system) + "
        "<|im_start|>user\\n{text}<|im_end|>\\n<|im_start|>assistant",
        "model": "Qwen/Qwen3.6-27B",
        "n_items": len(items),
        "n_sites": n_sites,
        "n_split_char_sites": n_split,
        "persona_counts": {str(k): v for k, v in sorted(personas.items(), key=str)},
    }
    return {"meta": meta, "items": items}


def dumps(bank: dict[str, Any]) -> str:
    """Indented JSON with one site per line (the bank has 50k sites)."""
    sites = {it["id"]: it["sites"] for it in bank["items"]}
    slim = {**bank, "items": [{**it, "sites": f"@@SITES:{it['id']}@@"} for it in bank["items"]]}
    text = json.dumps(slim, ensure_ascii=False, indent=1)
    for iid, ss in sites.items():
        body = ",\n".join("   " + json.dumps(s, ensure_ascii=False) for s in ss)
        text = text.replace(json.dumps(f"@@SITES:{iid}@@"), "[\n" + body + "\n  ]", 1)
    return text + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    ap.add_argument("--readouts-root", type=Path, default=DEFAULT_READOUTS)
    ap.add_argument("--allow-dirty", action="store_true", help="build from uncommitted source")
    args = ap.parse_args()
    bank = build(args.items, args.readouts_root, allow_dirty=args.allow_dirty)
    OUT.mkdir(parents=True, exist_ok=True)
    text = dumps(bank)
    if json.loads(text) != bank:
        raise SystemExit("serialisation round-trip failed")
    (OUT / "items.json").write_text(text, encoding="utf-8")
    m = bank["meta"]
    print(
        f"wrote {OUT}/items.json: {m['n_items']} items, {m['n_sites']} sites, "
        f"{m['n_split_char_sites']} split_char sites, personas {m['persona_counts']}"
    )


if __name__ == "__main__":
    main()
