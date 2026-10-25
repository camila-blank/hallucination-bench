"""Regenerate the 3 corrupted V4 chat-bank responses (plan 0014 §Audit) on the padded + canary
V4 vLLM runner, and splice them into the V4 bench copies' ``evals/hallucination`` bank.

The V4 bank (wsbench-pub 8e59016) came from an unpadded ``gen_vllm_modal.py`` call whose first
three requests (``chat-lmsys-0000..0002``) returned garbage. This re-asks exactly those prompts
with the bank's own rollout recipe (``wsbench_v4_content.py prep``: V4 chat render, thinking off,
T 1.0, top-p 1.0, top-k off, <= 512 new tokens, seed 7, EOS not kept) through
``gen_requests_modal.py`` (300 head + tail pads run to max_tokens, Jaws + Paris canary), and
rebuilds their read sites with ``wsbench_v4_content.py build``'s rule (summary-site tokens of the
response, thinned to 8, char offsets from prefix decodes).

    prep   -> requests JSON on the v4-flash volume (then run gen_requests_modal.py on Modal)
    build  -> items.json / capture_rows.json of every --bench copy with the 3 items replaced;
              the previous files are kept once as items.json.corrupt / capture_rows.json.corrupt

Needs the V4 tokenizer (HF cache, HF_HUB_OFFLINE=1) and global-workspace's scripts/v4_flash on
--gw (origin/main: the padded runner).
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

IDS = ("chat-lmsys-0000", "chat-lmsys-0001", "chat-lmsys-0002")
REQ = "runs/hb_v4_fix/chat3_requests.json"
OUT = "runs/hb_v4_fix/chat3_outputs.json"
BENCHES = ("/workspace/root/wsbench-pub-v4-opus", "/workspace/root/wsbench-pub-v4")


def _content(gw: Path) -> Any:
    sys.path.insert(0, str(gw / "scripts" / "v4_flash"))
    import wsbench_v4_content as c  # type: ignore[import-not-found]

    return c


def _bank(bench: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    hp = bench / "evals" / "hallucination"
    return (
        json.loads((hp / "items.json").read_text(encoding="utf-8")),
        json.loads((hp / "capture_rows.json").read_text(encoding="utf-8")),
    )


def prep(a: argparse.Namespace) -> None:
    c = _content(Path(a.gw))
    tok, enc = c.load_v4()
    d, rows = _bank(Path(a.bench[0]))
    by = {it["id"]: it for it in d["items"]}
    row = {r["id"]: r for r in rows}
    reqs = []
    for i in IDS:
        ids = c.chat(tok, enc, by[i]["prompt"])
        # the render must be the one the bank was generated from
        assert ids == row[i]["input_ids"][: row[i]["prompt_len"]], i
        reqs.append({"id": f"hal:{i}", "ids": ids, "n": 1, "temperature": 1.0, "top_p": 1.0,
                     "top_k": -1, "max_tokens": 512, "seed": 7, "keep_eos": False})  # fmt: skip
    p = Path("/tmp/hb_v4_chat3_requests.json")
    p.write_text(json.dumps(reqs))
    subprocess.run(["modal", "volume", "put", "-f", "v4-flash", str(p), REQ], check=True)
    print(f"{len(reqs)} requests -> v4-flash:{REQ}")


def build(a: argparse.Namespace) -> None:
    c = _content(Path(a.gw))
    tok, enc = c.load_v4()
    dec = tok.decode
    p = Path("/tmp/hb_v4_chat3_outputs.json")
    subprocess.run(["modal", "volume", "get", "--force", "v4-flash", OUT, str(p)], check=True)
    out = json.loads(p.read_text())
    new_items: dict[str, dict[str, Any]] = {}
    new_rows: dict[str, dict[str, Any]] = {}
    d0, rows0 = _bank(Path(a.bench[0]))
    by, row = {it["id"]: it for it in d0["items"]}, {r["id"]: r for r in rows0}
    for i in IDS:
        prompt = c.chat(tok, enc, by[i]["prompt"])
        o = out[f"hal:{i}"][0]
        resp, rs = o["ids"], len(prompt)
        sites = []
        for k, t in enumerate(resp):
            s = dec([t])
            if c.is_summary(s):
                sites.append({"pos": rs + k, "token": s, "kind": c.site_kind(s),
                              "char": len(dec(resp[: k + 1]))})  # fmt: skip
        if len(sites) > 8:  # thin evenly to 8 (select_sites)
            idx = sorted({round(j * (len(sites) - 1) / 7) for j in range(8)})
            sites = [sites[j] for j in idx]
        assert sites, f"{i}: no read sites in the regenerated response"
        it = by[i]
        new_items[i] = {**{k: it[k] for k in ("id", "stratum", "source", "prompt")},
                        "response": dec(resp), "response_start": rs, "sites": sites}  # fmt: skip
        new_rows[i] = {**{k: row[i][k] for k in ("id", "source", "source_id")},
                       "prompt_len": rs, "input_ids": prompt + resp,
                       "read_positions": [s["pos"] for s in sites]}  # fmt: skip
        print(f"{i}: finish {o['finish']}, {len(resp)} tokens, {len(sites)} sites: "
              f"{dec(resp)[:100]!r}")  # fmt: skip
    for b in a.bench:
        hp = Path(b) / "evals" / "hallucination"
        d, rows = _bank(Path(b))
        assert [it["id"] for it in d["items"]] == [it["id"] for it in d0["items"]], b
        for f in ("items.json", "capture_rows.json"):
            if not (hp / f"{f}.corrupt").exists():
                shutil.copy2(hp / f, hp / f"{f}.corrupt")
        d["items"] = [new_items.get(it["id"], it) for it in d["items"]]
        d["meta"]["n_sites"] = sum(len(it["sites"]) for it in d["items"])
        d["meta"]["regenerated"] = (
            f"{', '.join(IDS)}: re-asked 2026-10-24 with the same recipe on the padded + canary "
            "V4 vLLM runner (gen_requests_modal.py); the unpadded originals were garbage "
            "(head-of-call corruption). Originals: items.json.corrupt / capture_rows.json.corrupt"
        )
        rows = [new_rows.get(r["id"], r) for r in rows]
        (hp / "items.json").write_text(json.dumps(d, ensure_ascii=False, indent=1))
        (hp / "capture_rows.json").write_text(json.dumps(rows))
        print(f"{hp}: replaced {len(IDS)} items (n_sites {d['meta']['n_sites']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=["prep", "build"])
    ap.add_argument("--gw", default="/workspace/gw-v4-hbfix")
    ap.add_argument("--bench", action="append", default=None, help=f"default {BENCHES}")
    a = ap.parse_args()
    a.bench = a.bench or list(BENCHES)
    {"prep": prep, "build": build}[a.mode](a)
