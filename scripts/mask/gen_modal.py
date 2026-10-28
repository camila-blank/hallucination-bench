# ruff: noqa  (Modal launcher)
"""Stage G of plan 0017: on-policy Qwen3.6-27B rollouts for the MASK items, Modal vLLM.

Recipe copied from global-workspace ``scripts/ar_probe/deception/modal_gen.py`` (vllm 0.25.1,
CUDA 13 devel base, weights in the ``jlens-hf-cache`` volume, one H200 per shard), with one
change: the container also returns ``prompt_token_ids`` and each sample's ``token_ids``, so the
capture can teacher-force the exact generated tokens (no re-tokenisation drift). Loaded by
snapshot path, offline, with ``cpu=8`` (the first launches on 2026-10-27, on the default CPU
share, died in torch.compile or hung in engine start-up).

A request line: ``{id, messages, n, temperature, top_p, max_tokens}``. The prompt is rendered
in the container with ``apply_chat_template(add_generation_prompt=True, enable_thinking=False)``.
Seeds are crc32 of the request id, so a re-run reproduces its rollouts.

    MODAL_RESOURCE_ENV=camila-blank uvx --python 3.12 modal@latest run scripts/mask/gen_modal.py \\
        --requests outputs/mask/gen/requests.jsonl --out outputs/mask/gen/rollouts.jsonl --n-shards 8
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Any

import modal

MODEL_ID = "Qwen/Qwen3.6-27B"
# the cached snapshot on jlens-hf-cache (loaded by path, offline)
SNAPSHOT = "/hf/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
HF_PATH = "/hf"
MAX_MODEL_LEN = 8192

app = modal.App("hb-mask-gen")
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu24.04", add_python="3.12")
    .uv_pip_install("vllm==0.25.1", "huggingface_hub")
    # 2026-10-27: concurrent compile workers raced writing triton .pyc files at runtime ("bad
    # marshal data" on import); compile them at build time and never write bytecode at runtime
    .run_commands("python -m compileall -q /usr/local/lib/python3.12/site-packages/triton")
    .env(
        {
            "HF_HOME": HF_PATH,
            "HF_HUB_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
)
hf_cache = modal.Volume.from_name("jlens-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="H200",
    volumes={HF_PATH: hf_cache},
    timeout=3 * 60 * 60,
    memory=65536,
    cpu=8,  # the default fraction of a core starves engine start-up (2026-10-27: hung)
)
def shard(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    texts, prompt_ids = [], []
    for r in requests:
        text = str(
            tok.apply_chat_template(
                r["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        )
        texts.append(text)
        prompt_ids.append(tok(text, add_special_tokens=False)["input_ids"])
    params = [
        SamplingParams(
            n=int(r["n"]),
            temperature=float(r["temperature"]),
            top_p=float(r.get("top_p", 1.0)),
            max_tokens=int(r["max_tokens"]),
            seed=zlib.crc32(str(r["id"]).encode()) & 0x7FFFFFFF,
        )
        for r in requests
    ]
    print(
        f"[gen] {len(requests)} requests, {sum(int(r['n']) for r in requests)} rollouts", flush=True
    )
    llm = LLM(
        model=SNAPSHOT,
        dtype="bfloat16",
        gpu_memory_utilization=0.92,
        max_model_len=MAX_MODEL_LEN,
        tensor_parallel_size=1,
        enable_prefix_caching=True,
    )
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompt_ids], params)
    results = []
    for r, text, pids, o in zip(requests, texts, prompt_ids, outs, strict=True):
        results.append(
            {
                "id": r["id"],
                "prompt": text,
                "prompt_token_ids": list(pids),
                "completions": [c.text for c in o.outputs],
                "token_ids": [list(c.token_ids) for c in o.outputs],
                "finish": [str(c.finish_reason) for c in o.outputs],
                "sampling": {k: r[k] for k in ("n", "temperature", "top_p", "max_tokens")},
            }
        )
    return results


@app.local_entrypoint()
def main(requests: str, out: str, n_shards: int = 8) -> None:
    reqs = [json.loads(l) for l in Path(requests).read_text().split("\n") if l.strip()]
    assert reqs, "no requests"
    ids = [r["id"] for r in reqs]
    assert len(set(ids)) == len(ids), "duplicate request ids"
    order = sorted(reqs, key=lambda r: -int(r["n"]) * int(r["max_tokens"]))
    n_shards = max(1, min(n_shards, len(reqs)))
    chunks: list[list[dict[str, Any]]] = [[] for _ in range(n_shards)]
    load = [0] * n_shards
    for r in order:
        i = load.index(min(load))
        chunks[i].append(r)
        load[i] += int(r["n"]) * int(r["max_tokens"])
    print(f"[gen] {len(reqs)} requests over {n_shards} H200s", flush=True)
    got: dict[str, dict[str, Any]] = {}
    for res in shard.map(chunks):
        for rec in res:
            got[rec["id"]] = rec
        print(f"[gen] shard done ({len(got)}/{len(reqs)})", flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(out).with_suffix(".tmp")
    with tmp.open("w") as f:
        for i in ids:
            f.write(json.dumps(got[i], ensure_ascii=False) + "\n")
    tmp.rename(out)
    print(f"[gen] wrote {out}", flush=True)
