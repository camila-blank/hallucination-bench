"""Plan 0018 belief gate: sampled Qwen3.6-27B chat completions on Modal (vLLM, one GPU).

Input JSONL rows ``{"key", "messages", "temperature", "top_p", "max_new", "seed", ...}`` (written by
``hbench sad gate-requests``); output rows ``{"key", "text", "n_tokens", "finish"}`` appended to
``--out`` (idempotent on ``key``). Every prompt is rendered with the model's chat template,
``add_generation_prompt=True, enable_thinking=False``, and sampled with the row's own temperature /
top_p / per-request seed, ``enforce_eager=True`` (the compiled path hung after torch.compile
here and failed for hb-mask the same day). The recipe (image, volume, offline-first load) is
``scripts/sibling_probe/qwen_gen_modal.py``'s. One GPU: 4k short generations are minutes; the
fan-out rule does not pay for a second 27B load here.

    MODAL_PROFILE=safety-sahan-2 uvx --python 3.12 modal@latest run --env camila-blank \\
        scripts/sad/qwen_gate_modal.py --inp outputs/sad/gate/requests.jsonl \\
        --out outputs/sad/gate/samples.jsonl
"""

import json
import os
from pathlib import Path

import modal

MODEL_ID = "Qwen/Qwen3.6-27B"
HF_PATH = "/hf"
GPU = [g for g in os.environ.get("QWEN_GPU", "H200,B200,H100").split(",") if g]

app = modal.App("hb-sad-gate")
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu24.04", add_python="3.12")
    .uv_pip_install("vllm==0.25.1", "hf_transfer", "huggingface_hub")
    # PYTHONPYCACHEPREFIX: ignore the image's baked .pyc files (one shard died on a corrupt triton
    # .pyc, "bad marshal data", 2026-10-28)
    .env({"HF_HOME": HF_PATH, "HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONPYCACHEPREFIX": "/tmp/pyc"})
)
hf_cache = modal.Volume.from_name("iolens-hf-cache", create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface"), modal.Secret.from_name("iolens-hf")]


@app.function(
    image=image, gpu=GPU, volumes={HF_PATH: hf_cache}, secrets=secrets, timeout=60 * 60 * 2
)
def run(rows: list[dict]) -> list[dict]:
    import huggingface_hub.constants as hf_constants
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    kw = dict(
        model=MODEL_ID,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,
        enforce_eager=True,  # torch.compile hung / failed in this image (2026-10-27)
    )
    os.environ["HF_HUB_OFFLINE"] = "1"
    hf_constants.HF_HUB_OFFLINE = True
    try:
        llm = LLM(**kw)
    except Exception as e:  # not cached: go online once
        print(f"[gate] offline load failed ({type(e).__name__}: {e}); online", flush=True)
        os.environ["HF_HUB_OFFLINE"] = "0"
        hf_constants.HF_HUB_OFFLINE = False
        llm = LLM(**kw)
        hf_cache.commit()
    tok = llm.get_tokenizer()
    prompts, params = [], []
    for r in rows:
        ids = tok.apply_chat_template(
            r["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
        prompts.append(TokensPrompt(prompt_token_ids=ids))
        params.append(
            SamplingParams(
                temperature=float(r["temperature"]),
                top_p=float(r["top_p"]),
                max_tokens=int(r["max_new"]),
                seed=int(r["seed"]),
            )
        )
    outs = llm.generate(prompts, params, use_tqdm=False)
    return [
        {
            "key": r["key"],
            "text": o.outputs[0].text,
            "n_tokens": len(o.outputs[0].token_ids),
            "finish": o.outputs[0].finish_reason,
            "n_prompt_tokens": len(o.prompt_token_ids or []),
        }
        for r, o in zip(rows, outs, strict=True)
    ]


@app.local_entrypoint()
def main(inp: str, out: str, limit: int = 0, shards: int = 1) -> None:
    rows = [json.loads(x) for x in Path(inp).read_text().splitlines() if x.strip()]
    outp = Path(out)
    done = set()
    if outp.exists():
        done = {json.loads(x)["key"] for x in outp.read_text().splitlines() if x.strip()}
    todo = [r for r in rows if r["key"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"[gate] {len(rows)} rows, {len(done)} done, {len(todo)} to run (gpu {GPU})", flush=True)
    if not todo:
        return
    # fan out: one GPU per shard, whole items per shard (an item's 10 samples stay together)
    n = max(1, min(shards, len(todo)))
    chunks: list[list[dict]] = [[] for _ in range(n)]
    for r in todo:
        chunks[hash(r["key"].rsplit("#s", 1)[0]) % n].append(r)
    outp.parent.mkdir(parents=True, exist_ok=True)
    for res in run.map([c for c in chunks if c], return_exceptions=True):
        if isinstance(res, BaseException):
            print(f"[gate] shard failed: {type(res).__name__}: {res}", flush=True)
            continue
        with outp.open("a", encoding="utf-8") as fh:
            for r in res:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[gate] +{len(res)} rows -> {outp}", flush=True)
