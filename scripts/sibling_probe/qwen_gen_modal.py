# ruff: noqa: E501
"""Greedy Qwen3.6-27B chat completions on Modal (plan 0003 §Phase B: implication check + responses).

Input JSONL rows ``{"key": str, "messages": [{role, content}, ...], "max_new": int}``; output
rows ``{"key", "text", "n_tokens", "finish"}`` appended to ``--out`` (idempotent on ``key``:
rows already in ``--out`` are skipped). Every prompt is rendered with the model's chat template,
``add_generation_prompt=True, enable_thinking=False`` (the ``chat_context`` render of every bank)
and decoded greedily (temperature 0) to ``max_new`` tokens.

One GPU (B300, then B200, then H200; ``QWEN_GPU`` overrides, e.g. ``QWEN_GPU=H200``), vLLM on a
CUDA 13 image (the ``scripts/wrong_bridge/qwen_modal.py`` recipe of the global-workspace repo),
weights from the ``iolens-hf-cache`` volume (``HF_HOME=/hf``) as the gw readers do.

    MODAL_PROFILE=safety-sahan-2 uvx --python 3.12 modal@latest run --env camila-blank \\
        scripts/sibling_probe/qwen_gen_modal.py \\
        --inp outputs/sibling/gen/implication_in.jsonl --out outputs/sibling/gen/implication_out.jsonl
"""

import json
import os
from pathlib import Path

import modal

MODEL_ID = "Qwen/Qwen3.6-27B"
HF_PATH = "/hf"
GPU = [g for g in os.environ.get("QWEN_GPU", "B300,B200,H200").split(",") if g]

app = modal.App("sibling-qwen-gen")
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu24.04", add_python="3.12")
    .uv_pip_install("vllm==0.25.1", "hf_transfer", "huggingface_hub")
    .env({"HF_HOME": HF_PATH, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
hf_cache = modal.Volume.from_name("iolens-hf-cache", create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface"), modal.Secret.from_name("iolens-hf")]


@app.function(
    image=image, gpu=GPU, volumes={HF_PATH: hf_cache}, secrets=secrets, timeout=60 * 60 * 3
)
def run(rows: list[dict]) -> list[dict]:
    import huggingface_hub.constants as hf_constants
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    # resolve the snapshot from the shared cache first (HF API rate limits); online if missing
    os.environ["HF_HUB_OFFLINE"] = "1"
    hf_constants.HF_HUB_OFFLINE = True
    try:
        llm = LLM(
            model=MODEL_ID,
            dtype="bfloat16",
            max_model_len=4096,
            gpu_memory_utilization=0.90,
            enable_prefix_caching=True,
        )
    except Exception as e:  # not cached: go online once
        print(f"[qwen] offline load failed ({type(e).__name__}: {e}); online", flush=True)
        os.environ["HF_HUB_OFFLINE"] = "0"
        hf_constants.HF_HUB_OFFLINE = False
        llm = LLM(
            model=MODEL_ID,
            dtype="bfloat16",
            max_model_len=4096,
            gpu_memory_utilization=0.90,
            enable_prefix_caching=True,
        )
        hf_cache.commit()
    tok = llm.get_tokenizer()
    prompts, params = [], []
    for r in rows:
        ids = tok.apply_chat_template(
            r["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
        prompts.append(TokensPrompt(prompt_token_ids=ids))
        params.append(SamplingParams(temperature=0.0, max_tokens=int(r.get("max_new", 256))))
    outs = llm.generate(prompts, params, use_tqdm=False)
    return [
        {
            "key": r["key"],
            "text": o.outputs[0].text,
            "n_tokens": len(o.outputs[0].token_ids),
            "finish": o.outputs[0].finish_reason,
        }
        for r, o in zip(rows, outs, strict=True)
    ]


@app.local_entrypoint()
def main(inp: str, out: str, limit: int = 0) -> None:
    rows = [json.loads(x) for x in Path(inp).read_text().splitlines() if x.strip()]
    outp = Path(out)
    done = set()
    if outp.exists():
        done = {json.loads(x)["key"] for x in outp.read_text().splitlines() if x.strip()}
    todo = [r for r in rows if r["key"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"[qwen] {len(rows)} rows, {len(done)} done, {len(todo)} to run (gpu {GPU})", flush=True)
    if not todo:
        return
    res = run.remote(todo)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("a", encoding="utf-8") as fh:
        for r in res:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[qwen] +{len(res)} rows -> {outp}", flush=True)
