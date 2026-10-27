# ruff: noqa: E501
"""On-policy Qwen3.6-27B rollouts for the strongreject eval (plan 0016 §Ground truth), on Modal.

The story-imprinting "pure vLLM recipe" (global-workspace scripts/story_imprinting/sample_modal.py):
vLLM 0.25.1 on the cuda 13 image, the Qwen3.6-27B snapshot cached on the ``jlens-hf-cache``
volume. Chat template with ``enable_thinking=False``, the jailbreak prompt as the only user
message, n=10, temperature 1.0, top_p 0.95, max_tokens 512, seed 0. One H200 container: 2,800
completions are minutes of decoding and the 27B load dominates (more containers only add loads).

    MODAL_PROFILE=safety-sahan-2 uvx --python 3.12 modal@latest run scripts/strongreject/sample_modal.py \\
        [--limit 3] [--out outputs/strongreject/rollouts.jsonl]

Writes ``{id, render, samples: [10], finish: [10], n_tokens: [10]}`` per prompt (not committed:
rollouts may contain harmful text).
"""

import json
from pathlib import Path

import modal

SNAPSHOT = "/hf/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
_HERE = Path(__file__).resolve()
REPO = _HERE.parents[2] if len(_HERE.parents) > 2 else Path("/root")  # in-container: unused
N, TEMPERATURE, TOP_P, MAX_TOKENS, SEED = 10, 1.0, 0.95, 512, 0

app = modal.App("hb-strongreject-sample")
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu24.04", add_python="3.12")
    .uv_pip_install("vllm==0.25.1", "hf_transfer", "huggingface_hub")
    .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})
)
hf_cache = modal.Volume.from_name("jlens-hf-cache")


@app.function(
    image=image, gpu="H200", volumes={"/hf": hf_cache}, timeout=3600, memory=131072, cpu=8
)
def sample(prompts: list[dict]) -> list[dict]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=SNAPSHOT, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.9, seed=SEED
    )
    tok = llm.get_tokenizer()
    renders = [
        tok.apply_chat_template(
            [{"role": "user", "content": p["jailbreak_prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for p in prompts
    ]
    sp = SamplingParams(n=N, temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS, seed=SEED)
    outs = llm.generate(renders, sp)
    res = []
    for p, r, o in zip(prompts, renders, outs, strict=True):
        res.append(
            {
                "id": p["id"],
                "render": r,
                "samples": [c.text for c in o.outputs],
                "finish": [c.finish_reason for c in o.outputs],
                "n_tokens": [len(c.token_ids) for c in o.outputs],
            }
        )
    return res


@app.local_entrypoint()
def main(limit: int = 0, out: str = "outputs/strongreject/rollouts.jsonl") -> None:
    prompts = json.loads((REPO / "data/strongreject/prompts.json").read_text())["prompts"]
    if limit:
        prompts = prompts[:limit]
    res = sample.remote(prompts)
    dst = REPO / out
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as fh:
        for r in res:
            r["sampling"] = {
                "n": N,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_tokens": MAX_TOKENS,
                "seed": SEED,
                "snapshot": SNAPSHOT,
            }
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {dst}: {len(res)} prompts x {N}")
