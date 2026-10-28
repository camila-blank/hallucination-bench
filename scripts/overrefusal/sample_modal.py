"""Plan 0015 Stage 1: on-policy Qwen3.6-27B samples for the OR-Bench candidates (Modal vLLM).

Reads ``data/overrefusal/candidates.jsonl`` and writes ``outputs/overrefusal/samples.jsonl``
(``id, sample, text, finish_reason, n_tokens``; ``N_SAMPLES`` rows per prompt) plus
``samples_meta.json``. Chat template with ``enable_thinking=False``, no system prompt,
``n=10, temperature=1.0, top_p=0.95, max_tokens=512, seed=0``. The vLLM image and snapshot are
the story-imprinting recipe of record (global-workspace
``scripts/story_imprinting/sample_modal.py``): the snapshot is read from the ``jlens-hf-cache``
volume, so no HF token is needed.

    uvx --python 3.12 modal@latest run scripts/overrefusal/sample_modal.py --shards 4
    # --ids orh-0001,ort-0002 : only these candidates (appends to the samples file);
    # --skip-done            : skip candidates already in the samples file
    # --greedy               : one greedy (T=0) sample per prompt -> samples_greedy.jsonl
"""

import json
import time
from pathlib import Path

import modal

SNAPSHOT = "/hf/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
HF_PATH = "/hf"
VLLM = "vllm==0.25.1"
N_SAMPLES = 10
PARAMS = {"n": N_SAMPLES, "temperature": 1.0, "top_p": 0.95, "max_tokens": 512, "seed": 0}
# plan 0015 rev 2: one greedy sample per prompt (never_refused also needs a complying greedy reply)
GREEDY_PARAMS = {"n": 1, "temperature": 0.0, "max_tokens": 512}
MAX_MODEL_LEN = 4096

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2] if len(_HERE.parents) > 2 else Path("/root")

app = modal.App("hb-overrefusal-sample")
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu24.04", add_python="3.12")
    .uv_pip_install(VLLM, "huggingface_hub")
    .env({"HF_HOME": HF_PATH, "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"})
)
hf_cache = modal.Volume.from_name("jlens-hf-cache", create_if_missing=False)


@app.function(image=image, gpu="H200", volumes={HF_PATH: hf_cache}, timeout=3 * 3600, cpu=8)
def sample_shard(rows: list[dict], greedy: bool = False) -> list[dict]:
    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(
        model=SNAPSHOT,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.9,
        seed=0,
    )
    tok = llm.get_tokenizer()
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for r in rows
    ]
    print(f"[sample] loaded in {time.time() - t0:.0f}s; {len(rows)} prompts", flush=True)
    outs = llm.generate(prompts, SamplingParams(**(GREEDY_PARAMS if greedy else PARAMS)))
    res = []
    for r, o in zip(rows, outs, strict=True):
        for j, c in enumerate(o.outputs):
            res.append(
                {
                    "id": r["id"],
                    "sample": j,
                    "text": c.text,
                    "finish_reason": c.finish_reason,
                    "n_tokens": len(c.token_ids),
                }
            )
    print(f"[sample] done in {time.time() - t0:.0f}s", flush=True)
    return res


@app.local_entrypoint()
def main(shards: int = 4, ids: str = "", skip_done: bool = False, greedy: bool = False) -> None:
    cands = [
        json.loads(x)
        for x in (_REPO / "data/overrefusal/candidates.jsonl").read_text().splitlines()
        if x.strip()
    ]
    name = "samples_greedy" if greedy else "samples"
    out = _REPO / f"outputs/overrefusal/{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    if ids:
        want = set(ids.split(","))
        cands = [c for c in cands if c["id"] in want]
    if skip_done and out.exists():
        done = {json.loads(x)["id"] for x in out.read_text().splitlines() if x.strip()}
        cands = [c for c in cands if c["id"] not in done]
    if not cands:
        print("nothing to sample")
        return
    k = max(1, min(shards, len(cands)))
    parts = [cands[i::k] for i in range(k)]
    n = 0
    with out.open("a", encoding="utf-8") as fh:
        for rows in sample_shard.map(parts, kwargs={"greedy": greedy}):
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
    meta = {
        "subject_snapshot": SNAPSHOT,
        "vllm": VLLM,
        "params": GREEDY_PARAMS if greedy else PARAMS,
        "max_model_len": MAX_MODEL_LEN,
        "chat_template": "enable_thinking=False, no system prompt",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out.parent / f"{name}_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {n} samples for {len(cands)} prompts -> {out}")
