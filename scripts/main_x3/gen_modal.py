"""Plan 0028 step 2: on-policy rollouts of the new seeds by Qwen3.6-27B (Modal, one H200).

The WSB chat bank's generation settings, copied from global-workspace
``scripts/oracle_lens_evals/jlens_pr/modal_pipeline.py`` (stage 1): HF ``transformers.generate``,
bf16, SDPA, left-padded batches of 16, ``do_sample`` T=1.0, top-p 1.0, top-k 0 (full vocab; do
not inherit Qwen's top_k=20), max 512 new tokens, chat template with
``enable_thinking=False`` (an empty think block), output cut at the first stop id (EOS not
kept); an empty rollout is dropped. Additions: ``torch.manual_seed(7)`` (the WSB run did not
seed torch, so neither run is bit-reproducible) and labels that continue the WSB numbering per
source in seeded order (``chat-<source>-0075`` ...).

    MODAL_PROFILE=safety-sahan-2 /root/.local/bin/modal run scripts/main_x3/gen_modal.py \\
        --seeds <dir>/seeds.json --sources dailydialog --out <dir>/rollouts_dailydialog.json
"""

import json
from pathlib import Path

import modal

HF_PATH = "/hf"
MODEL_ID = "Qwen/Qwen3.6-27B"
FIRST_INDEX = 75  # the WSB bank used chat-<source>-0000 .. 0074

app = modal.App("hb28-main-x3-gen")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch>=2.11",
        "transformers>=5.11.0",
        "accelerate>=1.0",
        "huggingface_hub>=0.35",
    )
    .env(
        {
            "HF_HOME": HF_PATH,
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)
hf_cache = modal.Volume.from_name("jlens-hf-cache")


@app.function(image=image, volumes={HF_PATH: hf_cache}, gpu="H200", timeout=6 * 3600, memory=65536)
def generate(
    seeds: list[dict],
    wsb_prompts: list[dict],
    max_new: int = 512,
    seed: int = 7,
    batch_size: int = 16,
) -> dict:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    model.eval()
    tok.padding_side = "left"

    def prompt_ids(text: str) -> list[int]:
        enc = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
        )
        ids = enc["input_ids"] if hasattr(enc, "keys") else enc
        return [int(t) for t in ids]

    # template guard: the chat template must re-render every WSB prompt to its captured ids
    bad = [w["id"] for w in wsb_prompts if prompt_ids(w["prompt"]) != w["prompt_ids"]]
    if bad:
        raise RuntimeError(f"chat template drift: {len(bad)} WSB prompts differ, e.g. {bad[:3]}")
    guard = {"wsb_prompts_rerendered": len(wsb_prompts), "mismatches": 0}

    gc_eos = model.generation_config.eos_token_id
    stop_ids = {int(tok.eos_token_id)}
    if isinstance(gc_eos, list):
        stop_ids |= {int(x) for x in gc_eos}
    elif gc_eos is not None:
        stop_ids.add(int(gc_eos))
    if tok.pad_token_id is not None:
        stop_ids.add(int(tok.pad_token_id))

    rows, empty = [], []
    for b0 in range(0, len(seeds), batch_size):
        batch = seeds[b0 : b0 + batch_size]
        pids = [prompt_ids(s["text"]) for s in batch]
        enc = tok.pad({"input_ids": pids}, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                max_new_tokens=max_new,
                pad_token_id=tok.eos_token_id,
            )
        for s, p, row in zip(batch, pids, out, strict=True):
            oids: list[int] = []
            for t in row[enc["input_ids"].shape[1] :].tolist():
                if int(t) in stop_ids:
                    break
                oids.append(int(t))
            if not oids:
                empty.append(s["key"])
                continue
            rows.append(
                {
                    "seed_key": s["key"],
                    "source": s["source"],
                    "source_id": s["source_id"],
                    "prompt_len": len(p),
                    "input_ids": p + oids,
                    "hit_max_new": len(oids) >= max_new,
                }
            )
        print(f"[gen] {b0 + len(batch)}/{len(seeds)}", flush=True)
    return {
        "rows": rows,
        "empty": empty,
        "stop_ids": sorted(stop_ids),
        "versions": {
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
        },
        "template_guard": guard,
        "snapshot": str(getattr(model.config, "_commit_hash", None)),
    }


@app.local_entrypoint()
def main(seeds: str, sources: str, out: str, max_new: int = 512, seed: int = 7) -> None:
    doc = json.loads(Path(seeds).read_text())
    want = sources.split(",")
    todo = [s for s in doc["seeds"] if s["source"] in want]  # seeded order kept
    repo = Path(__file__).resolve().parents[2] / "data" / "wsbench_chat"
    wsb_rows = {r["id"]: r for r in json.loads((repo / "capture_rows.json").read_text())}
    wsb = [
        {
            "id": it["id"],
            "prompt": it["prompt"],
            "prompt_ids": wsb_rows[it["id"]]["input_ids"][: wsb_rows[it["id"]]["prompt_len"]],
        }
        for it in json.loads((repo / "items.json").read_text())["items"]
    ]
    res = generate.remote(todo, wsb, max_new=max_new, seed=seed)
    idx = {s: FIRST_INDEX for s in want}
    for r in res["rows"]:  # labels in seeded order, empty rollouts skipped (as the WSB run did)
        r["label"] = f"chat-{r['source']}-{idx[r['source']]:04d}"
        idx[r["source"]] += 1
    res["settings"] = {
        "engine": "hf-transformers generate",
        "dtype": "bfloat16",
        "attn": "sdpa",
        "batch_size": 16,
        "padding": "left",
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "max_new": max_new,
        "torch_seed": seed,
        "chat_template": "enable_thinking=False",
        "eos_kept": False,
    }
    Path(out).write_text(json.dumps(res))
    print(f"[gen] {len(res['rows'])} rollouts, {len(res['empty'])} empty -> {out}")
