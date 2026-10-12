# ruff: noqa: E501
"""Plan 0005 capture (app ``sibling-ladder-capture``): final-token residuals for the entity banks
and the 35 basic_readout entity items, modelled on the Ali case study's ``modal_capture.py``.

ONE GPU, one 27B load (sharding would only multiply the ~5 min load). For every row of
``outputs/sibling/ladder/capture_rows.json`` it records the FINAL-token block output at the
requested layers (forward hooks on the decoder blocks = the lens read convention, layer L ==
HF ``hidden_states[L+1]``), the next-token log-probs on the given category word ids (the
knowledge filter), the top-5 next tokens and an 8-token greedy continuation. Plain render, no
special tokens; rows with ``expect_ids`` (the 35 items) must tokenize to them or the run aborts.

    MODAL_PROFILE=safety-sahan-2 uvx --python 3.12 modal@latest run --env camila-blank \\
        scripts/sibling_probe/ladder_capture_modal.py --run ladder1
    MODAL_PROFILE=safety-sahan-2 uvx --python 3.12 modal@latest volume get --env camila-blank \\
        sibling-probes ladder/ladder1 outputs/sibling/ladder/capture/
"""

import json
import os
from pathlib import Path, PurePosixPath

import modal

MODEL_ID = "Qwen/Qwen3.6-27B"
HF_PATH = "/hf"
VOL_PATH = "/vol"
OUT_ROOT = f"{VOL_PATH}/ladder"
GPU = os.environ.get("LADDER_GPU", "B300")
_HERE = Path(__file__).resolve()
REPO = _HERE.parents[2] if len(_HERE.parents) >= 3 else Path("/root")

app = modal.App("sibling-ladder-capture")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch>=2.11",
        "transformers>=5.11.0",
        "safetensors>=0.8.0",
        "numpy>=2.3",
        "accelerate>=1.0",
        "hf_transfer",
    )
    .env(
        {
            "HF_HOME": HF_PATH,
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)
hf_cache = modal.Volume.from_name("iolens-hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("sibling-probes", create_if_missing=True)
volumes: dict[str | PurePosixPath, modal.Volume] = {HF_PATH: hf_cache, VOL_PATH: data_vol}
secrets = [modal.Secret.from_name("huggingface")]


@app.function(image=image, volumes=volumes, gpu=GPU, secrets=secrets, timeout=3 * 3600)
def capture_fn(spec: dict, run: str) -> str:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ["HF_HUB_OFFLINE"] = "1"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    model.to("cuda").eval().requires_grad_(False)
    n_total = None
    blocks = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 48 and name.endswith("layers"):
            blocks, n_total = mod, len(mod)
            print(f"[capture] decoder blocks at {name!r} ({n_total})", flush=True)
            break
    assert blocks is not None, "decoder blocks not found"
    layers = [int(x) for x in spec["layers"]]
    word_ids = [int(x) for x in spec["word_ids"]]
    rows = spec["rows"]
    acts = np.zeros(
        (len(rows), len(layers), model.config.get_text_config().hidden_size), dtype=np.float16
    )
    store: dict[int, torch.Tensor] = {}

    def mk(i):
        def hook(_m, _inp, out):
            t = out if torch.is_tensor(out) else out[0]
            store[i] = t[0, -1].float().cpu()

        return hook

    handles = [blocks[L].register_forward_hook(mk(L)) for L in layers]
    meta = []
    wid = torch.tensor(word_ids, device="cuda")
    for i, r in enumerate(rows):
        ids = tok(r["prompt"], add_special_tokens=False)["input_ids"]
        if r.get("expect_ids") is not None and list(ids) != list(r["expect_ids"]):
            raise RuntimeError(f"{r['key']} tokenizes to {ids}, expected {r['expect_ids']}")
        x = torch.tensor([ids], device="cuda")
        store.clear()
        with torch.no_grad():
            logits = model(input_ids=x, use_cache=False).logits[0, -1].float()
        for j, L in enumerate(layers):
            acts[i, j] = store[L].numpy()
        logp = torch.log_softmax(logits, -1)
        top = torch.topk(logp, 5)
        for h in handles:
            h.remove()
        greedy = None
        if r.get("gen"):
            with torch.no_grad():
                g = model.generate(
                    x, max_new_tokens=8, do_sample=False, attention_mask=torch.ones_like(x)
                )
            greedy = tok.decode(g[0, len(ids) :].tolist())
        handles = [blocks[L].register_forward_hook(mk(L)) for L in layers]
        meta.append(
            {
                "key": r["key"],
                "n_tokens": len(ids),
                "top5": [tok.decode([t]) for t in top.indices.tolist()],
                "top5_logp": [round(float(v), 3) for v in top.values.tolist()],
                "word_logp": [round(float(v), 3) for v in logp[wid].tolist()],
                "greedy": greedy,
            }
        )
        if i % 100 == 0:
            print(
                f"[capture] {i}/{len(rows)} {r['key']} -> {meta[-1]['top5'][:3]} | {meta[-1]['greedy']!r}",
                flush=True,
            )
    for h in handles:
        h.remove()
    out = Path(OUT_ROOT) / run
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "acts.npy", acts)
    (out / "meta.json").write_text(
        json.dumps({"layers": layers, "word_ids": word_ids, "rows": meta}, ensure_ascii=False)
        + "\n"
    )
    data_vol.commit()
    return f"wrote {out}: acts {acts.shape}"


@app.local_entrypoint()
def main(run: str = "ladder1", limit: int = 0) -> None:
    spec = json.loads((REPO / "outputs" / "sibling" / "ladder" / "capture_rows.json").read_text())
    rows = spec["rows"]
    items = [r for r in rows if r.get("expect_ids") is not None]
    rest = [r for r in rows if r.get("expect_ids") is None]
    spec["rows"] = items + (rest[:limit] if limit else rest)  # items first: a mismatch aborts early
    print(f"[capture] {len(spec['rows'])} rows, layers {spec['layers']}, gpu {GPU}")
    print(capture_fn.remote(spec, run))
