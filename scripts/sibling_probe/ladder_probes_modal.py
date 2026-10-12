"""Plan 0005 probe fitting on Modal CPU (app ``sibling-ladder-probes``).

One container per (bank, layer) task (``ladder_probe.probe_task``); the capture files are read
memory-mapped from the ``sibling-probes`` volume (``/vol/ladder/<run>/acts.npy``, written by
``ladder_capture_modal.py``). Driven by ``ladder.py probes --modal`` (pickled tasks in, pickled
results out).

    MODAL_PROFILE=safety-sahan-2 uvx --python 3.12 modal@latest run --env camila-blank \\
        scripts/sibling_probe/ladder_probes_modal.py --inp tasks.pkl --out results.pkl
"""

import pickle
from pathlib import Path

import modal

_HERE = Path(__file__).resolve()
REPO = _HERE.parents[2] if len(_HERE.parents) >= 3 else Path("/root")

app = modal.App("sibling-ladder-probes")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy>=2.0", "scikit-learn>=1.5", "threadpoolctl>=3.0")
    .env({"PYTHONPATH": "/root/src", "PYTHONUNBUFFERED": "1"})
    .add_local_file(
        str(REPO / "src/hallucination_bench/evals/sibling_probe/ladder_probe.py"),
        "/root/src/ladder_probe.py",
    )
)
vol = modal.Volume.from_name("sibling-probes")


@app.function(image=image, volumes={"/vol": vol}, cpu=4.0, memory=16384, timeout=3 * 3600)
def fit(a: tuple) -> tuple:
    import ladder_probe as LP

    return LP.probe_task(a, threads=4)


@app.local_entrypoint()
def main(inp: str, out: str) -> None:
    tasks = pickle.loads(Path(inp).read_bytes())
    print(f"[probes-modal] {len(tasks)} tasks")
    res = list(fit.map(tasks, order_outputs=True))
    Path(out).write_bytes(pickle.dumps(res))
    print(f"[probes-modal] -> {out}")
