"""Host the plan-0004 sibling-verdict readout viewer (a static site) on Modal, PUBLIC (no key).

``/`` is the viewer (``scripts/sibling_probe/viewer/readout_viewer.html``); it fetches
``/data/**.json`` (one file per item) on demand.
The site is built locally into ``outputs/sibling/viewer`` and baked into the image at deploy
time, so redeploy after every rebuild. CPU-only, scales to zero. Public since 2026-10-12
(Camila: "it can be public"); the link-key gate was removed.

    uv run python scripts/sibling_probe/viewer/build_viewer.py
    uvx --python 3.12 modal@latest deploy scripts/serve/sibling_viewer.py
"""

import os
import re
from pathlib import Path

import modal

SITE = Path(os.environ.get("SIBVIEW_SITE_DIR", "outputs/sibling/viewer")).resolve()
if modal.is_local():  # never runs inside the container
    if not ((SITE / "index.html").exists() and (SITE / "data" / "index.json").exists()):
        raise SystemExit(
            f"site not found: {SITE} (run scripts/sibling_probe/viewer/build_viewer.py)"
        )

app = modal.App("sibling-viewer")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]==0.115.12")
    .add_local_dir(str(SITE), "/site")
)
DATA_RE = re.compile(r"^data/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.json$")


@app.function(image=image, cpu=0.5, memory=1024, scaledown_window=600)
@modal.asgi_app()
def web():  # type: ignore[no-untyped-def]
    from fastapi import FastAPI
    from fastapi.middleware.gzip import GZipMiddleware
    from fastapi.responses import FileResponse, HTMLResponse

    api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    api.add_middleware(GZipMiddleware, minimum_size=1024)

    @api.get("/{name:path}")
    def static(name: str):  # type: ignore[no-untyped-def]
        if name in ("", "index.html"):
            return FileResponse("/site/index.html", media_type="text/html")
        if DATA_RE.match(name) and ".." not in name and Path(f"/site/{name}").is_file():
            return FileResponse(
                f"/site/{name}",
                media_type="application/json",
                headers={"Cache-Control": "public, max-age=3600"},
            )
        return HTMLResponse("not found", status_code=404)

    return api
