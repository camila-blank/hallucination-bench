"""Host the four-eval HallucinationBench readout viewer (one static HTML file) on Modal, PUBLIC.

The page is built locally into ``outputs/readout_viewer/index.html`` (data embedded) and baked
into the image at deploy time, so redeploy after every rebuild. CPU-only, scales to zero; same
pattern as ``sibling_viewer.py`` (public, no key).

    python3 scripts/viewer/build_readout_viewer.py
    uvx --python 3.12 modal@latest deploy scripts/serve/readout_viewer.py
"""

import os
from pathlib import Path

import modal

SITE = Path(os.environ.get("RVIEW_SITE_DIR", "outputs/readout_viewer")).resolve()
if modal.is_local() and not (SITE / "index.html").exists():  # never runs inside the container
    raise SystemExit(f"site not found: {SITE} (run scripts/viewer/build_readout_viewer.py)")

app = modal.App("hbench-readout-viewer")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]==0.115.12")
    .add_local_file(str(SITE / "index.html"), "/site/index.html")
)


@app.function(image=image, cpu=0.5, memory=512, scaledown_window=600)
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
        return HTMLResponse("not found", status_code=404)

    return api
