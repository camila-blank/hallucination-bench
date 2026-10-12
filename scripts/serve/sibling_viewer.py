"""Host the plan-0004 sibling-verdict readout viewer (a static site) on Modal, behind a link key.

``/`` is the viewer (``scripts/sibling_probe/viewer/readout_viewer.html``); it fetches
``/data/**.json`` (one file per item) on demand.
The site is built locally into ``outputs/sibling/viewer`` and baked into the image at deploy
time, so redeploy after every rebuild. CPU-only, scales to zero. Every route requires the key
(link-key gated like ``olensmeta-viewer``): open ``<url>/?key=<KEY>`` once; that sets an
HttpOnly cookie and redirects to a clean URL. Requests without the cookie or key get a 403.

    uv run python scripts/sibling_probe/viewer/build_viewer.py
    SIBVIEW_KEY="$(cat /workspace/.sibling_viewer_key)" \\
        uvx --python 3.12 modal@latest deploy scripts/serve/sibling_viewer.py

The key is never committed; it is passed as an inline Modal secret at deploy time.
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
    if not os.environ.get("SIBVIEW_KEY"):
        raise SystemExit("set SIBVIEW_KEY (the link key) for the deploy")

app = modal.App("sibling-viewer")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]==0.115.12")
    .add_local_dir(str(SITE), "/site")
)
key_secret = modal.Secret.from_dict(
    {"SIBVIEW_KEY": os.environ.get("SIBVIEW_KEY", "")} if modal.is_local() else {}
)
COOKIE = "sibview_k"
DATA_RE = re.compile(r"^data/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.json$")


@app.function(image=image, secrets=[key_secret], cpu=0.5, memory=1024, scaledown_window=600)
@modal.asgi_app()
def web():  # type: ignore[no-untyped-def]
    import hmac

    from fastapi import FastAPI, Request
    from fastapi.middleware.gzip import GZipMiddleware
    from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

    key = os.environ["SIBVIEW_KEY"]
    api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    api.add_middleware(GZipMiddleware, minimum_size=1024)

    def ok(value: str | None) -> bool:
        # bytes compare: str compare_digest raises on non-ASCII input (would 500, not 403)
        return bool(value) and hmac.compare_digest(str(value).encode(), key.encode())

    @api.middleware("http")
    async def gate(request: Request, call_next):  # type: ignore[no-untyped-def]
        if ok(request.query_params.get("key")):  # first visit: trade the key for a cookie
            resp = RedirectResponse(
                url=str(request.url.remove_query_params("key")), status_code=303
            )
            resp.set_cookie(
                COOKIE, key, httponly=True, secure=True, samesite="lax", max_age=90 * 86400
            )
            return resp
        if not ok(request.cookies.get(COOKIE)):
            return HTMLResponse(
                "<h3>403: this viewer needs its link key</h3><p>Open the full link you were given "
                "(…/?key=…).</p>",
                status_code=403,
            )
        return await call_next(request)

    @api.get("/{name:path}")
    def static(name: str):  # type: ignore[no-untyped-def]
        if name in ("", "index.html"):
            return FileResponse("/site/index.html", media_type="text/html")
        if DATA_RE.match(name) and ".." not in name and Path(f"/site/{name}").is_file():
            return FileResponse(
                f"/site/{name}",
                media_type="application/json",
                headers={"Cache-Control": "private, max-age=3600"},
            )
        return HTMLResponse("not found", status_code=404)

    return api
