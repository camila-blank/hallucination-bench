"""Host the safety-eval O-lens readout viewer on Modal, PRIVATE (secret path key).

The page embeds StrongREJECT jailbreak prompts and lens readouts with harmful content, so it is
never committed (``outputs/`` is gitignored) and never served publicly: the app answers only
``/<key>/`` and 404s every other path. The key lives in the Modal secret
``hbench-safety-viewer-key`` (env ``SAFETY_VIEWER_KEY``) and is generated at deploy time; never
write it into a tracked file. CPU-only, scales to zero. The pre-gzipped page
(``outputs/safety_viewer/index.html.gz``) is baked into the image, so redeploy after a rebuild.

    python3 scripts/viewer/build_safety_viewer.py
    KEY=$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')
    uvx --python 3.12 modal@latest secret create --force hbench-safety-viewer-key \\
        SAFETY_VIEWER_KEY="$KEY" >/dev/null
    uvx --python 3.12 modal@latest deploy scripts/serve/safety_viewer.py
    # URL: https://<workspace>--hbench-safety-viewer-web.modal.run/$KEY/

Rotate the link by re-running the secret command with a new key and redeploying.
"""

import os
from pathlib import Path

import modal

SITE = Path(os.environ.get("SVIEW_SITE_DIR", "outputs/safety_viewer")).resolve()
if modal.is_local() and not (SITE / "index.html.gz").exists():  # never runs inside the container
    raise SystemExit(f"site not found: {SITE} (run scripts/viewer/build_safety_viewer.py)")

app = modal.App("hbench-safety-viewer")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]==0.115.12")
    .add_local_file(str(SITE / "index.html.gz"), "/site/index.html.gz")
)


@app.function(
    image=image,
    cpu=0.5,
    memory=512,
    scaledown_window=300,
    min_containers=0,
    secrets=[modal.Secret.from_name("hbench-safety-viewer-key")],
)
@modal.asgi_app()
def web():  # type: ignore[no-untyped-def]
    import gzip
    import hmac

    from fastapi import FastAPI, Request
    from fastapi.responses import PlainTextResponse, RedirectResponse, Response

    key = os.environ["SAFETY_VIEWER_KEY"]
    assert len(key) >= 16, "SAFETY_VIEWER_KEY too short"
    blob = Path("/site/index.html.gz").read_bytes()
    headers = {
        "Cache-Control": "private, no-store",
        "X-Robots-Tag": "noindex, nofollow",
        "Referrer-Policy": "no-referrer",
    }
    api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    def not_found() -> Response:
        return PlainTextResponse("not found", status_code=404)

    @api.get("/{path:path}")
    def page(path: str, request: Request) -> Response:
        head, _, rest = path.partition("/")
        if not hmac.compare_digest(head.encode(), key.encode()):
            return not_found()
        if path == head:  # /<key> -> /<key>/
            return RedirectResponse(f"/{key}/", status_code=308, headers=headers)
        if rest not in ("", "index.html"):
            return not_found()
        if "gzip" in request.headers.get("accept-encoding", ""):
            return Response(
                blob,
                media_type="text/html; charset=utf-8",
                headers={**headers, "Content-Encoding": "gzip", "Vary": "Accept-Encoding"},
            )
        return Response(
            gzip.decompress(blob), media_type="text/html; charset=utf-8", headers=headers
        )

    return api
