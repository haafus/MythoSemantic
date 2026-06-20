import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, ORJSONResponse
from fastapi.staticfiles import StaticFiles

from server.api import corpus, geography, models, points, projections, search
from settings import settings


def create_app() -> FastAPI:
    app = FastAPI(
        title="MythoScope UI Server",
        default_response_class=ORJSONResponse,
    )

    srv = settings.server
    app.add_middleware(GZipMiddleware, minimum_size=srv.gzip_minimum_size)

    app.include_router(models.router)
    app.include_router(corpus.router)
    app.include_router(geography.router)
    app.include_router(projections.router)
    app.include_router(points.router)
    app.include_router(search.router)

    assets_dir = settings.web_root / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
    if settings.corpus_dir.exists():
        app.mount("/corpus", StaticFiles(directory=str(settings.corpus_dir)), name="corpus")

    @app.middleware("http")
    async def add_cache_headers(request: Request, call_next) -> Response:
        response = await call_next(request)
        path = request.url.path

        if path.startswith("/api/") or path.startswith("/assets/"):
            response.headers["Cache-Control"] = "no-store"
        elif path.startswith("/corpus/"):
            response.headers["Cache-Control"] = f"public, max-age={srv.cache_max_age}"
        else:
            response.headers["Cache-Control"] = "no-cache"

        return response

    @app.get("/")
    def index() -> FileResponse:
        return _index_response()

    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str) -> FileResponse:
        if full_path in ("docs", "redoc", "openapi.json"):
            raise HTTPException(status_code=404, detail="Not found")
        if not (settings.web_root / "index.html").exists():
            raise HTTPException(status_code=404, detail="Not found")
        return _index_response()

    return app


def _index_response() -> FileResponse:
    return FileResponse(settings.web_root / "index.html")


def run_server() -> None:
    srv = settings.server
    uvicorn.run("main:app", host=srv.host, port=srv.port, reload=False)
