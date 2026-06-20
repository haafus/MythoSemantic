import uvicorn
from fastapi import FastAPI, HTTPException
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

    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str) -> FileResponse:
        if full_path in ("docs", "redoc", "openapi.json"):
            raise HTTPException(status_code=404, detail="Not found")
        index_html = settings.web_root / "index.html"
        if not index_html.exists():
            raise HTTPException(status_code=404, detail="Not found")
        return FileResponse(index_html)

    return app


def run_server() -> None:
    srv = settings.server
    uvicorn.run("main:app", host=srv.host, port=srv.port, reload=False)
