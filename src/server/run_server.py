import uvicorn
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, ORJSONResponse
from fastapi.staticfiles import StaticFiles

from server.api import corpus, geography, graphs, models, points, projections, search
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
    app.include_router(graphs.router)
    app.include_router(projections.router)
    app.include_router(points.router)
    app.include_router(search.router)

    assets_dir = settings.web_root / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(settings.web_root / "index.html")

    return app


def run_server() -> None:
    srv = settings.server
    uvicorn.run("main:app", host=srv.host, port=srv.port, reload=False)
