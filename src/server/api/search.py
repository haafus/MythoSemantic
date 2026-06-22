import logging

from fastapi import APIRouter, BackgroundTasks

from server.schemas import SearchRequest, SearchResult, WarmupRequest
from server.services.embedding_index import embedding_index_service

router = APIRouter(prefix="/api/similarity", tags=["search"])
logger = logging.getLogger(__name__)

_warmed_models: set[str] = set()


@router.post("/search", response_model=list[SearchResult])
def search(request: SearchRequest):
    return embedding_index_service.search(request.model, request.query, request.top_k)


@router.post("/search/warmup")
def warmup_search(request: WarmupRequest, background_tasks: BackgroundTasks) -> dict:
    if request.model in _warmed_models:
        return {"model": request.model, "status": "complete"}
    _warmed_models.add(request.model)

    def _do_warmup(model: str) -> None:
        try:
            embedding_index_service.warmup(model)
        except Exception:
            logger.exception("Search warmup failed")
            _warmed_models.discard(model)

    background_tasks.add_task(_do_warmup, request.model)
    return {"model": request.model, "status": "queued"}
