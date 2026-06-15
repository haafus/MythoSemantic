from concurrent.futures import ThreadPoolExecutor
import logging
import threading
import time
from typing import Dict, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query

from ui_server.schemas import SearchRequest, SearchWarmupRequest
from ui_server.services.embedding_index import embedding_index_service
from ui_server.services.projections import get_projection_data, get_saved_html_plot

router = APIRouter(prefix="/api/similarity", tags=["similarity"])
logger = logging.getLogger(__name__)

_search_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="semantic-search")
_warmup_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="semantic-warmup")
_search_jobs: Dict[str, Dict] = {}
_search_jobs_lock = threading.Lock()
_SEARCH_JOB_TTL_SECONDS = 60 * 30
_warmup_status: Dict[str, Dict] = {}
_warmup_lock = threading.Lock()


@router.get("/projections/{model_key}/{method}")
def projection(model_key: str, method: str):
    data = get_projection_data(model_key, method)
    if not data:
        saved_html_plot = get_saved_html_plot(model_key, method)
        raise HTTPException(
            status_code=404,
            detail={
                "message": "Projection JSON not found",
                "saved_html_plot": saved_html_plot,
            },
        )
    return data


@router.get("/saved-html/{model_key}/{method}")
def saved_html_plot(model_key: str, method: str):
    return get_saved_html_plot(model_key, method)


@router.get("/points/{model_key}/{point_id}")
def point_info(model_key: str, point_id: str, chunk_index: Optional[int] = Query(None)):
    try:
        return embedding_index_service.get_point(model_key, point_id, chunk_index)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Point not found") from exc


@router.get("/points/{model_key}/{point_id}/neighbors")
def point_neighbors(
    model_key: str,
    point_id: str,
    n: int = Query(10, ge=1, le=100),
    chunk_index: Optional[int] = Query(None),
    exclude_same_tradition: bool = Query(False),
):
    try:
        neighbors = embedding_index_service.get_neighbors(
            model_key,
            point_id,
            n,
            chunk_index,
            exclude_same_tradition=exclude_same_tradition,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Point not found") from exc
    return {"point_id": point_id, "neighbors": neighbors}


def _cleanup_search_jobs_locked() -> None:
    cutoff = time.time() - _SEARCH_JOB_TTL_SECONDS
    expired = [
        job_id
        for job_id, job in _search_jobs.items()
        if job.get("status") in {"complete", "failed"} and job.get("finished_at", 0) < cutoff
    ]
    for job_id in expired:
        _search_jobs.pop(job_id, None)


def _set_search_job(job_id: str, **updates) -> None:
    with _search_jobs_lock:
        job = _search_jobs.get(job_id)
        if job is not None:
            job.update(updates)


def _run_search_job(job_id: str, model: str, query: str, top_k: int) -> None:
    _set_search_job(job_id, status="running", started_at=time.time())
    try:
        results = embedding_index_service.search(model, query, top_k)
        _set_search_job(
            job_id,
            status="complete",
            results=results,
            total=len(results),
            finished_at=time.time(),
        )
    except Exception as exc:
        logger.exception("Semantic search job failed")
        _set_search_job(
            job_id,
            status="failed",
            error=str(exc) or exc.__class__.__name__,
            finished_at=time.time(),
        )


def _run_warmup_job(model: str) -> None:
    with _warmup_lock:
        _warmup_status[model] = {"model": model, "status": "running", "started_at": time.time()}

    try:
        embedding_index_service.warmup(model)
        status = {"model": model, "status": "complete", "finished_at": time.time()}
    except Exception as exc:
        logger.exception("Semantic search warmup failed")
        status = {
            "model": model,
            "status": "failed",
            "error": str(exc) or exc.__class__.__name__,
            "finished_at": time.time(),
        }

    with _warmup_lock:
        _warmup_status[model] = status


@router.post("/search/jobs")
def start_search_job(request: SearchRequest):
    job_id = uuid4().hex
    now = time.time()
    job = {
        "job_id": job_id,
        "status": "queued",
        "query": request.query,
        "model": request.model,
        "top_k": request.top_k,
        "results": [],
        "total": 0,
        "submitted_at": now,
    }
    with _search_jobs_lock:
        _cleanup_search_jobs_locked()
        _search_jobs[job_id] = job

    _search_executor.submit(_run_search_job, job_id, request.model, request.query, request.top_k)
    return job


@router.post("/search/warmup")
def warmup_search(request: SearchWarmupRequest):
    with _warmup_lock:
        current = _warmup_status.get(request.model)
        if current and current.get("status") in {"queued", "running", "complete"}:
            return dict(current)

        status = {"model": request.model, "status": "queued", "submitted_at": time.time()}
        _warmup_status[request.model] = status

    _warmup_executor.submit(_run_warmup_job, request.model)
    return status


@router.get("/search/jobs/{job_id}")
def search_job(job_id: str):
    with _search_jobs_lock:
        job = _search_jobs.get(job_id)
        if job is None:
            raise HTTPException(
                status_code=404,
                detail="Search job not found. The server may have restarted; start a new search.",
            )
        return dict(job)


@router.post("/search")
def search(request: SearchRequest):
    try:
        results = embedding_index_service.search(request.model, request.query, request.top_k)
    except Exception as exc:
        logger.exception("Semantic search failed")
        raise HTTPException(
            status_code=503,
            detail=str(exc) or "Semantic search failed",
        ) from exc
    return {
        "query": request.query,
        "model": request.model,
        "results": results,
        "total": len(results),
    }
