import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from server.schemas import SearchRequest, SearchResponse, WarmupRequest
from server.services.embedding_index import embedding_index_service
from settings import settings

router = APIRouter(prefix="/api/similarity", tags=["search"])
logger = logging.getLogger(__name__)

_search_executor = ThreadPoolExecutor(max_workers=settings.server.search_max_workers, thread_name_prefix="semantic-search")
_warmup_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="semantic-warmup")
_search_jobs: dict[str, dict] = {}
_search_jobs_lock = threading.Lock()
_warmup_status: dict[str, dict] = {}
_warmup_lock = threading.Lock()


def _cleanup_search_jobs_locked() -> None:
    cutoff = time.time() - settings.server.search_job_ttl_seconds
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
    except Exception:
        logger.exception("Semantic search job failed")
        _set_search_job(
            job_id,
            status="failed",
            error="Search failed",
            finished_at=time.time(),
        )


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> dict:
    try:
        results = embedding_index_service.search(request.model, request.query, request.top_k)
    except Exception as exc:
        logger.exception("Semantic search failed")
        raise HTTPException(
            status_code=503,
            detail="Semantic search unavailable",
        ) from exc
    return {
        "query": request.query,
        "model": request.model,
        "results": results,
        "total": len(results),
    }


@router.post("/search/jobs")
def start_search_job(request: SearchRequest) -> dict:
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


@router.get("/search/jobs/{job_id}")
def search_job(job_id: str) -> dict:
    with _search_jobs_lock:
        _cleanup_search_jobs_locked()
        job = _search_jobs.get(job_id)
        if job is None:
            raise HTTPException(
                status_code=404,
                detail="Search job not found. The server may have restarted; start a new search.",
            )
        return dict(job)


def _run_warmup(model: str) -> None:
    with _warmup_lock:
        _warmup_status[model] = {"model": model, "status": "running", "started_at": time.time()}
    try:
        embedding_index_service.warmup(model)
        status = {"model": model, "status": "complete", "finished_at": time.time()}
    except Exception:
        logger.exception("Search warmup failed")
        status = {"model": model, "status": "failed", "finished_at": time.time()}
    with _warmup_lock:
        _warmup_status[model] = status


@router.post("/search/warmup")
def warmup_search(request: WarmupRequest) -> dict:
    with _warmup_lock:
        current = _warmup_status.get(request.model)
        if current and current.get("status") in {"queued", "running", "complete"}:
            return dict(current)
        status = {"model": request.model, "status": "queued", "submitted_at": time.time()}
        _warmup_status[request.model] = status
    _warmup_executor.submit(_run_warmup, request.model)
    return status
