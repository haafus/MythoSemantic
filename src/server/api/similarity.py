from fastapi import APIRouter, HTTPException, Query

from embeddings import chroma_manager
from projections import PROJECTION_METHODS
from server.schemas import ModelListResponse, SearchRequest, SearchResult
from server.services.projections import get_projection_data
from server.services.similarity import similarity_service

router = APIRouter(prefix="/api/similarity", tags=["similarity"])


@router.get("/models", response_model=ModelListResponse)
def list_models() -> dict:
    models = [{"name": key, "key": key} for key in chroma_manager.get_available_models()]
    return {"models": models}


@router.post("/search", response_model=list[SearchResult])
def search(request: SearchRequest):
    return similarity_service.search(request.model, request.query, request.top_k)


@router.get("/points/{model}/{text_id}", response_model=list[SearchResult])
def point_info(
    model: str,
    text_id: str,
    chunk_index: int = Query(...),
    top_k: int = Query(1, ge=1, le=100),
):
    return similarity_service.get_point(
        model, text_id, chunk_index, top_k=top_k,
    )


@router.get("/methods")
def methods() -> list[dict]:
    return PROJECTION_METHODS


@router.get("/projections/{model}/{method}")
def projection(model: str, method: str) -> dict:
    data = get_projection_data(model, method)
    if not data:
        raise HTTPException(status_code=404, detail="Projection data not found")
    return data
