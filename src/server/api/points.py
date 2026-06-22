from fastapi import APIRouter, Query

from server.schemas import SearchResult
from server.services.embedding_index import embedding_index_service

router = APIRouter(prefix="/api/similarity", tags=["points"])


@router.get("/points/{model}/{text_id}", response_model=list[SearchResult])
def point_info(
    model: str,
    text_id: str,
    chunk_index: int = Query(...),
    top_k: int = Query(1, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    return embedding_index_service.get_point(
        model, text_id, chunk_index, top_k=top_k, offset=offset,
    )
