from fastapi import APIRouter, HTTPException, Query

from server.schemas import SearchResult
from server.services.embedding_index import embedding_index_service

router = APIRouter(prefix="/api/similarity", tags=["points"])


@router.get("/points/{model_key}/{point_id}", response_model=list[SearchResult])
def point_info(
    model_key: str,
    point_id: str,
    chunk_index: int = Query(...),
    neighbors: int = Query(0, ge=0, le=100),
    offset: int = Query(0, ge=0),
):
    try:
        return embedding_index_service.get_point(
            model_key, point_id, chunk_index, neighbors=neighbors, offset=offset,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Point not found") from exc
