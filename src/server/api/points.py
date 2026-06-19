from fastapi import APIRouter, HTTPException, Query

from server.schemas import NeighborsResponse, PointInfo
from server.services.embedding_index import embedding_index_service

router = APIRouter(prefix="/api/similarity", tags=["points"])


@router.get("/points/{model_key}/{point_id}", response_model=PointInfo)
def point_info(model_key: str, point_id: str, chunk_index: int | None = Query(None)):
    try:
        return embedding_index_service.get_point(model_key, point_id, chunk_index)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Point not found") from exc


@router.get("/points/{model_key}/{point_id}/neighbors", response_model=NeighborsResponse)
def point_neighbors(
    model_key: str,
    point_id: str,
    n: int = Query(10, ge=1, le=100),
    chunk_index: int | None = Query(None),
):
    try:
        neighbors = embedding_index_service.get_neighbors(model_key, point_id, n, chunk_index)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Point not found") from exc
    return {"point_id": point_id, "neighbors": neighbors}
