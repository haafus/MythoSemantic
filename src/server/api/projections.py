from fastapi import APIRouter, HTTPException

from projections import PROJECTION_METHODS
from server.services.projections import get_projection_data

router = APIRouter(prefix="/api/similarity", tags=["projections"])


@router.get("/methods")
def methods() -> list[dict]:
    return PROJECTION_METHODS


@router.get("/projections/{model}/{method}")
def projection(model: str, method: str) -> dict:
    data = get_projection_data(model, method)
    if not data:
        raise HTTPException(status_code=404, detail="Projection data not found")
    return data
