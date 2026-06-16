from fastapi import APIRouter

from server.schemas import ModelListResponse
from server.services.models import list_model_summaries

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("", response_model=ModelListResponse)
def list_models() -> dict:
    return {"models": list_model_summaries()}
