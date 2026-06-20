from fastapi import APIRouter

from embeddings import chroma_manager
from server.schemas import ModelListResponse

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("", response_model=ModelListResponse)
def list_models() -> dict:
    models = [{"name": key, "key": key} for key in chroma_manager.get_available_models()]
    return {"models": models}
