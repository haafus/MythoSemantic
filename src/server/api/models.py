from fastapi import APIRouter

from embeddings.chroma_manager import ChromaStore
from model_registry import model_to_key
from server.schemas import ModelListResponse

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("", response_model=ModelListResponse)
def list_models() -> dict:
    models = [
        {"name": m, "key": model_to_key(m), "safe_dir": model_to_key(m)}
        for m in ChromaStore().get_available_models()
    ]
    return {"models": models}
