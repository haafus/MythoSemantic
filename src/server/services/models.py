import logging

from model_registry import model_to_key

logger = logging.getLogger(__name__)


def list_models_raw() -> list[str]:
    from embeddings.chroma_manager import get_available_models

    try:
        return get_available_models()
    except Exception:
        logger.exception("Failed to get available models from ChromaDB")
        return []


def list_model_summaries() -> list[dict[str, str]]:
    result = []
    for model in list_models_raw():
        key = model_to_key(model)
        result.append({"name": model, "key": key, "safe_dir": key})
    return result


