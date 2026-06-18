import logging

from model_registry import key_to_model, model_to_key

logger = logging.getLogger(__name__)


def list_models_raw() -> list[str]:
    from projections.loader import EmbeddingDataLoader

    try:
        return EmbeddingDataLoader().get_available_models()
    except Exception:
        logger.exception("Failed to get available models from ChromaDB")
        return []


def list_model_summaries() -> list[dict[str, str]]:
    result = []
    for model in list_models_raw():
        key = model_to_key(model)
        result.append({"name": model, "key": key, "safe_dir": key})
    return result


def get_model_output_dir(model_key: str):
    from settings import settings

    model_name = key_to_model(model_key)
    return settings.projections_dir / model_to_key(model_name)
