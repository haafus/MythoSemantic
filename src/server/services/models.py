import logging

from settings import Settings

logger = logging.getLogger(__name__)

_key_to_model_cache: dict[str, str] = {}


def model_to_key(model_name: str) -> str:
    return Settings.safe_model_name(model_name or "")


def key_to_model(model_key: str, models: list[str] | None = None) -> str:
    if not model_key:
        return model_key
    if "/" in model_key:
        return model_key
    if models is None and model_key in _key_to_model_cache:
        return _key_to_model_cache[model_key]

    candidates = models if models is not None else list_models_raw()
    for model in candidates:
        if model_to_key(model) == model_key:
            _key_to_model_cache[model_key] = model
            return model

    result = model_key.replace("_", "/")
    _key_to_model_cache[model_key] = result
    return result


def list_models_raw() -> list[str]:
    from projection.loader import EmbeddingDataLoader

    try:
        return EmbeddingDataLoader().get_available_models()
    except Exception:
        logger.exception("Failed to get available models from ChromaDB")
        return []


def list_model_summaries() -> list[dict[str, str]]:
    return [
        {
            "name": model,
            "key": model_to_key(model),
            "safe_dir": model_to_key(model),
        }
        for model in list_models_raw()
    ]


def get_model_output_dir(model_key: str):
    from settings import settings

    model_name = key_to_model(model_key)
    return settings.analysis_dir / model_to_key(model_name)
