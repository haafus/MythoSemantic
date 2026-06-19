from embeddings.chroma_manager import get_available_models
from model_registry import model_to_key


def list_model_summaries() -> list[dict[str, str]]:
    result = []
    for model in get_available_models():
        key = model_to_key(model)
        result.append({"name": model, "key": key, "safe_dir": key})
    return result
