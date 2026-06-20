import json
import os
from typing import Any

from settings import settings

_registry: dict[str, Any] | None = None


def _load_registry() -> dict[str, Any]:
    global _registry
    if _registry is None:
        path = settings.config_dir / "models.json"
        _registry = json.loads(path.read_text(encoding="utf-8"))
    return _registry


def resolve_embedding_model(name: str) -> str:
    registry = _load_registry()
    models = registry.get("embedding", {}).get("models", {})
    return models.get(name, name)


def resolve_llm_provider(name: str) -> dict[str, str | None]:
    registry = _load_registry()
    models = registry.get("llm", {}).get("models", {})
    if name not in models:
        available = ", ".join(sorted(models.keys()))
        raise ValueError(f"LLM provider '{name}' not found. Available: {available}")
    entry = models[name]
    api_key = None
    env_key = entry.get("env_key")
    if env_key:
        api_key = os.environ.get(env_key)
    return {
        "base_url": entry["base_url"],
        "model": entry["model"],
        "api_key": api_key,
    }


def list_llm_providers() -> list[str]:
    return sorted(_load_registry().get("llm", {}).get("models", {}).keys())


def active_embedding_models() -> list[str]:
    return list(_load_registry().get("embedding", {}).get("models", {}).values())


def list_embedding_aliases() -> dict[str, str]:
    return dict(_load_registry().get("embedding", {}).get("models", {}))


def model_to_key(model_name: str) -> str:
    return (model_name or "").replace("/", "_").replace(".", "_")[:63]


def model_name_for_key(key: str) -> str:
    emb = _load_registry().get("embedding", {})
    for section in ("models", "inactive"):
        for name in emb.get(section, {}).values():
            if model_to_key(name) == key:
                return name
    return key
