import json
import os
from pathlib import Path
from typing import Any

_registry: dict[str, Any] | None = None


def _load_registry() -> dict[str, Any]:
    global _registry
    if _registry is None:
        path = Path(__file__).resolve().parent.parent / "config" / "models.json"
        _registry = json.loads(path.read_text(encoding="utf-8"))
    return _registry


def resolve_embedding_model(name: str) -> str:
    registry = _load_registry()
    emb = registry.get("embedding", {})
    all_models = {**emb.get("models", {}), **emb.get("inactive", {})}
    return all_models.get(name, name)


def resolve_llm_provider(name: str) -> dict[str, str | None]:
    registry = _load_registry()
    llm = registry.get("llm", {})
    all_models = {**llm.get("models", {}), **llm.get("inactive", {})}
    if name not in all_models:
        available = ", ".join(sorted(all_models.keys()))
        raise ValueError(f"LLM provider '{name}' not found. Available: {available}")
    entry = all_models[name]
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
    emb = _load_registry().get("embedding", {})
    return {**emb.get("models", {}), **emb.get("inactive", {})}


def model_to_key(model_name: str) -> str:
    return (model_name or "").replace("/", "_").replace("\\", "_")


def key_to_model(model_key: str) -> str:
    return model_key.replace("_", "/")
