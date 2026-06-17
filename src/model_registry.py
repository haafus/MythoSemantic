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
    return registry.get("embedding", {}).get(name, name)


def resolve_llm_provider(name: str) -> dict[str, str | None]:
    registry = _load_registry()
    llm = registry.get("llm", {})
    if name not in llm:
        available = ", ".join(sorted(llm.keys()))
        raise ValueError(f"LLM provider '{name}' not found. Available: {available}")
    entry = llm[name]
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
    return sorted(_load_registry().get("llm", {}).keys())


def list_embedding_aliases() -> dict[str, str]:
    return dict(_load_registry().get("embedding", {}))
