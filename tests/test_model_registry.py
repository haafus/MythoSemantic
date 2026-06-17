import pytest

from model_registry import (
    active_embedding_models,
    list_embedding_aliases,
    list_llm_providers,
    resolve_embedding_model,
    resolve_llm_provider,
)


class TestResolveEmbeddingModel:
    def test_resolves_alias(self):
        assert resolve_embedding_model("bge-m3") == "BAAI/bge-m3"

    def test_resolves_labse(self):
        assert resolve_embedding_model("labse") == "sentence-transformers/LaBSE"

    def test_passes_through_full_name(self):
        assert resolve_embedding_model("BAAI/bge-m3") == "BAAI/bge-m3"

    def test_passes_through_unknown(self):
        assert resolve_embedding_model("some/unknown-model") == "some/unknown-model"


class TestResolveLLMProvider:
    def test_resolves_known_provider(self):
        result = resolve_llm_provider("gpt-4o-mini")
        assert result["model"] == "gpt-4o-mini"
        assert "openai" in result["base_url"]

    def test_resolves_local_model(self):
        result = resolve_llm_provider("qwen3-8b")
        assert result["model"] == "qwen3:8b"
        assert "localhost" in result["base_url"]

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="not found"):
            resolve_llm_provider("nonexistent-model")


class TestListFunctions:
    def test_list_llm_providers(self):
        providers = list_llm_providers()
        assert "gpt-4o-mini" in providers
        assert "gemini-25-flash" in providers

    def test_list_embedding_aliases(self):
        aliases = list_embedding_aliases()
        assert aliases["bge-m3"] == "BAAI/bge-m3"
        assert "labse" in aliases

    def test_active_embedding_models(self):
        models = active_embedding_models()
        assert len(models) == 4
        assert models[0] == "BAAI/bge-m3"
        assert "Qwen/Qwen3-Embedding-4B" not in models

    def test_inactive_models_still_resolve(self):
        assert resolve_embedding_model("qwen-4b") == "Qwen/Qwen3-Embedding-4B"
        assert resolve_embedding_model("story-emb") == "uhhlt/story-emb"
