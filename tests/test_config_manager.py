from settings import EmbeddingSettings, settings


class TestEmbeddingSettings:
    def test_defaults_have_embedding_params(self):
        emb = settings.embedding
        assert emb.default_chunking
        assert emb.batch_size == 32

    def test_override_via_constructor(self):
        emb = EmbeddingSettings(batch_size=64)
        assert emb.batch_size == 64

    def test_settings_embedding_is_embedding_settings(self):
        assert isinstance(settings.embedding, EmbeddingSettings)


class TestModelRegistry:
    def test_active_models_from_registry(self):
        from model_registry import active_embedding_models

        models = active_embedding_models()
        assert len(models) == 4
        assert "BAAI/bge-m3" in models

    def test_all_aliases_resolve(self):
        from model_registry import list_embedding_aliases, resolve_embedding_model

        for alias, full_name in list_embedding_aliases().items():
            assert resolve_embedding_model(alias) == full_name
