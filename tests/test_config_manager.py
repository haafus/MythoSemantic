from settings import EmbeddingSettings, settings


class TestEmbeddingSettings:
    def test_defaults_have_embedding_params(self):
        emb = settings.embedding
        assert emb.models
        assert emb.default_chunking
        assert emb.batch_size is None

    def test_override_via_constructor(self):
        emb = EmbeddingSettings(batch_size=64)
        assert emb.batch_size == 64

    def test_models_defaults(self):
        emb = EmbeddingSettings()
        assert len(emb.models) == 4
        assert "bge-m3" in emb.models

    def test_settings_embedding_is_embedding_settings(self):
        assert isinstance(settings.embedding, EmbeddingSettings)
