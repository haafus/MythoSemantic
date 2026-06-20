from model_registry import model_to_key


class TestModelToKeyAsCollectionName:
    def test_returns_string(self):
        name = model_to_key("BAAI/bge-m3")
        assert isinstance(name, str)
        assert len(name) > 0

    def test_deterministic(self):
        assert model_to_key("model-a") == model_to_key("model-a")

    def test_different_models_different_names(self):
        assert model_to_key("model-a") != model_to_key("model-b")

    def test_no_slash(self):
        assert "/" not in model_to_key("BAAI/bge-m3")

    def test_no_dot(self):
        assert "." not in model_to_key("Qwen/Qwen3-Embedding-0.6B")

    def test_max_length(self):
        name = model_to_key("very-long-model-name/" + "a" * 100)
        assert len(name) <= 63
