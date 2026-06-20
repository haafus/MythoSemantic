from model_registry import model_name_for_key, model_to_key


class TestModelToKey:
    def test_slash_replaced(self):
        assert model_to_key("BAAI/bge-m3") == "BAAI_bge-m3"

    def test_dot_replaced(self):
        assert model_to_key("Qwen/Qwen3-Embedding-0.6B") == "Qwen_Qwen3-Embedding-0_6B"

    def test_no_special_chars(self):
        assert model_to_key("simple-model") == "simple-model"

    def test_empty(self):
        assert model_to_key("") == ""

    def test_max_length(self):
        assert len(model_to_key("a" * 100)) == 63


class TestModelNameForKey:
    def test_active_model(self):
        assert model_name_for_key("BAAI_bge-m3") == "BAAI/bge-m3"

    def test_inactive_model(self):
        assert model_name_for_key("Qwen_Qwen3-Embedding-4B") == "Qwen/Qwen3-Embedding-4B"

    def test_unknown_key_returns_key(self):
        assert model_name_for_key("unknown_model") == "unknown_model"
