from unittest.mock import patch

from model_registry import key_to_model, model_to_key
from server.services.models import list_model_summaries


class TestModelToKey:
    def test_slash_replaced(self):
        assert model_to_key("BAAI/bge-m3") == "BAAI_bge-m3"

    def test_no_special_chars(self):
        assert model_to_key("simple-model") == "simple-model"

    def test_empty(self):
        assert model_to_key("") == ""


class TestKeyToModel:
    def test_underscore_to_slash(self):
        assert key_to_model("BAAI_bge-m3") == "BAAI/bge-m3"

    def test_no_underscores(self):
        assert key_to_model("simple-model") == "simple-model"

    def test_empty_string(self):
        assert key_to_model("") == ""


class TestListModelSummaries:
    def test_returns_dicts_with_keys(self):
        with patch("server.services.models.get_available_models", return_value=["BAAI/bge-m3"]):
            result = list_model_summaries()
        assert len(result) == 1
        assert result[0]["name"] == "BAAI/bge-m3"
        assert result[0]["key"] == "BAAI_bge-m3"
        assert result[0]["safe_dir"] == "BAAI_bge-m3"
