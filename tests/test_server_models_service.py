from unittest.mock import patch

from server.services.models import (
    get_model_output_dir,
    key_to_model,
    list_model_summaries,
    list_models_raw,
    model_to_key,
)
from settings import settings


class TestModelToKey:
    def test_slash_replaced(self):
        assert model_to_key("BAAI/bge-m3") == "BAAI_bge-m3"

    def test_backslash_replaced(self):
        assert model_to_key("path\\model") == "path_model"

    def test_no_special_chars(self):
        assert model_to_key("simple-model") == "simple-model"

    def test_empty(self):
        assert model_to_key("") == ""


class TestKeyToModel:
    def test_passthrough_with_slash(self):
        assert key_to_model("BAAI/bge-m3") == "BAAI/bge-m3"

    def test_empty_string(self):
        assert key_to_model("") == ""

    def test_finds_in_models_list(self):
        result = key_to_model("BAAI_bge-m3", models=["BAAI/bge-m3", "other/model"])
        assert result == "BAAI/bge-m3"

    def test_fallback_replaces_underscore(self):
        result = key_to_model("unknown_xyz_abc", models=[])
        assert result == "unknown/xyz/abc"

    def test_first_match_wins(self):
        result = key_to_model("a_b", models=["a/b", "a_b"])
        assert result == "a/b"


class TestListModelsRaw:
    def test_returns_models_from_loader(self):
        with patch("projection.loader.EmbeddingDataLoader") as mock_cls:
            mock_cls.return_value.get_available_models.return_value = ["model/a", "model/b"]
            result = list_models_raw()
        assert result == ["model/a", "model/b"]

    def test_returns_empty_on_error(self):
        with patch("projection.loader.EmbeddingDataLoader", side_effect=Exception("no db")):
            result = list_models_raw()
        assert result == []


class TestListModelSummaries:
    def test_returns_dicts_with_keys(self):
        with patch("server.services.models.list_models_raw", return_value=["BAAI/bge-m3"]):
            result = list_model_summaries()
        assert len(result) == 1
        assert result[0]["name"] == "BAAI/bge-m3"
        assert result[0]["key"] == "BAAI_bge-m3"
        assert result[0]["safe_dir"] == "BAAI_bge-m3"


class TestGetModelOutputDir:
    def test_returns_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "projections_dir", tmp_path)
        result = get_model_output_dir("BAAI_bge-m3")
        assert result.parent == tmp_path
