import importlib.util
import os
import sys
import types

chromadb_stub = types.ModuleType("chromadb")
chromadb_stub.Collection = type("Collection", (), {})  # type: ignore[attr-defined]
chromadb_stub.PersistentClient = type("PersistentClient", (), {})  # type: ignore[attr-defined]
sys.modules["chromadb"] = chromadb_stub

_spec = importlib.util.spec_from_file_location(
    "chroma_manager",
    os.path.join(os.path.dirname(__file__), "..", "src", "embedding", "chroma_manager.py"),
)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

collection_name_for_model = _mod.collection_name_for_model
is_model_collection_name = _mod.is_model_collection_name


class TestCollectionNameForModel:
    def test_returns_string(self):
        name = collection_name_for_model("BAAI/bge-m3")
        assert isinstance(name, str)
        assert len(name) > 0

    def test_deterministic(self):
        assert collection_name_for_model("model-a") == collection_name_for_model("model-a")

    def test_different_models_different_names(self):
        assert collection_name_for_model("model-a") != collection_name_for_model("model-b")

    def test_contains_hash(self):
        name = collection_name_for_model("BAAI/bge-m3")
        assert "_" in name

    def test_safe_characters(self):
        name = collection_name_for_model("sentence-transformers/LaBSE")
        assert "/" not in name

    def test_max_length(self):
        name = collection_name_for_model("very-long-model-name/" + "a" * 100)
        assert len(name) <= 63


class TestIsModelCollectionName:
    def test_valid_collection_name(self):
        name = collection_name_for_model("BAAI/bge-m3")
        assert is_model_collection_name(name) is True

    def test_invalid_collection_name(self):
        assert is_model_collection_name("random_string") is False

    def test_empty_string(self):
        assert is_model_collection_name("") is False
