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
    os.path.join(os.path.dirname(__file__), "..", "src", "embeddings", "chroma_manager.py"),
)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_collection_name = _mod.ChromaStore._collection_name


class TestCollectionNameForModel:
    def test_returns_string(self):
        name = _collection_name("BAAI/bge-m3")
        assert isinstance(name, str)
        assert len(name) > 0

    def test_deterministic(self):
        assert _collection_name("model-a") == _collection_name("model-a")

    def test_different_models_different_names(self):
        assert _collection_name("model-a") != _collection_name("model-b")

    def test_contains_hash(self):
        name = _collection_name("BAAI/bge-m3")
        assert "_" in name

    def test_safe_characters(self):
        name = _collection_name("sentence-transformers/LaBSE")
        assert "/" not in name

    def test_max_length(self):
        name = _collection_name("very-long-model-name/" + "a" * 100)
        assert len(name) <= 63


