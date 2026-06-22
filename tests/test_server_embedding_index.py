import numpy as np
import pytest

from server.services.embedding_index import EmbeddingIndexService, ModelIndex


def _make_index(items, matrix=None):
    if matrix is None:
        dim = 3
        matrix = np.random.randn(len(items), dim).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        matrix = matrix / norms

    id_to_index = {}
    for idx, item in enumerate(items):
        pid = str(item.get("id", idx))
        id_to_index.setdefault(pid, idx)
        ci = item.get("chunk_index")
        if ci is not None:
            id_to_index[f"{pid}::{ci}"] = idx

    return ModelIndex(model_name="test", items=items, normalized_matrix=matrix, id_to_index=id_to_index)


class TestTopResults:
    def _item(self, id, text="", tradition="Greek", **kw):
        return {"id": id, "text": text, "tradition": tradition, "major_tradition": "", "chunk_index": 0, "filename": "", **kw}

    def test_returns_top_k(self):
        items = [self._item("a"), self._item("b"), self._item("c")]
        index = _make_index(items)
        sims = np.array([0.9, 0.5, 0.1], dtype=np.float32)
        results = EmbeddingIndexService._top_results(index, sims, 2)
        assert len(results) == 2
        assert results[0]["id"] == "a"
        assert results[1]["id"] == "b"

    def test_sorted_by_similarity(self):
        items = [self._item("a"), self._item("b"), self._item("c")]
        index = _make_index(items)
        sims = np.array([0.3, 0.9, 0.6], dtype=np.float32)
        results = EmbeddingIndexService._top_results(index, sims, 3)
        scores = [r["similarity_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_index(self):
        index = _make_index([], np.zeros((0, 3), dtype=np.float32))
        results = EmbeddingIndexService._top_results(index, np.array([]), 5)
        assert results == []

    def test_skips_negative_inf(self):
        items = [self._item("a"), self._item("b")]
        index = _make_index(items)
        sims = np.array([0.9, -np.inf], dtype=np.float32)
        results = EmbeddingIndexService._top_results(index, sims, 2)
        assert len(results) == 1
        assert results[0]["id"] == "a"

    def test_result_fields(self):
        items = [self._item("doc", text="hello", tradition="Norse", major_tradition="Euro", filename="doc.txt")]
        index = _make_index(items)
        results = EmbeddingIndexService._top_results(index, np.array([0.8]), 1)
        r = results[0]
        assert r["id"] == "doc"
        assert r["tradition"] == "Norse"
        assert r["major_tradition"] == "Euro"
        assert r["similarity_score"] == 0.8
        assert r["filename"] == "doc.txt"
        assert "book_title" not in r
        assert "text_preview" not in r

    def test_limit_clamped_to_items(self):
        items = [self._item("a")]
        index = _make_index(items)
        results = EmbeddingIndexService._top_results(index, np.array([0.9]), 100)
        assert len(results) == 1

    def test_offset(self):
        items = [self._item("a"), self._item("b"), self._item("c")]
        index = _make_index(items)
        sims = np.array([0.9, 0.5, 0.1], dtype=np.float32)
        results = EmbeddingIndexService._top_results(index, sims, 2, offset=1)
        assert len(results) == 2
        assert results[0]["id"] == "b"
        assert results[1]["id"] == "c"

    def test_offset_beyond_items(self):
        items = [self._item("a"), self._item("b")]
        index = _make_index(items)
        sims = np.array([0.9, 0.5], dtype=np.float32)
        results = EmbeddingIndexService._top_results(index, sims, 5, offset=10)
        assert results == []
