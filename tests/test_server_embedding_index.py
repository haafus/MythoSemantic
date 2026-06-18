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


class TestPointKey:
    def test_no_chunk(self):
        assert EmbeddingIndexService._point_key("doc1") == "doc1"

    def test_with_chunk(self):
        assert EmbeddingIndexService._point_key("doc1", 3) == "doc1::3"

    def test_chunk_zero(self):
        assert EmbeddingIndexService._point_key("doc1", 0) == "doc1::0"

    def test_none_chunk(self):
        assert EmbeddingIndexService._point_key("doc1", None) == "doc1"


class TestNormalizeMatrix:
    def test_unit_vectors(self):
        m = np.array([[3.0, 4.0], [6.0, 8.0]], dtype=np.float32)
        result = EmbeddingIndexService._normalize_matrix(m)
        for row in result:
            np.testing.assert_allclose(np.linalg.norm(row), 1.0, atol=1e-6)

    def test_zero_vector_unchanged(self):
        m = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        result = EmbeddingIndexService._normalize_matrix(m)
        np.testing.assert_allclose(result[0], [0.0, 0.0])
        np.testing.assert_allclose(result[1], [1.0, 0.0])

    def test_already_normalized(self):
        m = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        result = EmbeddingIndexService._normalize_matrix(m)
        np.testing.assert_allclose(result, m)

    def test_preserves_direction(self):
        m = np.array([[3.0, 4.0]], dtype=np.float32)
        result = EmbeddingIndexService._normalize_matrix(m)
        np.testing.assert_allclose(result[0], [0.6, 0.8], atol=1e-6)


class TestTopResults:
    def _item(self, text_id, text="", tradition="Greek", **kw):
        return {"text_id": text_id, "text": text, "tradition": tradition, "major_tradition": "", "chunk_index": 0, "filename": "", **kw}

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

    def test_text_preview_truncated(self):
        items = [self._item("a", text="x" * 1000)]
        index = _make_index(items)
        results = EmbeddingIndexService._top_results(index, np.array([0.9]), 1)
        assert results[0]["text_preview"].endswith("...")
        assert len(results[0]["text_preview"]) == 703

    def test_text_preview_short_text(self):
        items = [self._item("a", text="short")]
        index = _make_index(items)
        results = EmbeddingIndexService._top_results(index, np.array([0.9]), 1)
        assert results[0]["text_preview"] == "short"

    def test_result_fields(self):
        items = [self._item("doc", text="hello", tradition="Norse", major_tradition="Euro", filename="doc.txt")]
        index = _make_index(items)
        results = EmbeddingIndexService._top_results(index, np.array([0.8]), 1)
        r = results[0]
        assert r["id"] == "doc"
        assert r["tradition"] == "Norse"
        assert r["major_tradition"] == "Euro"
        assert r["similarity_score"] == 0.8
        assert r["distance"] == pytest.approx(0.2)
        assert r["filename"] == "doc.txt"
        assert r["book_title"] == "doc"

    def test_limit_clamped_to_items(self):
        items = [self._item("a")]
        index = _make_index(items)
        results = EmbeddingIndexService._top_results(index, np.array([0.9]), 100)
        assert len(results) == 1
