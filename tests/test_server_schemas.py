import pytest
from pydantic import ValidationError

from server.schemas import (
    CatalogResponse,
    CorpusDocument,
    ModelListResponse,
    ModelSummary,
    SearchRequest,
    SearchResult,
    TraditionsResponse,
)


class TestSearchRequest:
    def test_valid(self):
        req = SearchRequest(query="test query", model="model_a")
        assert req.top_k == 20

    def test_custom_top_k(self):
        req = SearchRequest(query="q", model="m", top_k=50)
        assert req.top_k == 50

    def test_empty_query_rejected(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="", model="m")

    def test_top_k_too_large(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="q", model="m", top_k=200)

    def test_top_k_zero(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="q", model="m", top_k=0)


class TestSearchResult:
    def test_defaults(self):
        r = SearchResult(id="1", similarity_score=0.95)
        assert r.tradition == "Unknown"
        assert r.major_tradition == ""
        assert r.chunk_index == 0
        assert r.text == ""

    def test_all_fields(self):
        r = SearchResult(
            id="doc1",
            tradition="Greek",
            major_tradition="European",
            chunk_index=3,
            similarity_score=0.9,
            text="full text",
            filename="doc.txt",
        )
        assert r.filename == "doc.txt"


class TestCorpusDocument:
    def test_defaults(self):
        doc = CorpusDocument(title="test")
        assert doc.color == "#6b7280"
        assert doc.word_count == 0
        assert doc.major_tradition == ""

    def test_all_fields(self):
        doc = CorpusDocument(
            title="Iliad",
            major_tradition="European",
            tradition="Greek",
            language="en",
            type="epic",
            word_count=12000,
        )
        assert doc.tradition == "Greek"


class TestCatalogResponse:
    def test_empty(self):
        resp = CatalogResponse(documents=[], total=0)
        assert resp.total == 0

    def test_with_documents(self):
        resp = CatalogResponse(
            documents=[CorpusDocument(title="d1"), CorpusDocument(title="d2")],
            total=2,
        )
        assert len(resp.documents) == 2


class TestTraditionsResponse:
    def test_structure(self):
        resp = TraditionsResponse(traditions={"Greek": {"color": "#red"}}, total=1)
        assert resp.total == 1


class TestModelSummary:
    def test_fields(self):
        m = ModelSummary(name="BAAI/bge-m3", key="BAAI_bge-m3")
        assert m.name == "BAAI/bge-m3"


class TestModelListResponse:
    def test_structure(self):
        resp = ModelListResponse(
            models=[ModelSummary(name="m1", key="k1")]
        )
        assert len(resp.models) == 1
