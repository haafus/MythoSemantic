import pytest
from pydantic import ValidationError

from server.schemas import (
    CatalogResponse,
    CorpusDocument,
    ModelListResponse,
    ModelSummary,
    NeighborsResponse,
    PointInfo,
    SavedPlotResponse,
    SearchRequest,
    SearchResponse,
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
        r = SearchResult(id="1", similarity_score=0.95, distance=0.05)
        assert r.tradition == "Unknown"
        assert r.major_tradition == ""
        assert r.chunk_index == 0
        assert r.text == ""
        assert r.text_preview == ""

    def test_all_fields(self):
        r = SearchResult(
            id="doc1",
            tradition="Greek",
            major_tradition="European",
            chunk_index=3,
            similarity_score=0.9,
            distance=0.1,
            text="full text",
            text_preview="full...",
            filename="doc.txt",
            book_title="The Book",
        )
        assert r.book_title == "The Book"


class TestSearchResponse:
    def test_structure(self):
        resp = SearchResponse(
            query="test",
            model="m1",
            results=[SearchResult(id="1", similarity_score=0.9, distance=0.1)],
            total=1,
        )
        assert len(resp.results) == 1


class TestPointInfo:
    def test_defaults(self):
        p = PointInfo(id="p1")
        assert p.text == ""
        assert p.tradition == "Unknown"
        assert p.metadata == {}

    def test_with_metadata(self):
        p = PointInfo(id="p1", metadata={"key": "val"})
        assert p.metadata["key"] == "val"


class TestNeighborsResponse:
    def test_structure(self):
        n = NeighborsResponse(
            point_id="p1",
            neighbors=[SearchResult(id="p2", similarity_score=0.8, distance=0.2)],
        )
        assert n.point_id == "p1"
        assert len(n.neighbors) == 1


class TestCorpusDocument:
    def test_defaults(self):
        doc = CorpusDocument(id="test")
        assert doc.color == "#6b7280"
        assert doc.word_count == 0
        assert doc.major_tradition == ""

    def test_all_fields(self):
        doc = CorpusDocument(
            id="Iliad",
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
            documents=[CorpusDocument(id="d1"), CorpusDocument(id="d2")],
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


class TestSavedPlotResponse:
    def test_not_found(self):
        r = SavedPlotResponse(exists=False, reason="not found")
        assert r.url is None

    def test_found(self):
        r = SavedPlotResponse(exists=True, url="/projections/model/plot.html", path="/tmp/plot.html")
        assert r.exists
