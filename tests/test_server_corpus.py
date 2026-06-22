import json

from corpus.utils import read_traditions, read_document
from server.services.corpus import get_catalog_documents
from settings import settings


def _make_corpus(tmp_path, docs=None):
    """Create a minimal corpus directory structure for testing."""
    if docs is None:
        docs = [("European", "Greek", "Iliad", "Sing, O goddess, the anger of Achilles")]

    for major, tradition, title, text in docs:
        doc_dir = tmp_path / major / tradition
        doc_dir.mkdir(parents=True, exist_ok=True)
        (doc_dir / f"{title}.txt").write_text(text)


def _patch_corpus(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "corpus_dir", tmp_path)


class TestReadDocument:
    def test_reads_text(self, tmp_path, monkeypatch):
        _make_corpus(tmp_path)
        _patch_corpus(monkeypatch, tmp_path)

        text, title = read_document(tmp_path, "Iliad", "European", "Greek")
        assert "Achilles" in text
        assert title == "Iliad"

    def test_missing_raises_not_found(self, tmp_path, monkeypatch):
        _patch_corpus(monkeypatch, tmp_path)

        import pytest

        with pytest.raises(FileNotFoundError):
            read_document(tmp_path, "Nonexistent", "A", "B")

    def test_traversal_sanitized_raises_not_found(self, tmp_path, monkeypatch):
        _patch_corpus(monkeypatch, tmp_path)

        import pytest

        with pytest.raises(FileNotFoundError):
            read_document(tmp_path, "../../etc/passwd", "a", "b")


class TestGetCatalogDocuments:
    def test_from_metadata_json(self, tmp_path, monkeypatch):
        metadata = [
            {"title": "Iliad", "major_tradition": "European", "tradition": "Greek"},
        ]
        traditions = {"Greek": {"color": "#ff0000"}}
        (tmp_path / "corpus.json").write_text(json.dumps(metadata))
        (tmp_path / "traditions.json").write_text(json.dumps(traditions))
        _patch_corpus(monkeypatch, tmp_path)

        docs = get_catalog_documents()
        assert len(docs) == 1
        assert docs[0]["title"] == "Iliad"
        assert docs[0]["color"] == "#ff0000"

    def test_missing_corpus_json_raises(self, tmp_path, monkeypatch):
        _patch_corpus(monkeypatch, tmp_path)

        import pytest

        with pytest.raises(FileNotFoundError):
            get_catalog_documents()

    def test_sorted_by_tradition(self, tmp_path, monkeypatch):
        metadata = [
            {"title": "B", "major_tradition": "Z", "tradition": "Z"},
            {"title": "A", "major_tradition": "A", "tradition": "A"},
        ]
        traditions = {"Z": {"color": "#000000"}, "A": {"color": "#ffffff"}}
        (tmp_path / "corpus.json").write_text(json.dumps(metadata))
        (tmp_path / "traditions.json").write_text(json.dumps(traditions))
        _patch_corpus(monkeypatch, tmp_path)

        docs = get_catalog_documents()
        assert docs[0]["major_tradition"] == "A"
        assert docs[1]["major_tradition"] == "Z"


class TestReadTraditions:
    def test_from_corpus_dir(self, tmp_path, monkeypatch):
        info = {"Greek": {"color": "#ff0000", "description": "Ancient Greek"}}
        (tmp_path / "traditions.json").write_text(json.dumps(info))
        _patch_corpus(monkeypatch, tmp_path)

        result = read_traditions(tmp_path)
        assert result["Greek"]["color"] == "#ff0000"

    def test_missing_returns_empty(self, tmp_path, monkeypatch):
        _patch_corpus(monkeypatch, tmp_path)

        result = read_traditions(tmp_path)
        assert result == {}
