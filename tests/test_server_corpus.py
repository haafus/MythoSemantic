import json

import server.services.corpus as corpus_mod
from server.services.corpus import (
    get_catalog_documents,
    get_traditions_info,
    read_document,
    resolve_document_path,
)
from settings import settings


def _make_corpus(tmp_path, docs=None):
    """Create a minimal corpus directory structure for testing."""
    if docs is None:
        docs = [("European", "Greek", "Iliad", "Sing, O goddess, the anger of Achilles")]

    for major, tradition, title, text in docs:
        doc_dir = tmp_path / major / tradition / title
        doc_dir.mkdir(parents=True, exist_ok=True)
        (doc_dir / f"{title}.txt").write_text(text)


def _patch_corpus(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "corpus_dir", tmp_path)


class TestResolveDocumentPath:
    def test_existing_file(self, tmp_path, monkeypatch):
        _make_corpus(tmp_path)
        _patch_corpus(monkeypatch, tmp_path)

        file_path, title = resolve_document_path("Iliad", "European", "Greek")
        assert file_path is not None
        assert file_path.exists()
        assert title == "Iliad"

    def test_missing_file(self, tmp_path, monkeypatch):
        _patch_corpus(monkeypatch, tmp_path)

        file_path, title = resolve_document_path("Missing", "No", "Where")
        assert file_path is not None
        assert not file_path.exists()
        assert title == "Missing"

    def test_path_traversal_sanitized(self, tmp_path, monkeypatch):
        _patch_corpus(monkeypatch, tmp_path)

        file_path, title = resolve_document_path("../../etc/passwd", "a", "b")
        assert file_path is not None
        assert tmp_path.resolve() in file_path.resolve().parents
        assert "/" not in title

    def test_sanitizes_special_chars(self, tmp_path, monkeypatch):
        _patch_corpus(monkeypatch, tmp_path)

        _, title = resolve_document_path('bad<>file:name', "a", "b")
        assert "<" not in title
        assert ">" not in title
        assert ":" not in title


class TestReadDocument:
    def test_reads_text(self, tmp_path, monkeypatch):
        _make_corpus(tmp_path)
        _patch_corpus(monkeypatch, tmp_path)

        text, title = read_document("Iliad", "European", "Greek")
        assert "Achilles" in text
        assert title == "Iliad"

    def test_missing_raises_not_found(self, tmp_path, monkeypatch):
        _patch_corpus(monkeypatch, tmp_path)

        import pytest

        with pytest.raises(FileNotFoundError):
            read_document("Nonexistent", "A", "B")

    def test_traversal_sanitized_raises_not_found(self, tmp_path, monkeypatch):
        _patch_corpus(monkeypatch, tmp_path)

        import pytest

        with pytest.raises(FileNotFoundError):
            read_document("../../etc/passwd", "a", "b")


class TestGetCatalogDocuments:
    def test_from_metadata_json(self, tmp_path, monkeypatch):
        metadata = [
            {"id": "Iliad", "major_tradition": "European", "tradition": "Greek"},
        ]
        (tmp_path / "corpus.json").write_text(json.dumps(metadata))
        _patch_corpus(monkeypatch, tmp_path)
        monkeypatch.setattr(corpus_mod, "_catalog_cache", {})

        docs = get_catalog_documents()
        assert len(docs) == 1
        assert docs[0]["id"] == "Iliad"

    def test_empty_corpus(self, tmp_path, monkeypatch):
        _patch_corpus(monkeypatch, tmp_path)
        monkeypatch.setattr(corpus_mod, "_catalog_cache", {})

        docs = get_catalog_documents()
        assert docs == []

    def test_cache_hit(self, tmp_path, monkeypatch):
        _patch_corpus(monkeypatch, tmp_path)
        monkeypatch.setattr(corpus_mod, "_catalog_cache", {})

        get_catalog_documents()
        (tmp_path / "corpus.json").write_text(json.dumps([{"id": "new"}]))
        docs = get_catalog_documents()
        assert docs == []

    def test_sorted_by_tradition(self, tmp_path, monkeypatch):
        metadata = [
            {"id": "B", "major_tradition": "Z", "tradition": "Z"},
            {"id": "A", "major_tradition": "A", "tradition": "A"},
        ]
        (tmp_path / "corpus.json").write_text(json.dumps(metadata))
        _patch_corpus(monkeypatch, tmp_path)
        monkeypatch.setattr(corpus_mod, "_catalog_cache", {})

        docs = get_catalog_documents()
        assert docs[0]["major_tradition"] == "A"
        assert docs[1]["major_tradition"] == "Z"


class TestGetTraditionsInfo:
    def test_from_corpus_dir(self, tmp_path, monkeypatch):
        info = {"Greek": {"color": "#ff0000", "description": "Ancient Greek"}}
        (tmp_path / "traditions.json").write_text(json.dumps(info))
        _patch_corpus(monkeypatch, tmp_path)

        result = get_traditions_info()
        assert result["Greek"]["color"] == "#ff0000"

    def test_missing_returns_empty(self, tmp_path, monkeypatch):
        _patch_corpus(monkeypatch, tmp_path)

        result = get_traditions_info()
        assert result == {}
