import json

import pytest

from embedding.corpus_iterator import _normalize_catalog_id, iter_corpus_files


class TestNormalizeCatalogId:
    def test_strips_whitespace(self):
        assert _normalize_catalog_id("  hello  ") == "hello"

    def test_replaces_spaces_with_underscore(self):
        assert _normalize_catalog_id("hello world") == "hello_world"

    def test_collapses_multiple_spaces(self):
        assert _normalize_catalog_id("a  b   c") == "a_b_c"

    def test_none_becomes_empty(self):
        assert _normalize_catalog_id(None) == ""

    def test_integer_input(self):
        assert _normalize_catalog_id(42) == "42"


class TestIterCorpusFiles:
    def _create_corpus(self, tmp_path, items):
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()

        catalog = []
        for item in items:
            path = item["path"]
            p = corpus_dir / path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(item.get("content", "text"), encoding="utf-8")
            catalog.append({k: v for k, v in item.items() if k != "content"})

        with open(corpus_dir / "corpus.json", "w", encoding="utf-8") as f:
            json.dump(catalog, f)

        return corpus_dir

    def test_yields_files_from_catalog(self, tmp_path):
        corpus = self._create_corpus(tmp_path, [
            {"id": "file1", "tradition": "t1", "major_tradition": "mt1", "path": "mt1/t1/file1/file1.txt"},
            {"id": "file2", "tradition": "t2", "major_tradition": "mt2", "path": "mt2/t2/file2/file2.txt"},
        ])

        results = list(iter_corpus_files(corpus))
        filenames = {r["filename"] for r in results}
        assert filenames == {"file1.txt", "file2.txt"}

    def test_metadata_populates_fields(self, tmp_path):
        corpus = self._create_corpus(tmp_path, [
            {
                "id": "mytext",
                "tradition": "Buddhism",
                "major_tradition": "Eastern",
                "color": "#FF0000",
                "url": "http://example.com",
                "path": "Eastern/Buddhism/mytext/mytext.txt",
            },
        ])

        results = list(iter_corpus_files(corpus))
        assert len(results) == 1
        r = results[0]
        assert r["tradition"] == "Buddhism"
        assert r["major_tradition"] == "Eastern"
        assert r["color"] == "#FF0000"
        assert r["url"] == "http://example.com"
        assert r["text_id"] == "mytext"

    def test_missing_corpus_json_raises(self, tmp_path):
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="corpus.json not found"):
            list(iter_corpus_files(corpus_dir))

    def test_empty_catalog(self, tmp_path):
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        (corpus_dir / "corpus.json").write_text("[]")
        assert list(iter_corpus_files(corpus_dir)) == []

    def test_skips_entry_without_id(self, tmp_path):
        corpus = self._create_corpus(tmp_path, [
            {"id": "good", "tradition": "t", "major_tradition": "m", "path": "m/t/good/good.txt"},
        ])
        with open(corpus / "corpus.json") as f:
            catalog = json.load(f)
        catalog.append({"tradition": "t", "path": "no_id.txt"})
        with open(corpus / "corpus.json", "w") as f:
            json.dump(catalog, f)

        results = list(iter_corpus_files(corpus))
        assert len(results) == 1

    def test_skips_missing_file(self, tmp_path):
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        catalog = [{"id": "gone", "tradition": "t", "major_tradition": "m", "path": "m/t/gone/gone.txt"}]
        (corpus_dir / "corpus.json").write_text(json.dumps(catalog))

        results = list(iter_corpus_files(corpus_dir))
        assert results == []

    def test_does_not_read_file_content(self, tmp_path):
        corpus = self._create_corpus(tmp_path, [
            {"id": "big", "tradition": "t", "major_tradition": "m", "path": "m/t/big/big.txt", "content": "x" * 10000},
        ])
        results = list(iter_corpus_files(corpus))
        assert "content" not in results[0]
        assert "text" not in results[0]
