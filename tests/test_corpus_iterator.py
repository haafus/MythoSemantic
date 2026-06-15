import json

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
    def _create_corpus(self, tmp_path, files, metadata_items=None):
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()

        for rel_path, content in files.items():
            p = corpus_dir / rel_path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")

        if metadata_items is not None:
            metadata_file = corpus_dir / "corpus.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata_items, f)

        return corpus_dir

    def test_yields_txt_files(self, tmp_path):
        corpus = self._create_corpus(tmp_path, {
            "tradition1/sub/file1.txt": "content1",
            "tradition2/sub/file2.txt": "content2",
        })

        results = list(iter_corpus_files(corpus))
        filenames = {r["filename"] for r in results}
        assert "file1.txt" in filenames
        assert "file2.txt" in filenames

    def test_ignores_non_txt(self, tmp_path):
        corpus = self._create_corpus(tmp_path, {
            "file.txt": "text",
            "file.json": '{"key": "val"}',
            "file.md": "# heading",
        })

        results = list(iter_corpus_files(corpus))
        assert len(results) == 1
        assert results[0]["filename"] == "file.txt"

    def test_returns_all_files(self, tmp_path):
        corpus = self._create_corpus(tmp_path, {
            "a.txt": "aaa",
            "b.txt": "bbb",
        }, metadata_items=[
            {"id": "a", "tradition": "t1", "major_tradition": "mt1"},
            {"id": "b", "tradition": "t1", "major_tradition": "mt1"},
        ])

        results = list(iter_corpus_files(corpus))
        assert len(results) == 2

    def test_metadata_populates_fields(self, tmp_path):
        corpus = self._create_corpus(tmp_path, {
            "mytext.txt": "content",
        }, metadata_items=[
            {
                "id": "mytext",
                "tradition": "Buddhism",
                "major_tradition": "Eastern",
                "color": "#FF0000",
                "url": "http://example.com",
            },
        ])

        results = list(iter_corpus_files(corpus))
        assert len(results) == 1
        r = results[0]
        assert r["tradition"] == "Buddhism"
        assert r["major_tradition"] == "Eastern"
        assert r["color"] == "#FF0000"
        assert r["url"] == "http://example.com"

    def test_no_metadata_uses_directory_structure(self, tmp_path):
        corpus = self._create_corpus(tmp_path, {
            "Eastern/Buddhism/text.txt": "content",
        })

        results = list(iter_corpus_files(corpus))
        assert len(results) == 1
        r = results[0]
        assert r["major_tradition"] == "Eastern"
        assert r["tradition"] == "Buddhism"

    def test_empty_corpus(self, tmp_path):
        corpus = self._create_corpus(tmp_path, {})
        results = list(iter_corpus_files(corpus))
        assert results == []

    def test_does_not_read_file_content(self, tmp_path):
        corpus = self._create_corpus(tmp_path, {"big.txt": "x" * 10000})
        results = list(iter_corpus_files(corpus))
        assert "content" not in results[0]
        assert "text" not in results[0]
