import pytest

pytest.importorskip("chromadb")

from corpus.corpus_iterator import CorpusFileInfo
from embedding.chroma_writer import ChromaWriter, _safe_id_part, _safe_meta


def _info(**overrides) -> CorpusFileInfo:
    defaults = {
        "filename": "unknown",
        "path": "",
        "text_id": "unknown",
        "catalog_id": "",
        "major_tradition": "unknown",
        "tradition": "unknown",
        "url": "",
    }
    defaults.update(overrides)
    return CorpusFileInfo(**defaults)


class TestSafeIdPart:
    def test_alphanumeric_unchanged(self):
        assert _safe_id_part("hello123") == "hello123"

    def test_special_chars_replaced(self):
        assert _safe_id_part("model/name:v2") == "model_name_v2"

    def test_none_becomes_unknown(self):
        assert _safe_id_part(None) == "unknown"

    def test_empty_becomes_unknown(self):
        assert _safe_id_part("") == "unknown"

    def test_dots_and_dashes_preserved(self):
        assert _safe_id_part("v1.2-beta") == "v1.2-beta"

    def test_leading_trailing_underscores_stripped(self):
        assert _safe_id_part("///name///") == "name"


class TestSafeMeta:
    def test_none_returns_empty_string(self):
        assert _safe_meta(None) == ""

    def test_string_returned_as_is(self):
        assert _safe_meta("hello") == "hello"

    def test_empty_string_returned(self):
        assert _safe_meta("") == ""


class TestBuildEntries:
    @pytest.fixture
    def writer(self):
        return ChromaWriter(chroma_batch_size=100)

    def test_ids_format(self, writer):
        chunks = ["chunk1", "chunk2", "chunk3"]
        info = _info(text_id="my_text", tradition="Buddhism")

        ids, metadatas = writer.build_entries(chunks, info, "BAAI/bge-m3", "paragraph")

        assert len(ids) == 3
        assert all("my_text" in id_ for id_ in ids)
        assert all("BAAI_bge-m3" in id_ for id_ in ids)
        assert ids[0].endswith("_0")
        assert ids[2].endswith("_2")

    def test_metadata_fields(self, writer):
        chunks = ["text"]
        info = _info(
            text_id="tid",
            filename="file.txt",
            tradition="Buddhism",
            major_tradition="Eastern",
            url="http://example.com",
        )

        ids, metadatas = writer.build_entries(chunks, info, "model-x", "sentence")

        assert len(metadatas) == 1
        m = metadatas[0]
        assert m["filename"] == "file"
        assert m["tradition"] == "Buddhism"
        assert m["major_tradition"] == "Eastern"
        assert m["model"] == "model-x"
        assert m["chunking"] == "sentence"
        assert m["chunk_index"] == 0

    def test_txt_extension_stripped_from_filename(self, writer):
        chunks = ["text"]
        info = _info(filename="document.txt")

        _, metadatas = writer.build_entries(chunks, info, "m", "c")
        assert metadatas[0]["filename"] == "document"

    def test_non_txt_extension_kept(self, writer):
        chunks = ["text"]
        info = _info(filename="document.pdf")

        _, metadatas = writer.build_entries(chunks, info, "m", "c")
        assert metadatas[0]["filename"] == "document.pdf"

    def test_default_info_uses_defaults(self, writer):
        chunks = ["text"]
        ids, metadatas = writer.build_entries(chunks, _info(), "model", "chunk")

        assert metadatas[0]["tradition"] == "unknown"
        assert metadatas[0]["filename"] == "unknown"
        assert "unknown" in ids[0]
