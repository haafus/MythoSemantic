import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "embedding_chunking",
    os.path.join(os.path.dirname(__file__), "..", "src", "embeddings", "chunking.py"),
)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
chunk_text = _mod.chunk_text


class TestChunkText:
    def test_short_text_single_chunk(self):
        text = "Hello world."
        chunks = chunk_text(text, chunk_size=512, chunk_overlap=64)
        assert chunks == [text]

    def test_empty_text(self):
        assert chunk_text("", chunk_size=512, chunk_overlap=64) == []

    def test_all_text_covered(self):
        text = "Word " * 200
        chunks = chunk_text(text.strip(), chunk_size=100, chunk_overlap=20)
        joined = " ".join(c.strip() for c in chunks)
        for word in text.strip().split():
            assert word in joined

    def test_chunks_respect_max_size(self):
        text = "A " * 500
        chunks = chunk_text(text.strip(), chunk_size=100, chunk_overlap=10)
        for chunk in chunks:
            assert len(chunk) <= 120  # allow small overrun from merge

    def test_overlap_produces_more_chunks(self):
        text = "Word " * 100
        no_overlap = chunk_text(text, chunk_size=100, chunk_overlap=0)
        with_overlap = chunk_text(text, chunk_size=100, chunk_overlap=30)
        assert len(with_overlap) >= len(no_overlap)

    def test_overlap_preserved_after_recursive_split(self):
        short_a = "A" * 50
        long_b = "B" * 200
        short_c = "C" * 50
        text = f"{short_a}\n\n{long_b}\n\n{short_c}"
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
        last_chunk = chunks[-1]
        assert "B" in last_chunk and "C" in last_chunk
