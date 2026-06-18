import importlib.util
import os

import pytest

# Load module directly from file to avoid __init__.py pulling in chromadb
_spec = importlib.util.spec_from_file_location(
    "embedding_chunking",
    os.path.join(os.path.dirname(__file__), "..", "src", "embedding", "chunking.py"),
)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
character_based_chunking = _mod.character_based_chunking
sentence_based_chunking = _mod.sentence_based_chunking
paragraph_based_chunking = _mod.paragraph_based_chunking
ChunkingStrategy = _mod.ChunkingStrategy
create_chunking_strategies = _mod.create_chunking_strategies
RegexPatterns = _mod.RegexPatterns


class TestCharacterChunking:
    def test_short_text_single_chunk(self):
        text = "Hello world."
        chunks = character_based_chunking(text, chunk_size=512)
        assert chunks == [text]

    def test_empty_text(self):
        assert character_based_chunking("", chunk_size=512) == []

    def test_all_text_covered(self):
        text = "Word " * 200
        chunks = character_based_chunking(text.strip(), chunk_size=100, chunk_overlap=20)
        joined = " ".join(c.strip() for c in chunks)
        for word in text.strip().split():
            assert word in joined

    def test_chunks_respect_max_size(self):
        text = "A " * 500
        chunks = character_based_chunking(text.strip(), chunk_size=100, chunk_overlap=10)
        for chunk in chunks:
            assert len(chunk) <= 120  # allow small overrun from merge

    def test_overlap_produces_more_chunks(self):
        text = "Word " * 100
        no_overlap = character_based_chunking(text, chunk_size=100, chunk_overlap=0)
        with_overlap = character_based_chunking(text, chunk_size=100, chunk_overlap=30)
        assert len(with_overlap) >= len(no_overlap)

    def test_validation_chunk_size_too_small(self):
        with pytest.raises(ValueError, match="chunk_size must be at least 10"):
            ChunkingStrategy(name="test", chunk_size=5)

    def test_validation_overlap_exceeds_size(self):
        with pytest.raises(ValueError, match="chunk_overlap cannot be greater"):
            ChunkingStrategy(name="test", chunk_size=100, chunk_overlap=100)

    def test_validation_negative_overlap(self):
        with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
            ChunkingStrategy(name="test", chunk_size=100, chunk_overlap=-1)


class TestSentenceChunking:
    def test_short_text_single_chunk(self):
        text = "One sentence."
        chunks = sentence_based_chunking(text, chunk_size=512)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_splits_on_sentence_boundaries(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = sentence_based_chunking(text, chunk_size=40, chunk_overlap=0)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert chunk.endswith(".") or chunk.endswith(". ")

    def test_empty_text(self):
        assert sentence_based_chunking("", chunk_size=512) == []

    def test_all_sentences_present(self):
        sentences = [f"Sentence number {i}." for i in range(20)]
        text = " ".join(sentences)
        chunks = sentence_based_chunking(text, chunk_size=100, chunk_overlap=20)
        joined = " ".join(chunks)
        for s in sentences:
            assert s in joined

    def test_long_sentence_falls_back_to_character(self):
        long = "A" * 600
        text = f"Short. {long} End."
        chunks = sentence_based_chunking(text, chunk_size=100, chunk_overlap=10)
        assert len(chunks) >= 2


class TestParagraphChunking:
    def test_single_paragraph(self):
        text = "One paragraph only."
        chunks = paragraph_based_chunking(text, chunk_size=1024)
        assert chunks == [text]

    def test_splits_on_blank_lines(self):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = paragraph_based_chunking(text, chunk_size=30, chunk_overlap=0)
        assert len(chunks) >= 2

    def test_empty_text(self):
        assert paragraph_based_chunking("", chunk_size=1024) == []

    def test_all_paragraphs_present(self):
        paragraphs = [f"This is paragraph {i} with some extra words." for i in range(10)]
        text = "\n\n".join(paragraphs)
        chunks = paragraph_based_chunking(text, chunk_size=100, chunk_overlap=20)
        joined = " ".join(chunks)
        for p in paragraphs:
            assert p in joined


class TestChunkingStrategy:
    def test_call_returns_chunks(self):
        strategy = ChunkingStrategy(name="test", chunk_size=100, chunk_overlap=10)
        text = "Word " * 50
        chunks = strategy(text)
        assert len(chunks) >= 1
        assert all(isinstance(c, str) for c in chunks)

    def test_empty_text(self):
        strategy = ChunkingStrategy(name="test", chunk_size=100, chunk_overlap=10)
        assert strategy("") == []


class TestCreateChunkingStrategies:
    def test_returns_three_strategies(self):
        strategies = create_chunking_strategies()
        assert set(strategies.keys()) == {"character", "sentence", "paragraph"}

    def test_strategies_callable(self):
        strategies = create_chunking_strategies()
        text = "Some test text. " * 20
        for name, strategy in strategies.items():
            chunks = strategy(text)
            assert len(chunks) >= 1, f"Strategy {name} returned no chunks"


class TestRegexPatterns:
    def test_sentence_splitter_english(self):
        text = "Hello world. How are you? Fine! Thanks."
        parts = RegexPatterns.SENTENCE_SPLITTER.split(text)
        non_empty = [p for p in parts if p.strip()]
        assert len(non_empty) >= 3

    def test_paragraph_splitter(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        parts = RegexPatterns.PARAGRAPH_SPLITTER.split(text)
        non_empty = [p for p in parts if p.strip()]
        assert len(non_empty) == 3
