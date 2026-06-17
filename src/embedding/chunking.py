import logging
import re
from typing import Callable

logger = logging.getLogger(__name__)


class RegexPatterns:
    SENTENCE_SPLITTER = re.compile(r"(?<=[.!?])\s+|(?<=[。！？।])\s*")
    PARAGRAPH_SPLITTER = re.compile(r"\n\s*\n")


def character_based_chunking(
    text: str, chunk_size: int = 512, chunk_overlap: int = 64, separators: list[str] | None = None
) -> list[str]:
    if separators is None:
        separators = ["\n\n", "\n", ". ", "! ", "? ", "。", "！", "？", "।", "; ", ", ", "、", " ", ""]

    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    def _extract_tail(chunk: str, overlap_size: int) -> str:
        if overlap_size <= 0 or len(chunk) <= overlap_size:
            return ""
        tail_start = len(chunk) - overlap_size
        priority_seps = ["\n\n", "\n", ". ", "! ", "? ", "。", "！", "？", "।", "; ", ", ", "、", " ", ""]
        for sep in priority_seps:
            if not sep:
                continue
            search_start = max(0, tail_start - 50)
            last_sep_pos = chunk.rfind(sep, search_start, len(chunk))
            if last_sep_pos != -1 and last_sep_pos >= tail_start - 100:
                return chunk[last_sep_pos + len(sep) :]
        return chunk[-overlap_size:]

    def _split_recursive(text_to_split: str, seps: list[str], tail: str = "", depth: int = 0) -> list[str]:
        MAX_DEPTH = 10
        if depth > MAX_DEPTH:
            step = chunk_size - chunk_overlap
            if step <= 0:
                step = chunk_size // 2
            return [text_to_split[i : i + chunk_size] for i in range(0, len(text_to_split), step)]

        if not seps:
            full_text = tail + text_to_split
            if len(full_text) <= chunk_size:
                return [full_text]
            chunks = []
            step = chunk_size - chunk_overlap
            if step <= 0:
                step = chunk_size // 2
            for i in range(0, len(full_text), step):
                chunk = full_text[i : i + chunk_size]
                if chunk:
                    chunks.append(chunk)
            return chunks

        separator = seps[0]
        remaining_seps = seps[1:]
        if separator:
            splits = text_to_split.split(separator)
            splits = [s + (separator if i < len(splits) - 1 else "") for i, s in enumerate(splits) if s]
        else:
            splits = [text_to_split] if text_to_split else []

        if tail and splits:
            splits[0] = tail + splits[0]
            tail = ""
        chunks, current_chunk, current_tail = [], "", ""

        for split in splits:
            if len(current_chunk) + len(split) <= chunk_size:
                current_chunk += split
            else:
                if current_chunk:
                    current_tail = _extract_tail(current_chunk, chunk_overlap)
                    chunks.append(current_chunk)
                if len(split) > chunk_size:
                    sub_chunks = _split_recursive(split, remaining_seps, current_tail, depth + 1)
                    if sub_chunks:
                        chunks.extend(sub_chunks)
                        current_tail = _extract_tail(sub_chunks[-1], chunk_overlap)
                        current_chunk = ""
                    else:
                        current_chunk = current_tail
                else:
                    current_chunk = current_tail + split
                    current_tail = ""
        if current_chunk:
            chunks.append(current_chunk)
        return _merge_small_chunks(chunks, chunk_size)

    def _merge_small_chunks(chunks: list[str], min_size: int) -> list[str]:
        if not chunks:
            return []
        merged, current = [], chunks[0]
        for i in range(1, len(chunks)):
            next_chunk = chunks[i]
            if len(current) + len(next_chunk) <= min_size * 1.2:
                current += next_chunk
            else:
                merged.append(current)
                current = next_chunk
        merged.append(current)
        return merged

    return _split_recursive(text, separators)


def sentence_based_chunking(text: str, chunk_size: int = 512, chunk_overlap: int = 64) -> list[str]:
    if not text:
        return []
    sentences = RegexPatterns.SENTENCE_SPLITTER.split(text)
    sentences = [s for s in sentences if s.strip()]
    if not sentences:
        return [text] if text else []

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_length = 0
    overlap_sentences_count = max(1, chunk_overlap // 100) if chunk_overlap > 0 else 0

    for sent in sentences:
        sent_len = len(sent)
        if sent_len > chunk_size:
            if current_sentences:
                chunks.append(" ".join(current_sentences))
            chunks.extend(character_based_chunking(sent, chunk_size, chunk_overlap))
            current_sentences, current_length = [], 0
            continue

        if current_length + sent_len > chunk_size and current_sentences:
            chunks.append(" ".join(current_sentences))
            if overlap_sentences_count > 0 and len(current_sentences) >= overlap_sentences_count:
                current_sentences = current_sentences[-overlap_sentences_count:]
                current_length = sum(len(s) for s in current_sentences) + len(current_sentences) - 1
            else:
                current_sentences, current_length = [], 0

        current_sentences.append(sent)
        current_length += sent_len + 1

    if current_sentences:
        chunks.append(" ".join(current_sentences))
    return chunks


def paragraph_based_chunking(text: str, chunk_size: int = 1024, chunk_overlap: int = 128) -> list[str]:
    if not text:
        return []
    paragraphs = RegexPatterns.PARAGRAPH_SPLITTER.split(text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    if not paragraphs:
        return [text[:chunk_size]] if text else []

    chunks: list[str] = []
    current_paragraphs: list[str] = []
    current_length = 0
    overlap_para_count = max(1, chunk_overlap // 200) if chunk_overlap > 0 else 0

    for para in paragraphs:
        para_len = len(para)
        if para_len > chunk_size:
            if current_paragraphs:
                chunks.append("\n\n".join(current_paragraphs))
            chunks.extend(character_based_chunking(para, chunk_size, chunk_overlap))
            current_paragraphs, current_length = [], 0
            continue

        if current_length + para_len > chunk_size and current_paragraphs:
            chunks.append("\n\n".join(current_paragraphs))
            if overlap_para_count > 0 and len(current_paragraphs) >= overlap_para_count:
                current_paragraphs = current_paragraphs[-overlap_para_count:]
                current_length = sum(len(p) for p in current_paragraphs) + (len(current_paragraphs) - 1) * 2
            else:
                current_paragraphs, current_length = [], 0

        current_paragraphs.append(para)
        current_length += para_len + (2 if len(current_paragraphs) > 1 else 0)

    if current_paragraphs:
        chunks.append("\n\n".join(current_paragraphs))
    return chunks


class ChunkingStrategy:
    def __init__(
        self,
        name: str,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        chunking_func: Callable | None = None,
    ):
        if chunk_size < 10:
            raise ValueError("chunk_size must be at least 10 characters")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap cannot be greater than or equal to chunk_size")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        self.name = name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_func = chunking_func or character_based_chunking

    def __call__(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        return self.chunking_func(text, self.chunk_size, self.chunk_overlap)


def create_chunking_strategies() -> dict[str, ChunkingStrategy]:
    return {
        "character": ChunkingStrategy(
            name="character", chunk_size=512, chunk_overlap=64, chunking_func=character_based_chunking
        ),
        "sentence": ChunkingStrategy(
            name="sentence", chunk_size=512, chunk_overlap=64, chunking_func=sentence_based_chunking
        ),
        "paragraph": ChunkingStrategy(
            name="paragraph", chunk_size=1024, chunk_overlap=128, chunking_func=paragraph_based_chunking
        ),
    }
