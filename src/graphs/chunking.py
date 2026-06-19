def chunk_text(text: str, max_chars: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + max_chars, text_len)
        if end < text_len:
            split_pos = max(text.rfind("\n", start, end), text.rfind(".", start, end), text.rfind(" ", start, end))
            if split_pos == -1 or split_pos <= start:
                split_pos = end
            else:
                split_pos += 1
        else:
            split_pos = end

        chunks.append(text[start:split_pos])
        next_start = split_pos - overlap
        start = split_pos if next_start <= start else next_start

    return chunks
