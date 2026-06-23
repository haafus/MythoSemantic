def chunk_text(
    text: str, chunk_size: int, chunk_overlap: int,
    separators: list[str] | None = None,
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
