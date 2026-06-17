import logging
import re

logger = logging.getLogger(__name__)


def clean_gutenberg_text(text: str, filename: str | None = None) -> str:
    if not text:
        return text

    original_length = len(text)
    debug_info = filename if filename else "text"

    start_patterns = [
        (r"\*\*\* START OF (?:THE |THIS )?PROJECT GUTENBERG EBOOK[^*]*\*\*\*", "START OF PROJECT GUTENBERG"),
        (r"\*\*\* START OF THIS PROJECT GUTENBERG EBOOK[^*]*\*\*\*", "START OF THIS PROJECT GUTENBERG"),
        (r"\*\*\*THE PROJECT GUTENBERG EBOOK[^*]*\*\*\*", "THE PROJECT GUTENBERG EBOOK"),
        (r"Produced by .*?\n{2,}", "Produced by"),
    ]

    end_patterns = [
        (r"\*\*\* END OF (?:THE |THIS )?PROJECT GUTENBERG EBOOK[^*]*\*\*\*", "END OF PROJECT GUTENBERG"),
        (r"\*\*\* END OF THIS PROJECT GUTENBERG EBOOK[^*]*\*\*\*", "END OF THIS PROJECT GUTENBERG"),
        (r"End of (?:the )?Project Gutenberg[^\n]*", "End of Project Gutenberg"),
    ]

    start_pos = 0
    for pattern, description in start_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            start_pos = match.end()
            while start_pos < len(text) and text[start_pos] in "\n\r":
                start_pos += 1
            logger.debug(f"{debug_info}: Text start marker found: {description}")
            break

    end_pos = len(text)
    for pattern, description in end_patterns:
        match = re.search(pattern, text[start_pos:], re.IGNORECASE | re.MULTILINE)
        if match:
            end_pos = start_pos + match.start()
            logger.debug(f"{debug_info}: Text end marker found: {description}")
            break

    cleaned_text = text[start_pos:end_pos].strip()

    if not cleaned_text:
        logger.warning(f"{debug_info}: Could not extract text, returning original")
        return text

    cleaned_text, _footnote_count = _remove_gutenberg_footer_notes_with_count(cleaned_text)
    cleaned_text = _normalize_gutenberg_whitespace(cleaned_text)
    cleaned_text = _remove_header_metadata(cleaned_text)

    logger.debug(f"{debug_info}: Text cleanup: {original_length} -> {len(cleaned_text)} characters")

    return cleaned_text


def _remove_gutenberg_footer_notes_with_count(text: str) -> tuple[str, int]:
    footnote_count = 0

    footnote_patterns = [
        (r"\n\nFOOTNOTES:\n.*?(?=\n\n\*\*\* END|\Z)", "FOOTNOTES section"),
        (r"\n\n\*\s*FOOTNOTES?\s*\*\n.*?(?=\n\n\*\*\* END|\Z)", "FOOTNOTES with asterisks"),
        (r"\n\n\[?\d+\] .*?(?=\n\n\*\*\* END|\Z)", "numbered footnotes"),
    ]

    for pattern, description in footnote_patterns:
        matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if matches:
            footnote_count += len(matches)
            text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
            logger.debug(f"Removed {len(matches)} footnotes of type '{description}'")

    return text, footnote_count


def _normalize_gutenberg_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)

    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        if re.match(r"^[\s*_\-=]{10,}$", line):
            continue
        cleaned_lines.append(line.rstrip())

    return "\n".join(cleaned_lines).strip()


def _remove_header_metadata(text: str) -> str:
    lines = text.split("\n")

    metadata_patterns = [
        r"^Translated by ",
        r"^Edited by ",
        r"^With an Introduction by ",
        r"^A Prolegomenon by ",
        r"^Preface by ",
        r"^\[Transcriber['’ ]s Note:",
    ]

    for i, line in enumerate(lines[:20]):
        line_stripped = line.strip()
        if not line_stripped:
            continue

        is_metadata = False
        for pattern in metadata_patterns:
            if re.match(pattern, line_stripped, re.IGNORECASE):
                is_metadata = True
                break

        if re.match(r"^©|^Copyright|^\[\d{4}\]|^\d{4}\.", line_stripped, re.IGNORECASE):
            is_metadata = True

        if not is_metadata:
            return "\n".join(lines[i:])

    return text


def is_gutenberg_text(text: str) -> bool:
    patterns = [
        r"Project Gutenberg",
        r"www\.gutenberg\.org",
        r"\*\*\* START OF (?:THE |THIS )?PROJECT GUTENBERG",
        r"End of (?:the )?Project Gutenberg",
    ]

    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def clean_gutenberg_in_builder(original_text: str, url: str = "", tid: str = "") -> str:
    if url and ("gutenberg.org" in url or "gutenberg" in url.lower()):
        logger.debug(f"{tid}: Project Gutenberg URL detected, applying cleanup")
        return clean_gutenberg_text(original_text, tid or url)

    if is_gutenberg_text(original_text):
        logger.debug(f"{tid}: Project Gutenberg text detected, applying cleanup")
        return clean_gutenberg_text(original_text, tid or "unknown file")

    return original_text
