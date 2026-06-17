from corpus.clean_gutenberg import (
    _normalize_gutenberg_whitespace,
    _remove_header_metadata,
    clean_gutenberg_text,
    is_gutenberg_text,
)


class TestIsGutenbergText:
    def test_detects_start_marker(self):
        text = "*** START OF THE PROJECT GUTENBERG EBOOK BIBLE ***\nActual content."
        assert is_gutenberg_text(text) is True

    def test_detects_end_marker(self):
        text = "Some text.\nEnd of the Project Gutenberg EBook of something"
        assert is_gutenberg_text(text) is True

    def test_detects_url(self):
        text = "Visit www.gutenberg.org for more."
        assert is_gutenberg_text(text) is True

    def test_rejects_plain_text(self):
        text = "In the beginning God created the heaven and the earth."
        assert is_gutenberg_text(text) is False

    def test_empty_string(self):
        assert is_gutenberg_text("") is False


class TestCleanGutenbergText:
    def test_strips_header_and_footer(self):
        text = (
            "Header stuff\n"
            "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
            "\n"
            "Actual book content here.\n"
            "More content.\n"
            "\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
            "Footer license stuff"
        )
        result = clean_gutenberg_text(text)
        assert "Actual book content here." in result
        assert "More content." in result
        assert "Header stuff" not in result
        assert "Footer license stuff" not in result
        assert "PROJECT GUTENBERG" not in result

    def test_strips_this_project_variant(self):
        text = (
            "Preamble\n"
            "*** START OF THIS PROJECT GUTENBERG EBOOK FOO ***\n\n"
            "The real text.\n\n"
            "*** END OF THIS PROJECT GUTENBERG EBOOK FOO ***\n"
            "Postamble"
        )
        result = clean_gutenberg_text(text)
        assert "The real text." in result
        assert "Preamble" not in result

    def test_returns_original_if_no_markers(self):
        text = "Just plain text with no gutenberg markers."
        result = clean_gutenberg_text(text)
        assert result == text

    def test_empty_input(self):
        assert clean_gutenberg_text("") == ""
        assert clean_gutenberg_text(None) is None  # type: ignore[arg-type]

    def test_preserves_content_between_markers(self):
        content_lines = [f"Line {i} of the book." for i in range(50)]
        content = "\n".join(content_lines)
        text = (
            "*** START OF THE PROJECT GUTENBERG EBOOK X ***\n\n"
            f"{content}\n\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK X ***"
        )
        result = clean_gutenberg_text(text)
        for line in content_lines:
            assert line in result

    def test_only_start_marker(self):
        text = "License header\n*** START OF THE PROJECT GUTENBERG EBOOK Y ***\n\nContent without end marker."
        result = clean_gutenberg_text(text)
        assert "Content without end marker." in result
        assert "License header" not in result

    def test_only_end_marker(self):
        text = "Content before end marker.\n\n*** END OF THE PROJECT GUTENBERG EBOOK Z ***\nLicense footer"
        result = clean_gutenberg_text(text)
        assert "Content before end marker." in result
        assert "License footer" not in result


class TestNormalizeWhitespace:
    def test_removes_decorative_lines(self):
        text = "Before.\n**********\nAfter."
        result = _normalize_gutenberg_whitespace(text)
        assert "****" not in result
        assert "Before." in result
        assert "After." in result

    def test_strips_trailing_whitespace(self):
        text = "Line with trailing spaces   \nNext line"
        result = _normalize_gutenberg_whitespace(text)
        assert "   \n" not in result


class TestRemoveHeaderMetadata:
    def test_removes_translated_by(self):
        text = "Translated by John Smith\n\nActual content."
        result = _remove_header_metadata(text)
        assert "Translated by" not in result
        assert "Actual content." in result

    def test_removes_copyright(self):
        text = "Copyright 1922\n\nThe story begins."
        result = _remove_header_metadata(text)
        assert "Copyright" not in result
        assert "The story begins." in result

    def test_preserves_normal_text(self):
        text = "In the beginning was the Word.\nAnd the Word was with God."
        result = _remove_header_metadata(text)
        assert "In the beginning" in result
