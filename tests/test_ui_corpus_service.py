from corpus.utils import sanitize_filename


class TestSanitizeFilename:
    def test_replaces_slashes(self):
        assert "/" not in sanitize_filename("a/b/c")
        assert "\\" not in sanitize_filename("a\\b\\c")

    def test_replaces_spaces(self):
        result = sanitize_filename("hello world")
        assert " " not in result

    def test_preserves_normal_text(self):
        assert sanitize_filename("simple") == "simple"

    def test_empty_string(self):
        result = sanitize_filename("")
        assert isinstance(result, str)
