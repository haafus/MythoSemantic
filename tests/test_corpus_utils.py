import importlib.util
import os
import sys
import types

for stub in ["pymupdf", "trafilatura", "bs4", "fake_useragent"]:
    sys.modules.setdefault(stub, types.ModuleType(stub))
bs4_mod = sys.modules["bs4"]
if not hasattr(bs4_mod, "BeautifulSoup"):
    bs4_mod.BeautifulSoup = type("BeautifulSoup", (), {})  # type: ignore[attr-defined]

sys.modules.setdefault("corpus", types.ModuleType("corpus"))

_spec = importlib.util.spec_from_file_location(
    "corpus.utils",
    os.path.join(os.path.dirname(__file__), "..", "src", "corpus", "utils.py"),
)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

sanitize_filename = _mod.sanitize_filename
corpus_text_path = _mod.corpus_text_path
md5 = _mod.md5
normalize_text = _mod.normalize_text
count_words = _mod.count_words
count_sentences = _mod.count_sentences
ensure_dir = _mod.ensure_dir
get_tradition_color = _mod.get_tradition_color


class TestSanitizeFilename:
    def test_removes_illegal_chars(self):
        assert sanitize_filename('test:file*name?.txt') == "test_file_name_.txt"

    def test_preserves_normal_filename(self):
        assert sanitize_filename("hello_world.txt") == "hello_world.txt"

    def test_strips_whitespace(self):
        assert sanitize_filename("  test.txt  ") == "test.txt"

    def test_replaces_all_special_chars(self):
        result = sanitize_filename('a\\b/c*d?e"f<g>h|i')
        assert "\\" not in result
        assert "/" not in result
        assert "*" not in result
        assert "?" not in result
        assert '"' not in result
        assert "<" not in result
        assert ">" not in result
        assert "|" not in result

    def test_blocks_path_traversal(self):
        assert ".." not in sanitize_filename("..")
        assert ".." not in sanitize_filename("../etc/passwd")
        assert ".." not in sanitize_filename("foo/../bar")


class TestCorpusTextPath:
    def test_canonical_layout(self, tmp_path):
        path = corpus_text_path(tmp_path, "European", "Greek", "Iliad")
        assert path == tmp_path / "European" / "Greek" / "Iliad.txt"

    def test_spaces_and_slashes_normalized(self, tmp_path):
        path = corpus_text_path(tmp_path, "Near East", "Sumer/Akkad", "Epic of Gilgamesh")
        assert path.name == "Epic_of_Gilgamesh.txt"
        assert "Near_East" in path.parts
        assert "Sumer_Akkad" in path.parts

    def test_traversal_blocked(self, tmp_path):
        path = corpus_text_path(tmp_path, "..", "..", "../../etc/passwd")
        assert tmp_path in path.parents
        assert ".." not in path.parts


class TestMd5:
    def test_known_hash(self):
        assert md5(b"hello") == "5d41402abc4b2a76b9719d911017c592"

    def test_empty_bytes(self):
        assert md5(b"") == "d41d8cd98f00b204e9800998ecf8427e"

    def test_deterministic(self):
        assert md5(b"test") == md5(b"test")

    def test_different_inputs_differ(self):
        assert md5(b"a") != md5(b"b")


class TestNormalizeText:
    def test_strips_whitespace(self):
        assert normalize_text("  hello  ") == "hello"

    def test_collapses_multiple_spaces(self):
        result = normalize_text("hello   world")
        assert "   " not in result

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_preserves_normal_text(self):
        text = "This is a normal sentence."
        assert normalize_text(text) == text


class TestCountWords:
    def test_simple_sentence(self):
        assert count_words("hello world") == 2

    def test_empty_string(self):
        assert count_words("") == 0

    def test_multiple_spaces(self):
        assert count_words("one   two   three") == 3

    def test_single_word(self):
        assert count_words("hello") == 1


class TestCountSentences:
    def test_single_sentence(self):
        assert count_sentences("Hello world.") >= 1

    def test_multiple_sentences(self):
        assert count_sentences("First. Second. Third.") >= 2

    def test_empty_string(self):
        assert count_sentences("") == 0

    def test_exclamation_and_question(self):
        assert count_sentences("Hello! How are you? Fine.") >= 2


class TestEnsureDir:
    def test_creates_directory(self, tmp_path):
        new_dir = tmp_path / "new" / "nested" / "dir"
        assert not new_dir.exists()
        ensure_dir(new_dir)
        assert new_dir.exists()

    def test_existing_directory_no_error(self, tmp_path):
        ensure_dir(tmp_path)
        ensure_dir(tmp_path)


class TestGetTraditionColor:
    def test_returns_hex_color(self):
        color = get_tradition_color("test_tradition_unique_1")
        assert color.startswith("#")
        assert len(color) == 7

    def test_same_tradition_same_color(self):
        c1 = get_tradition_color("test_tradition_unique_2")
        c2 = get_tradition_color("test_tradition_unique_2")
        assert c1 == c2

    def test_different_traditions_different_colors(self):
        c1 = get_tradition_color("test_tradition_unique_3")
        c2 = get_tradition_color("test_tradition_unique_4")
        assert c1 != c2
