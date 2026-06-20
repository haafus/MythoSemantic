import json
import sys
import types

for stub in ["pymupdf", "trafilatura", "bs4", "fake_useragent"]:
    sys.modules.setdefault(stub, types.ModuleType(stub))
bs4_mod = sys.modules["bs4"]
if not hasattr(bs4_mod, "BeautifulSoup"):
    bs4_mod.BeautifulSoup = type("BeautifulSoup", (), {})  # type: ignore[attr-defined]
fu_mod = sys.modules["fake_useragent"]
if not hasattr(fu_mod, "UserAgent"):

    class _FakeUA:
        def __init__(self, **_kw):
            pass

        random = "test-agent"

    fu_mod.UserAgent = _FakeUA  # type: ignore[attr-defined]

from datetime import datetime

from corpus.builder import _build_metadata, _item_tid, _update_traditions


class TestItemTid:
    def test_prefers_title(self):
        assert _item_tid({"title": "Iliad", "id": "123"}) == "Iliad"

    def test_falls_back_to_id(self):
        assert _item_tid({"id": "abc"}) == "abc"

    def test_falls_back_to_unknown(self):
        assert _item_tid({}) == "unknown_id"

    def test_empty_title_still_returned(self):
        assert _item_tid({"title": "", "id": "x"}) == ""


_BASE_ITEM = {
    "major_tradition": "Greek",
    "tradition": "Hellenic",
    "url": "http://example.com/text",
}


class TestBuildMetadata:
    def test_date_downloaded_is_timezone_aware(self):
        stats = {"md5": "abc", "char_count": 10, "word_count": 2, "sentence_count": 1}
        item = {**_BASE_ITEM, "title": "Iliad"}
        meta = _build_metadata(item, path="/tmp/x.txt", stats=stats)
        parsed = datetime.fromisoformat(meta["date_downloaded"])
        assert parsed.tzinfo is not None


class TestBuildMetadataFields:
    def test_word_count(self):
        stats = {"md5": "abc", "char_count": 10, "word_count": 500, "sentence_count": 40}
        item = {**_BASE_ITEM, "title": "Iliad"}
        meta = _build_metadata(item, path="/tmp/x.txt", stats=stats)
        assert meta["word_count"] == 500

    def test_description_from_item(self):
        stats = {"md5": "abc", "char_count": 10, "word_count": 10, "sentence_count": 1}
        item = {**_BASE_ITEM, "title": "Iliad", "description": "An epic poem"}
        meta = _build_metadata(item, path="/tmp/x.txt", stats=stats)
        assert meta["description"] == "An epic poem"

    def test_empty_description(self):
        stats = {"md5": "abc", "char_count": 10, "word_count": 10, "sentence_count": 1}
        item = {**_BASE_ITEM, "title": "Iliad"}
        meta = _build_metadata(item, path="/tmp/x.txt", stats=stats)
        assert meta["description"] == ""

    def test_missing_major_tradition_defaults(self):
        stats = {"md5": "abc", "char_count": 10, "word_count": 1, "sentence_count": 1}
        item = {"tradition": "T", "url": "http://example.com/no-major"}
        meta = _build_metadata(item, path="/tmp/x.txt", stats=stats)
        assert meta["major_tradition"] == "Unknown"



class TestUpdateTraditions:
    def _setup(self, tmp_path, monkeypatch, *, books=None, traditions=None):
        from settings import settings

        config_dir = tmp_path / "config"
        config_dir.mkdir()
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()

        (config_dir / "corpus.json").write_text(json.dumps(books or []))
        (config_dir / "traditions.json").write_text(json.dumps(traditions or {}))

        monkeypatch.setattr(settings, "config_dir", config_dir)
        monkeypatch.setattr(settings, "corpus_dir", corpus_dir)
        return corpus_dir

    def test_merges_config_with_color(self, tmp_path, monkeypatch):
        traditions = {"Greek": {"description": "Ancient Greek mythology", "coordinates": [37.9, 23.7]}}
        books = [{"title": "Iliad", "tradition": "Greek"}]
        corpus_dir = self._setup(tmp_path, monkeypatch, books=books, traditions=traditions)

        _update_traditions(force=False)

        data = json.loads((corpus_dir / "traditions.json").read_text())
        assert data["Greek"]["description"] == "Ancient Greek mythology"
        assert data["Greek"]["coordinates"] == [37.9, 23.7]
        assert data["Greek"]["color"].startswith("#")

    def test_creates_stub_for_unknown_tradition(self, tmp_path, monkeypatch):
        books = [{"title": "Edda", "tradition": "Norse"}]
        corpus_dir = self._setup(tmp_path, monkeypatch, books=books)

        _update_traditions(force=False)

        data = json.loads((corpus_dir / "traditions.json").read_text())
        assert "Norse" in data
        assert data["Norse"]["description"] == ""
        assert data["Norse"]["color"].startswith("#")

    def test_includes_traditions_from_both_sources(self, tmp_path, monkeypatch):
        traditions = {"Celtic": {"description": "Celtic myths", "coordinates": [53.1, -7.7]}}
        books = [{"title": "Edda", "tradition": "Norse"}]
        corpus_dir = self._setup(tmp_path, monkeypatch, books=books, traditions=traditions)

        _update_traditions(force=False)

        data = json.loads((corpus_dir / "traditions.json").read_text())
        assert "Celtic" in data
        assert "Norse" in data

    def test_no_sources(self, tmp_path, monkeypatch):
        from settings import settings

        config_dir = tmp_path / "empty_config"
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        monkeypatch.setattr(settings, "config_dir", config_dir)
        monkeypatch.setattr(settings, "corpus_dir", corpus_dir)

        _update_traditions(force=False)

        data = json.loads((corpus_dir / "traditions.json").read_text())
        assert data == {}
