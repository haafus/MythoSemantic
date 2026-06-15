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

from corpus.builder import _build_failure_metadata, _build_metadata, _item_tid, _update_traditions_info


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
        meta = _build_metadata(item, path="/tmp/x.txt", color="#000", stats=stats)
        parsed = datetime.fromisoformat(meta["date_downloaded"])
        assert parsed.tzinfo is not None


class TestBuildMetadataFields:
    def test_available_is_true(self):
        stats = {"md5": "abc", "char_count": 10, "word_count": 500, "sentence_count": 40}
        item = {**_BASE_ITEM, "title": "Iliad"}
        meta = _build_metadata(item, path="/tmp/x.txt", color="#FF0000", stats=stats)
        assert meta["available"] is True
        assert meta["word_count"] == 500

    def test_description_from_item(self):
        stats = {"md5": "abc", "char_count": 10, "word_count": 10, "sentence_count": 1}
        item = {**_BASE_ITEM, "title": "Iliad", "description": "An epic poem"}
        meta = _build_metadata(item, path="/tmp/x.txt", color="#000", stats=stats)
        assert meta["description"] == "An epic poem"

    def test_empty_description(self):
        stats = {"md5": "abc", "char_count": 10, "word_count": 10, "sentence_count": 1}
        item = {**_BASE_ITEM, "title": "Iliad"}
        meta = _build_metadata(item, path="/tmp/x.txt", color="#000", stats=stats)
        assert meta["description"] == ""

    def test_missing_major_tradition_defaults(self):
        stats = {"md5": "abc", "char_count": 10, "word_count": 1, "sentence_count": 1}
        item = {"tradition": "T", "url": "http://example.com/no-major"}
        meta = _build_metadata(item, path="/tmp/x.txt", color="#000", stats=stats)
        assert meta["major_tradition"] == "Unknown"


class TestBuildFailureMetadata:
    def test_failure_has_zero_counts(self):
        meta = _build_failure_metadata(_BASE_ITEM, color="#FF0000", error="Timeout")
        assert meta["available"] is False
        assert meta["word_count"] == 0
        assert meta["sentence_count"] == 0
        assert "Timeout" in meta["description"]

    def test_failure_with_description_includes_both(self):
        item = {**_BASE_ITEM, "description": "The Odyssey"}
        meta = _build_failure_metadata(item, color="#FF0000", error="404")
        assert "The Odyssey" in meta["description"]
        assert "404" in meta["description"]


class TestUpdateTraditionsInfo:
    def _setup(self, tmp_path, monkeypatch, items):
        from settings import settings

        dl_file = tmp_path / "corpus.json"
        dl_file.write_text(json.dumps(items))
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()

        monkeypatch.setattr(settings, "corpus_config_file", dl_file)
        monkeypatch.setattr(settings, "corpus_dir", corpus_dir)
        return corpus_dir

    def test_creates_new_file(self, tmp_path, monkeypatch):
        items = [
            {"title": "Iliad", "tradition": "Greek"},
            {"title": "Odyssey", "tradition": "Greek"},
            {"title": "Edda", "tradition": "Norse"},
        ]
        corpus_dir = self._setup(tmp_path, monkeypatch, items)

        _update_traditions_info(force=False)

        data = json.loads((corpus_dir / "traditions_info.json").read_text())
        assert "Greek" in data
        assert "Norse" in data
        assert sorted(data["Greek"]["books"]) == ["Iliad", "Odyssey"]
        assert data["Norse"]["books"] == ["Edda"]
        assert data["Greek"]["description"] == ""
        assert data["Greek"]["color"].startswith("#")

    def test_preserves_existing_descriptions(self, tmp_path, monkeypatch):
        items = [
            {"title": "Iliad", "tradition": "Greek"},
            {"title": "Odyssey", "tradition": "Greek"},
        ]
        corpus_dir = self._setup(tmp_path, monkeypatch, items)

        existing = {
            "Greek": {
                "description": "Ancient Greek mythology",
                "coordinates": [37.9, 23.7],
                "color": "#123456",
                "books": ["Iliad"],
            }
        }
        (corpus_dir / "traditions_info.json").write_text(json.dumps(existing))

        _update_traditions_info(force=False)

        data = json.loads((corpus_dir / "traditions_info.json").read_text())
        assert data["Greek"]["description"] == "Ancient Greek mythology"
        assert sorted(data["Greek"]["books"]) == ["Iliad", "Odyssey"]

    def test_force_creates_backup(self, tmp_path, monkeypatch):
        corpus_dir = self._setup(tmp_path, monkeypatch, [])

        existing = {"Greek": {"description": "old data", "color": "#000", "books": []}}
        (corpus_dir / "traditions_info.json").write_text(json.dumps(existing))

        _update_traditions_info(force=True)

        backup = json.loads((corpus_dir / "traditions_info_backup.json").read_text())
        assert backup["Greek"]["description"] == "old data"

    def test_includes_all_traditions(self, tmp_path, monkeypatch):
        items = [
            {"title": "Iliad", "tradition": "Greek"},
            {"title": "Edda", "tradition": "Norse"},
        ]
        corpus_dir = self._setup(tmp_path, monkeypatch, items)

        _update_traditions_info(force=False)

        data = json.loads((corpus_dir / "traditions_info.json").read_text())
        assert "Greek" in data
        assert "Norse" in data

    def test_no_download_list(self, tmp_path, monkeypatch):
        from settings import settings

        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        monkeypatch.setattr(settings, "corpus_config_file", tmp_path / "nonexistent.json")
        monkeypatch.setattr(settings, "corpus_dir", corpus_dir)

        _update_traditions_info(force=False)

        data = json.loads((corpus_dir / "traditions_info.json").read_text())
        assert data == {}

    def test_adds_missing_color(self, tmp_path, monkeypatch):
        items = [{"title": "Edda", "tradition": "Norse"}]
        corpus_dir = self._setup(tmp_path, monkeypatch, items)

        existing = {"Norse": {"description": "Norse myths", "books": ["Edda"]}}
        (corpus_dir / "traditions_info.json").write_text(json.dumps(existing))

        _update_traditions_info(force=False)

        data = json.loads((corpus_dir / "traditions_info.json").read_text())
        assert data["Norse"]["color"].startswith("#")
        assert data["Norse"]["description"] == "Norse myths"

    def test_uses_item_tid_for_books(self, tmp_path, monkeypatch):
        items = [
            {"id": "book_42", "tradition": "Egyptian"},
        ]
        corpus_dir = self._setup(tmp_path, monkeypatch, items)

        _update_traditions_info(force=False)

        data = json.loads((corpus_dir / "traditions_info.json").read_text())
        assert data["Egyptian"]["books"] == ["book_42"]
