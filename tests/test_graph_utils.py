import importlib.util
import os
import sys
import types

_parent = os.path.join(os.path.dirname(__file__), "..", "src", "graphs")

_stubs_added: list[str] = []
for stub_name in ["networkx", "openai"]:
    if stub_name not in sys.modules:
        sys.modules[stub_name] = types.ModuleType(stub_name)
        _stubs_added.append(stub_name)

_llm_stub = types.ModuleType("llm_client")
_llm_stub.LLMProcessor = type("LLMProcessor", (), {})  # type: ignore[attr-defined]
sys.modules["llm_client"] = _llm_stub


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_parent, filename))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_chunking = _load_module("graphs.chunking", "chunking.py")
_extraction = _load_module("graphs.extraction", "extraction.py")
_checkpointing = _load_module("graphs.checkpointing", "checkpointing.py")

chunk_text = _chunking.chunk_text
deduplicate_entities = _extraction.deduplicate_entities
deduplicate_relations = _extraction.deduplicate_relations
extract_from_chunk = _extraction.extract_from_chunk
load_checkpoint = _checkpointing.load_checkpoint
save_checkpoint = _checkpointing.save_checkpoint
clear_checkpoint = _checkpointing.clear_checkpoint


class TestChunkText:
    def test_short_text_single_chunk(self):
        chunks = chunk_text("Hello world.", max_chars=1000, overlap=100)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world."

    def test_empty_text(self):
        assert chunk_text("", max_chars=100, overlap=10) == []

    def test_splits_long_text(self):
        text = "A" * 500 + " " + "B" * 500
        chunks = chunk_text(text, max_chars=300, overlap=50)
        assert len(chunks) > 1

    def test_all_text_covered(self):
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        chunks = chunk_text(text, max_chars=30, overlap=5)
        combined = " ".join(chunks)
        for word in ["one", "two", "three", "four", "five"]:
            assert word in combined


class TestDeduplicateEntities:
    def test_empty_list(self):
        assert deduplicate_entities([]) == []

    def test_no_duplicates(self):
        entities = [{"name": "Zeus"}, {"name": "Hera"}]
        result = deduplicate_entities(entities)
        assert len(result) == 2

    def test_removes_duplicates(self):
        entities = [{"name": "Zeus"}, {"name": "zeus"}, {"name": "ZEUS"}]
        result = deduplicate_entities(entities)
        assert len(result) == 1

    def test_merges_properties(self):
        entities = [
            {"name": "Zeus", "type": "god"},
            {"name": "zeus", "description": "king of gods"},
        ]
        result = deduplicate_entities(entities)
        assert len(result) == 1


class TestDeduplicateRelations:
    def test_empty_list(self):
        assert deduplicate_relations([]) == []

    def test_no_duplicates(self):
        rels = [
            {"subject": "Zeus", "object": "Hera", "relation": "married"},
            {"subject": "Zeus", "object": "Athena", "relation": "father"},
        ]
        result = deduplicate_relations(rels)
        assert len(result) == 2

    def test_removes_duplicates(self):
        rels = [
            {"subject": "Zeus", "object": "Hera", "relation": "married"},
            {"subject": "zeus", "object": "hera", "relation": "married"},
        ]
        result = deduplicate_relations(rels)
        assert len(result) == 1


class TestCheckpoint:
    def test_roundtrip(self, tmp_path):
        results = {"beings": [{"Name": "Zeus"}], "relations": [], "locations": [], "times": []}
        save_checkpoint(tmp_path, 3, results)

        cp = load_checkpoint(tmp_path)
        assert cp["next_chunk"] == 3
        assert cp["beings"] == [{"Name": "Zeus"}]

    def test_missing_returns_none(self, tmp_path):
        assert load_checkpoint(tmp_path) is None

    def test_corrupt_returns_none(self, tmp_path):
        (tmp_path / "checkpoint.json").write_text("{not json")
        assert load_checkpoint(tmp_path) is None

    def test_missing_next_chunk_returns_none(self, tmp_path):
        (tmp_path / "checkpoint.json").write_text('{"characters": []}')
        assert load_checkpoint(tmp_path) is None

    def test_clear_is_idempotent(self, tmp_path):
        save_checkpoint(tmp_path, 1, {})
        clear_checkpoint(tmp_path)
        assert load_checkpoint(tmp_path) is None
        clear_checkpoint(tmp_path)


class _FakeLLM:
    def ask_json(self, system_prompt, user_content):
        if system_prompt == "c":
            return [{"Name": "Zeus"}]
        if system_prompt == "r":
            return [{"Subject": "Zeus", "Object": "Hera", "Relation": "spouse"}]
        if system_prompt == "l":
            return [{"Name": "Olympus"}]
        if system_prompt == "t":
            return "not-a-list"
        return []


class TestExtractFromChunk:
    def test_collects_all_entity_types(self):
        prompts = {"beings": "c", "relations": "r", "locations": "l", "time": "t"}
        out = extract_from_chunk(_FakeLLM(), "some text", prompts)

        assert out["beings"] == [{"Name": "Zeus"}]
        assert out["relations"][0]["Subject"] == "Zeus"
        assert out["locations"] == [{"Name": "Olympus"}]
        assert out["times"] == []
