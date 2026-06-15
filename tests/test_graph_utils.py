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

_graph_gen_stub = types.ModuleType("graphs.graph_generator")
_graph_gen_stub.generate_and_save_graph = lambda *a, **kw: None  # type: ignore[attr-defined]
sys.modules["graphs.graph_generator"] = _graph_gen_stub

_llm_stub = types.ModuleType("llm_processing")
_llm_stub.LLMProcessor = type("LLMProcessor", (), {})  # type: ignore[attr-defined]
sys.modules["llm_processing"] = _llm_stub

_spec = importlib.util.spec_from_file_location(
    "graphs.run_graph_generation",
    os.path.join(_parent, "run_graph_generation.py"),
)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules["graphs.run_graph_generation"] = _mod
_spec.loader.exec_module(_mod)

chunk_text = _mod.chunk_text
deduplicate_entities = _mod.deduplicate_entities
deduplicate_relations = _mod.deduplicate_relations


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


load_checkpoint = _mod.load_checkpoint
save_checkpoint = _mod.save_checkpoint
clear_checkpoint = _mod.clear_checkpoint
extract_from_chunk = _mod.extract_from_chunk


class TestCheckpoint:
    def test_roundtrip(self, tmp_path):
        results = {"characters": [{"Name": "Zeus"}], "relations": [], "locations": [], "times": []}
        save_checkpoint(tmp_path, 3, results)

        cp = load_checkpoint(tmp_path)
        assert cp["next_chunk"] == 3
        assert cp["characters"] == [{"Name": "Zeus"}]

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
    def extract_characters(self, text, prompt):
        return [{"Name": "Zeus"}]

    def extract_relations(self, text, characters, prompt):
        assert characters == [{"Name": "Zeus"}], "relations must receive extracted characters"
        return [{"Subject": "Zeus", "Object": "Hera", "Relation": "spouse"}]

    def extract_locations(self, text, prompt):
        return [{"Name": "Olympus"}]

    def extract_time(self, text, prompt):
        return "not-a-list"


class TestExtractFromChunk:
    def test_collects_all_entity_types(self):
        prompts = {"characters": "c", "relations": "r", "locations": "l", "time": "t"}
        out = extract_from_chunk(_FakeLLM(), "some text", prompts)

        assert out["characters"] == [{"Name": "Zeus"}]
        assert out["relations"][0]["Subject"] == "Zeus"
        assert out["locations"] == [{"Name": "Olympus"}]
        assert out["times"] == []
