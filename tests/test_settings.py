from pathlib import Path


def test_default_paths():
    from settings import Settings

    s = Settings()
    assert s.corpus_dir == Path("outputs/corpus")
    assert s.chroma_dir == Path("outputs/chroma_db")
    assert s.analysis_dir == Path("outputs/analysis")
    assert s.logs_dir == Path("outputs/logs")


def test_derived_paths():
    from settings import Settings

    s = Settings()
    assert s.corpus_metadata_path == Path("outputs/corpus/corpus.json")


def test_model_output_dir():
    from settings import Settings

    s = Settings()
    assert s.model_output_dir("BAAI/bge-m3") == Path("outputs/analysis/BAAI_bge-m3")
    assert s.model_output_dir("sentence-transformers/LaBSE") == Path("outputs/analysis/sentence-transformers_LaBSE")


def test_env_override(monkeypatch):
    monkeypatch.setenv("MYTHO_CORPUS_DIR", "/tmp/my_corpus")
    monkeypatch.setenv("MYTHO_LOG_LEVEL", "DEBUG")

    from settings import Settings

    s = Settings()
    assert s.corpus_dir == Path("/tmp/my_corpus")
    assert s.log_level == "DEBUG"
    assert s.corpus_metadata_path == Path("/tmp/my_corpus/corpus.json")


def test_env_override_chroma(monkeypatch):
    monkeypatch.setenv("MYTHO_CHROMA_DIR", "/data/chroma")

    from settings import Settings

    s = Settings()
    assert s.chroma_dir == Path("/data/chroma")


def test_default_embedding_model():
    from settings import Settings

    s = Settings()
    assert s.embedding.models[0] == "bge-m3"
    assert s.embedding.default_chunking == "paragraph"
