from pathlib import Path


def test_default_paths():
    from settings import Settings

    s = Settings()
    assert s.corpus_dir == Path("outputs/corpus")
    assert s.embeddings_dir == Path("outputs/embeddings")
    assert s.projections_dir == Path("outputs/projections")
    assert s.logs_dir == Path("outputs/logs")


def test_default_metadata_path():
    from settings import Settings

    s = Settings()
    assert s.corpus_metadata_path == Path("outputs/corpus/corpus.json")



def test_env_override(monkeypatch):
    monkeypatch.setenv("MYTHO_CORPUS_DIR", "/tmp/my_corpus")
    monkeypatch.setenv("MYTHO_LOG_LEVEL", "DEBUG")

    from settings import Settings

    s = Settings()
    assert s.corpus_dir == Path("/tmp/my_corpus")
    assert s.log_level == "DEBUG"
    assert s.corpus_metadata_path == Path("outputs/corpus/corpus.json")


def test_env_override_embeddings_dir(monkeypatch):
    monkeypatch.setenv("MYTHO_EMBEDDINGS_DIR", "/data/embeddings")

    from settings import Settings

    s = Settings()
    assert s.embeddings_dir == Path("/data/embeddings")


def test_default_embedding_settings():
    from settings import Settings

    s = Settings()
    assert s.embedding.default_chunking == "paragraph"


def test_active_embedding_models():
    from model_registry import active_embedding_models

    models = active_embedding_models()
    assert len(models) == 4
    assert models[0] == "BAAI/bge-m3"
