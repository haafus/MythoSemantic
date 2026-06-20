from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import BaseSettings

# ---------------------------------------------------------------------------
# Sub-models (BaseModel — not BaseSettings, nested inside Settings)
# ---------------------------------------------------------------------------


class CorpusSettings(BaseModel):
    max_workers: int = 10
    timeout_connect: int = 10
    timeout_read: int = 30
    retry_total: int = 4
    retry_backoff_factor: float = 1.5
    retry_status_forcelist: list[int] = [429, 500, 502, 503, 504]
    html_include_comments: bool = False
    html_include_tables: bool = True
    pdf_extract_tables: bool = False
    pdf_preserve_layout: bool = True


class EmbeddingSettings(BaseModel):
    default_chunking: str = "paragraph"
    batch_size: int = 32
    max_workers: int = 16


class LLMSettings(BaseModel):
    model: str = "gpt4o-mini"
    temperature: float = 0.1
    max_retries: int = 5


class GraphsSettings(BaseModel):
    use_json_mode: bool = True
    chunk_size: int = 4000
    chunk_overlap: int = 1000


class ProjectionSettings(BaseModel):
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1


class ServerSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    gzip_minimum_size: int = 1024
    search_job_ttl_seconds: int = 1800
    search_max_workers: int = 1


# ---------------------------------------------------------------------------
# Root settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    config_dir: Path = Path("config")
    corpus_dir: Path = Path("outputs/corpus")
    embeddings_dir: Path = Path("outputs/embeddings")
    projections_dir: Path = Path("outputs/projections")
    graphs_dir: Path = Path("outputs/graphs")
    logs_dir: Path = Path("outputs/logs")
    web_root: Path = Path("src/server/web")

    log_level: str = "INFO"

    # sub-settings
    corpus: CorpusSettings = CorpusSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    llm: LLMSettings = LLMSettings()
    graphs: GraphsSettings = GraphsSettings()
    projection: ProjectionSettings = ProjectionSettings()
    server: ServerSettings = ServerSettings()

    model_config = {
        "env_file": [".env", "config/.env"],
        "env_prefix": "MYTHO_",
        "env_nested_delimiter": "__",
        "extra": "ignore",
    }


settings = Settings()
