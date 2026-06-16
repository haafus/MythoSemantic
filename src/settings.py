from pathlib import Path

from pydantic import BaseModel, Field
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
    batch_size: int | None = None
    chroma_batch_size: int = 100
    max_workers: int = 16
    queue_maxsize: int = 10
    models: list[str] = [
        "BAAI/bge-m3",
        "sentence-transformers/LaBSE",
        "intfloat/e5-large-v2",
        "Qwen/Qwen3-Embedding-0.6B",
        "Qwen/Qwen3-Embedding-4B",
        "uhhlt/story-emb",
    ]


class LLMSettings(BaseModel):
    model_name: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.1
    max_retries: int = 5
    retry_backoff_factor: float = 5.0


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
    cache_max_age: int = 86400
    search_job_ttl_seconds: int = 1800
    search_max_workers: int = 1


# ---------------------------------------------------------------------------
# Root settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    corpus_dir: Path = Path("outputs/corpus")
    chroma_dir: Path = Path("outputs/chroma_db")
    analysis_dir: Path = Path("outputs/analysis")
    logs_dir: Path = Path("outputs/logs")
    graphs_dir: Path = Path("outputs/graphs")
    corpus_config_file: Path = Path("config/corpus.json")
    traditions_config_file: Path = Path("config/traditions.json")

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

    @property
    def corpus_metadata_path(self) -> Path:
        return self.corpus_dir / "corpus.json"

    @property
    def server_dir(self) -> Path:
        return self.project_root / "src" / "server"

    @property
    def web_root(self) -> Path:
        return self.server_dir / "web"

    @property
    def assets_dir(self) -> Path:
        return self.web_root / "assets"

    @staticmethod
    def safe_model_name(model_name: str) -> str:
        return model_name.replace("/", "_").replace("\\", "_")

    def model_output_dir(self, model_name: str) -> Path:
        return self.analysis_dir / self.safe_model_name(model_name)

    def ensure_dirs(self) -> None:
        for d in (
            self.corpus_dir,
            self.analysis_dir,
            self.logs_dir,
            self.graphs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
