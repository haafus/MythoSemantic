from typing import Any

from pydantic import BaseModel, Field


class ModelSummary(BaseModel):
    name: str
    key: str


class ModelListResponse(BaseModel):
    models: list[ModelSummary]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    model: str
    top_k: int = Field(default=20, ge=1, le=100)


class WarmupRequest(BaseModel):
    model: str


class SearchResult(BaseModel):
    id: str
    tradition: str = "Unknown"
    major_tradition: str = ""
    chunk_index: int = 0
    similarity_score: float
    text: str = ""
    filename: str = ""


class SearchResponse(BaseModel):
    query: str
    model: str
    results: list[SearchResult]
    total: int


class PointInfo(BaseModel):
    id: str
    text: str = ""
    tradition: str = "Unknown"
    chunk_index: int = 0
    model: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class NeighborsResponse(BaseModel):
    point_id: str
    neighbors: list[SearchResult]


class CorpusDocument(BaseModel):
    id: str
    major_tradition: str = ""
    tradition: str = ""
    url: str = ""
    word_count: int = 0
    sentence_count: int = 0
    char_count: int = 0
    color: str = "#6b7280"
    description: str = ""
    source: str = ""


class CatalogResponse(BaseModel):
    documents: list[CorpusDocument]
    total: int


class TraditionsResponse(BaseModel):
    traditions: dict[str, Any]
    total: int
