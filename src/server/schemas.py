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
    url: str = ""


class SearchResponse(BaseModel):
    query: str
    model: str
    results: list[SearchResult]
    total: int


class CorpusDocument(BaseModel):
    title: str
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
