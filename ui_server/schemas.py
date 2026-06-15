from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ModelSummary(BaseModel):
    name: str
    key: str
    safe_dir: str


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    model: str
    top_k: int = Field(default=20, ge=1, le=100)


class SearchWarmupRequest(BaseModel):
    model: str


class SearchResult(BaseModel):
    id: str
    tradition: str
    chunk_index: int
    similarity_score: float
    distance: float
    text_preview: str


class PointInfo(BaseModel):
    id: str
    text: str
    tradition: str
    chunk_index: int
    model: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Neighbor(BaseModel):
    id: str
    tradition: str
    chunk_index: int
    similarity_score: float
    distance: float
    text_preview: str


class CorpusDocument(BaseModel):
    id: str
    major_tradition: str
    tradition: str
    language: str = ""
    type: str = ""
    url: str = ""
    word_count: int = 0
    sentence_count: int = 0
    char_count: int = 0
    color: str = "#6b7280"
    description: str = ""


class ProjectionResponse(BaseModel):
    model: str
    method: str
    points: List[Dict[str, Any]]


class SavedPlotResponse(BaseModel):
    exists: bool
    url: Optional[str] = None
    path: Optional[str] = None
    reason: Optional[str] = None
