from fastapi import APIRouter

from server.schemas import SearchRequest, SearchResult
from server.services.embedding_index import embedding_index_service

router = APIRouter(prefix="/api/similarity", tags=["search"])


@router.post("/search", response_model=list[SearchResult])
def search(request: SearchRequest):
    return embedding_index_service.search(request.model, request.query, request.top_k)
