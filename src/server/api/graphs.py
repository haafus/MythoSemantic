from fastapi import APIRouter, HTTPException

from server.services.graphs import get_graph_data

router = APIRouter(prefix="/api/graphs", tags=["graphs"])


@router.get("/{book_id}/{graph_type}")
def graph(book_id: str, graph_type: str) -> dict:
    data = get_graph_data(book_id, graph_type)
    if not data:
        raise HTTPException(status_code=404, detail="Graph data not found")
    return data
