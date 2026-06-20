from fastapi import APIRouter, HTTPException

from server.services.graphs import get_graph_data, list_books_with_graphs

router = APIRouter(prefix="/api/graphs", tags=["graphs"])


@router.get("/")
def list_graphs() -> dict:
    books = list_books_with_graphs()
    return {"books": books, "total": len(books)}


@router.get("/{book_id}/{graph_type}")
def graph(book_id: str, graph_type: str) -> dict:
    data = get_graph_data(book_id, graph_type)
    if not data:
        raise HTTPException(status_code=404, detail="Graph data not found")
    return data
