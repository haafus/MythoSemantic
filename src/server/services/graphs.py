import json
from pathlib import Path

from corpus.utils import normalize_catalog_id
from settings import settings

GRAPH_TYPES = {"characters", "realms", "ages"}


def list_books_with_graphs() -> list[dict]:
    if not settings.graphs_dir.exists():
        return []
    books = []
    for book_dir in sorted(settings.graphs_dir.iterdir()):
        if not book_dir.is_dir():
            continue
        available = [t for t in GRAPH_TYPES if (book_dir / f"{t}.json").exists()]
        if available:
            books.append({"id": book_dir.name, "graphs": available})
    return books


def get_graph_data(book_id: str, graph_type: str) -> dict | None:
    if graph_type not in GRAPH_TYPES:
        return None
    book_dir = settings.graphs_dir / normalize_catalog_id(book_id)
    json_path = book_dir / f"{graph_type}.json"
    if not json_path.exists():
        return None
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)
