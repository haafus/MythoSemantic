from fastapi import APIRouter

from corpus.utils import read_traditions
from server.schemas import TraditionsResponse
from server.services.corpus import get_catalog_documents
from settings import settings

router = APIRouter(prefix="/api/geography", tags=["geography"])


@router.get("/traditions", response_model=TraditionsResponse)
def traditions() -> dict:
    data = read_traditions(settings.corpus_dir)
    books_by_tradition: dict[str, list[str]] = {}
    for doc in get_catalog_documents():
        trad = doc.get("tradition", "")
        if trad:
            books_by_tradition.setdefault(trad, []).append(doc.get("id", ""))
    for trad, info in data.items():
        info["books"] = sorted(books_by_tradition.get(trad, []))
    return {"traditions": data, "total": len(data)}
