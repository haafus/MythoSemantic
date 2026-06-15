from fastapi import APIRouter

from server.schemas import TraditionsResponse
from server.services.corpus import get_catalog_documents, get_traditions_info

router = APIRouter(prefix="/api/geography", tags=["geography"])


@router.get("/traditions", response_model=TraditionsResponse)
def traditions() -> dict:
    data = get_traditions_info()
    books_by_tradition: dict[str, list[str]] = {}
    for doc in get_catalog_documents():
        trad = doc.get("tradition", "")
        if trad:
            books_by_tradition.setdefault(trad, []).append(doc.get("id", ""))
    for trad, info in data.items():
        info["books"] = sorted(books_by_tradition.get(trad, []))
    return {"traditions": data, "total": len(data)}
