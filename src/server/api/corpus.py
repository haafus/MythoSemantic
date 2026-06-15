from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse, StreamingResponse

from server.schemas import CatalogResponse
from server.services.corpus import build_corpus_archive, get_catalog_documents, read_document

router = APIRouter(prefix="/api/corpus", tags=["corpus"])


@router.get("/catalog", response_model=CatalogResponse)
def catalog():
    documents = get_catalog_documents()
    return {"documents": documents, "total": len(documents)}


@router.get("/documents", response_class=PlainTextResponse)
def document(
    doc_id: str = Query(..., alias="id"),
    major_tradition: str = Query(...),
    tradition: str = Query(...),
):
    try:
        text, _ = read_document(doc_id, major_tradition, tradition)
        return text
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="Access denied") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc


@router.get("/archive")
def archive() -> StreamingResponse:
    buf = build_corpus_archive()
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="mythoscope_corpus.zip"'},
    )
