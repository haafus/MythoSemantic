from fastapi import APIRouter, HTTPException

from server.schemas import SavedPlotResponse
from server.services.projections import get_projection_data, get_saved_html_plot

router = APIRouter(prefix="/api/similarity", tags=["projections"])


@router.get("/projections/{model_key}/{method}")
def projection(model_key: str, method: str) -> dict:
    data = get_projection_data(model_key, method)
    if not data:
        saved = get_saved_html_plot(model_key, method)
        raise HTTPException(
            status_code=404,
            detail={
                "message": "Projection JSON not found",
                "saved_html_plot": saved,
            },
        )
    return data


@router.get("/saved-html/{model_key}/{method}", response_model=SavedPlotResponse)
def saved_html_plot(model_key: str, method: str) -> dict:
    return get_saved_html_plot(model_key, method)
