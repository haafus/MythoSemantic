import json
from pathlib import Path

from settings import settings

METHOD_JSON_FILES = {
    "umap": "umap_2d_traditions.json",
    "residual_umap": "residual_umap_2d.json",
    "residual_normalized_umap": "residual_normalized_umap_2d.json",
    "rlace_umap": "rlace_umap_2d.json",
    "story_umap": "story_umap_2d.json",
    "motif_umap": "motif_umap_2d.json",
}

METHOD_HTML_FILES = {
    "distance_heatmap": "distance_heatmap.html",
    "tradition_distribution": "tradition_distribution.html",
}


def get_projection_data(model_key: str, method: str) -> dict | None:
    output_dir = settings.projections_dir / model_key
    filename = METHOD_JSON_FILES.get(method)
    if not filename:
        return None
    json_path = output_dir / filename
    if not json_path.exists():
        return None
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    data["method"] = method
    data.setdefault("source", "json")
    return data


def get_saved_html_plot(model_key: str, method: str) -> dict:
    output_dir = settings.projections_dir / model_key
    filename = METHOD_HTML_FILES.get(method)
    if not filename:
        return {"exists": False, "reason": f"No HTML plot for {method}"}
    html_path = output_dir / filename

    if not html_path.exists():
        return {"exists": False, "reason": f"Saved HTML plot not found for {method}"}

    return {
        "exists": True,
        "url": f"/projections/{model_key}/{filename}",
        "path": str(html_path),
    }
