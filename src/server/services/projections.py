import json
from pathlib import Path

from settings import settings

SAVED_HTML_METHOD_FILES = {
    "umap": "umap_2d_traditions.html",
    "residual_umap": "residual_umap_2d.html",
    "residual_normalized_umap": "residual_normalized_umap_2d.html",
    "rlace_umap": "rlace_umap_2d.html",
    "story_umap": "story_umap_2d.html",
    "motif_umap": "motif_umap_2d.html",
    "distance_heatmap": "distance_heatmap.html",
    "tradition_distribution": "tradition_distribution.html",
}


def get_projection_data(model_key: str, method: str) -> dict | None:
    output_dir = settings.projections_dir / model_key

    html_filename = SAVED_HTML_METHOD_FILES.get(method)
    if html_filename:
        json_path = output_dir / html_filename.replace(".html", ".json")
        if json_path.exists():
            with json_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            data["method"] = method
            data.setdefault("source", "json")
            return data

    return None


def get_saved_html_plot(model_key: str, method: str, output_dir: Path | None = None) -> dict:
    if output_dir is None:
        output_dir = settings.projections_dir / model_key
    filename = SAVED_HTML_METHOD_FILES.get(method, f"{method}.html")
    html_path = output_dir / filename

    if not html_path.exists():
        return {
            "exists": False,
            "reason": f"Saved HTML plot not found for {method}",
        }

    return {
        "exists": True,
        "url": f"/projections/{model_key}/{filename}",
        "path": str(html_path),
    }
