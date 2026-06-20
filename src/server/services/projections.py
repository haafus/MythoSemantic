import json
from pathlib import Path

from settings import settings

METHOD_FILES = {
    "umap": "umap_2d_traditions.json",
    "residual_umap": "residual_umap_2d.json",
    "residual_normalized_umap": "residual_normalized_umap_2d.json",
    "rlace_umap": "rlace_umap_2d.json",
    "story_umap": "story_umap_2d.json",
    "motif_umap": "motif_umap_2d.json",
    "distance_heatmap": "distance_heatmap.json",
    "tradition_distribution": "tradition_distribution.json",
}


def get_projection_data(model_key: str, method: str) -> dict | None:
    filename = METHOD_FILES.get(method)
    if not filename:
        return None
    json_path = settings.projections_dir / model_key / filename
    if not json_path.exists():
        return None
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    data["method"] = method
    return data
