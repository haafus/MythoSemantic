import json
from pathlib import Path

from settings import settings

METHODS = {
    "umap",
    "residual_umap",
    "residual_normalized_umap",
    "rlace_umap",
    "story_umap",
    "motif_umap",
    "distance_heatmap",
    "tradition_distribution",
}


def get_projection_data(model_key: str, method: str) -> dict | None:
    if method not in METHODS:
        return None
    json_path = settings.projections_dir / model_key / f"{method}.json"
    if not json_path.exists():
        return None
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    data["method"] = method
    return data
