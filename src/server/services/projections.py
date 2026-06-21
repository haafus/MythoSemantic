import json
from pathlib import Path

from projections import PROJECTION_METHODS
from settings import settings


def get_projection_data(model_key: str, method: str) -> dict | None:
    if method not in PROJECTION_METHODS:
        return None
    json_path = settings.projections_dir / model_key / f"{method}.json"
    if not json_path.exists():
        return None
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    data["method"] = method
    return data
