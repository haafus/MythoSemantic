import logging
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from embeddings import chroma_manager
from model_registry import model_to_key
from settings import settings

logger = logging.getLogger(__name__)


class ModelData(NamedTuple):
    model_name: str
    data: list[dict[str, Any]]
    embeddings: np.ndarray
    output_dir: Path


def load_model_data(model_name: str) -> ModelData | None:
    output_dir = settings.projections_dir / model_to_key(model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading data for model: {model_name}...")
    data, embeddings = chroma_manager.get_collection(model_name).load_data()

    if not data:
        logger.warning(f"No data found for model '{model_name}'")
        return None

    logger.info(f"Chunks loaded: {len(data)}")
    return ModelData(model_name=model_name, data=data, embeddings=embeddings, output_dir=output_dir)
