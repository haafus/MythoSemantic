import logging
from pathlib import Path
from typing import Any

import numpy as np

from settings import settings

from embeddings.chroma_manager import load_data

logger = logging.getLogger(__name__)


class EmbeddingAnalyzer:
    def __init__(self, model_name: str | None = None):
        self.model_name: str | None = None
        self.data: list[dict[str, Any]] = []
        self.embeddings: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.output_dir: Path = settings.projections_dir

        if model_name:
            self.set_model(model_name)

    def set_model(self, model_name: str) -> None:
        self.model_name = model_name
        from model_registry import model_to_key
        self.output_dir = settings.projections_dir / model_to_key(model_name)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading data for model: {model_name}...")
        self.data, self.embeddings = load_data(model_name=model_name)

        if not self.data:
            logger.warning(f"No data found for model '{model_name}' ")
        else:
            logger.info(f"Chunks loaded: {len(self.data)}")


