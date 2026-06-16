import logging
from pathlib import Path
from typing import Any

from settings import settings

from .loader import EmbeddingDataLoader

logger = logging.getLogger(__name__)


class EmbeddingAnalyzer:
    def __init__(self, model_name: str | None = None):
        self.loader = EmbeddingDataLoader()
        self.model_name: str | None = None
        self.data: list[dict[str, Any]] = []
        self.output_dir: Path = settings.analysis_dir

        if model_name:
            self.set_model(model_name)

    def set_model(self, model_name: str) -> None:
        self.model_name = model_name
        self.output_dir = settings.model_output_dir(model_name)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading data for model: {model_name}...")
        self.data = self.loader.load_data(model_name=model_name)

        if not self.data:
            logger.warning(f"No data found for model '{model_name}' ")
        else:
            logger.info(f"Chunks loaded: {len(self.data)}")

    def filter_by_model(self) -> list[dict[str, Any]]:
        if not self.data:
            raise RuntimeError("Data is not loaded. Call .set_model() first.")
        return self.data

