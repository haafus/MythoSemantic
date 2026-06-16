import logging
from pathlib import Path
from typing import Any

import numpy as np

from settings import settings

from .loader import EmbeddingDataLoader

logger = logging.getLogger(__name__)


class EmbeddingAnalyzer:
    def __init__(self, model_name: str | None = None):
        self.loader = EmbeddingDataLoader()
        self.model_name: str | None = None
        self.data: list[dict[str, Any]] = []
        self._is_loaded = False
        self.output_dir: Path = settings.analysis_dir

        if model_name:
            self.set_model(model_name)

    def set_model(self, model_name: str) -> None:
        self.model_name = model_name
        self.output_dir = settings.model_output_dir(model_name)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading data for model: {model_name}...")
        self.data = self.loader.load_data(model_name=model_name)
        self._is_loaded = bool(self.data)

        if not self.data:
            logger.warning(f"No data found for model '{model_name}' ")
        else:
            logger.info(f"Chunks loaded: {len(self.data)}")

    def filter_by_model(self) -> list[dict[str, Any]]:
        if not self._is_loaded or not self.data:
            raise RuntimeError("Data is not loaded. Call .set_model() first.")
        return self.data

    def get_statistics(self) -> dict[str, Any]:
        if not self._is_loaded or not self.data:
            raise RuntimeError("Data is not loaded. Call .set_model() first.")

        embeddings = np.stack([item["embedding"] for item in self.data])
        traditions = {item["tradition"] for item in self.data}

        return {
            "n_samples": len(self.data),
            "embedding_dim": embeddings.shape[1],
            "traditions": len(traditions),
            "tradition_counts": {t: sum(1 for item in self.data if item["tradition"] == t) for t in traditions},
            "model": self.model_name,
            "total_chunks_in_db": len(self.data),
        }

    def print_statistics(self) -> None:
        if not self._is_loaded or not self.data:
            logger.warning("No data loaded!")
            return

        stats = self.get_statistics()
        lines = [
            "Embedding statistics:",
            f"   Model: {self.model_name}",
            f"   Chunks: {stats['n_samples']}",
            f"   Dimension: {stats['embedding_dim']}",
            f"   Traditions: {stats['traditions']}",
            "   Tradition Distribution:",
        ]
        for trad, count in sorted(stats["tradition_counts"].items(), key=lambda x: -x[1]):
            lines.append(f"     {trad:<20}: {count:>4}")
        logger.info("\n".join(lines))
