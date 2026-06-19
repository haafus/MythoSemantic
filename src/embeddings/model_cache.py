import logging
import threading
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingModelCache:
    def __init__(self):
        self._model: Any = None
        self._model_name: str | None = None
        self._lock = threading.RLock()

    def encode(self, model_name: str, texts: list[str]) -> np.ndarray:
        model = self._get_model(model_name)
        raw = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(raw[0] if len(texts) == 1 else raw, dtype=np.float32)

    def _get_model(self, model_name: str) -> Any:
        from sentence_transformers import SentenceTransformer

        with self._lock:
            if self._model_name == model_name:
                return self._model

            logger.info(f"Loading search model: {model_name}")
            self._model = SentenceTransformer(model_name)
            self._model_name = model_name
            return self._model
