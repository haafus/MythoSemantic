import logging
import threading
from collections import OrderedDict
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MAX_CACHED_MODELS = 2


class EmbeddingModelCache:
    def __init__(self, max_models: int = MAX_CACHED_MODELS):
        self._models: OrderedDict[str, Any] = OrderedDict()
        self._max_models = max_models
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
            if model_name in self._models:
                self._models.move_to_end(model_name)
                return self._models[model_name]

            model = SentenceTransformer(model_name)
            self._models[model_name] = model

            while len(self._models) > self._max_models:
                evicted_name, _ = self._models.popitem(last=False)
                logger.info(f"Evicted model cache: {evicted_name}")

            return model
