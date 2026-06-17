import gc
import logging
from typing import Any

from sentence_transformers import SentenceTransformer

from model_registry import active_embedding_models, resolve_embedding_model

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 32


class ModelManager:
    def __init__(self) -> None:
        self.model_name: str | None = None
        self.model: Any = None
        self.model_dim: int = 0

    def set_model(self, model_name: str) -> None:
        model_name = resolve_embedding_model(model_name)
        available = active_embedding_models()
        if model_name not in available:
            raise ValueError(f"Model '{model_name}' not found. Available: {available}")

        if self.model is not None and self.model_name == model_name:
            return

        self._unload()
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.model_name = model_name
        self.model_dim = self.model.get_embedding_dimension()
        logger.info(f"Model '{model_name}' loaded on {self.model.device}.")

    def _unload(self) -> None:
        if self.model is None:
            return
        self.model = None
        self.model_name = None
        self.model_dim = 0
        gc.collect()
        logger.info("Model unloaded from memory")
