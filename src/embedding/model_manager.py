import gc
import logging
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

from model_registry import active_embedding_models, resolve_embedding_model

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 32


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ModelManager:
    def __init__(self, *, batch_size: int | None = None):
        self.available_models: list[str] = active_embedding_models()
        self.model_name: str | None = None
        self.model: Any = None
        self.model_dim: int = 0
        self.batch_size: int = batch_size or DEFAULT_BATCH_SIZE

    def unload_model(self) -> None:
        if self.model is not None:
            self.model = None
            self.model_name = None
            self.model_dim = 0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()
            logger.info("Model unloaded from memory")

    def _load_model(self, model_name: str) -> SentenceTransformer:
        if self.model is not None and self.model_name == model_name:
            return self.model

        if self.model is not None:
            self.unload_model()

        device = _select_device()
        model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        logger.info(f"Model '{model_name}' loaded on {device}.")
        return model

    def set_model(self, model_name: str) -> None:
        model_name = resolve_embedding_model(model_name)
        if model_name not in self.available_models:
            raise ValueError(f"Model '{model_name}' not found. Available: {self.available_models}")

        self.model = self._load_model(model_name)
        self.model_name = model_name
        self.model_dim = self.model.get_embedding_dimension()

    def close(self) -> None:
        self.unload_model()
