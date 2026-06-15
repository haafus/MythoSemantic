import gc
import logging
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

from settings import settings

logger = logging.getLogger(__name__)

BATCH_SIZE_THRESHOLDS = [(3072, 8), (1024, 16), (768, 24)]
DEFAULT_BATCH_SIZE = 32


def get_optimal_batch_size(model_dim: int) -> int:
    for min_dim, opt_batch in BATCH_SIZE_THRESHOLDS:
        if model_dim >= min_dim:
            return opt_batch
    return DEFAULT_BATCH_SIZE


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ModelManager:
    def __init__(self, *, batch_size: int | None = None):
        self.available_models: list[str] = list(settings.embedding.models)
        self.model_name: str | None = None
        self.model: Any = None
        self.model_dim: int = 0
        self._override_batch_size = batch_size is not None
        self.batch_size: int = batch_size if batch_size is not None else DEFAULT_BATCH_SIZE

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
        model = SentenceTransformer(model_name, device=device)
        logger.info(f"Model '{model_name}' loaded on {device}.")
        return model

    def set_model(self, model_name: str) -> None:
        if model_name not in self.available_models:
            raise ValueError(f"Model '{model_name}' not found. Available: {self.available_models}")

        self.model = self._load_model(model_name)
        self.model_name = model_name
        self.model_dim = self.model.get_sentence_embedding_dimension()
        if not self._override_batch_size:
            self.batch_size = get_optimal_batch_size(self.model_dim)
            logger.info(f"Batch size automatically set to {self.batch_size} for model {model_name}")
        else:
            logger.info(f"Using manual batch size: {self.batch_size}")

    def close(self) -> None:
        self.unload_model()
