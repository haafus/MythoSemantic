import gc
import logging
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

from model_registry import active_embedding_models, resolve_embedding_model

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 32

DEVICE: str = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def _clear_device_cache() -> None:
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps":
        torch.mps.empty_cache()


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

        self.unload_model()
        self.model = SentenceTransformer(model_name, device=DEVICE, trust_remote_code=True)
        self.model_name = model_name
        self.model_dim = self.model.get_embedding_dimension()
        logger.info(f"Model '{model_name}' loaded on {DEVICE}.")

    def unload_model(self) -> None:
        if self.model is None:
            return
        self.model = None
        self.model_name = None
        self.model_dim = 0
        _clear_device_cache()
        gc.collect()
        logger.info("Model unloaded from memory")
