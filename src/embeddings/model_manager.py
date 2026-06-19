import gc
import logging
import os
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from model_registry import resolve_embedding_model

logger = logging.getLogger(__name__)


def _memory_info(device: str) -> str:
    try:
        if device.startswith("cuda"):
            free, total = torch.cuda.mem_get_info(device)
            return f"VRAM: {free / 2**30:.1f} / {total / 2**30:.1f} GB free"
        if device == "mps":
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            return f"RAM: {total / 2**30:.1f} GB total (shared)"
        import psutil
        mem = psutil.virtual_memory()
        return f"RAM: {mem.available / 2**30:.1f} / {mem.total / 2**30:.1f} GB available"
    except Exception:
        return ""


class EmbeddingEncoder:
    def __init__(self) -> None:
        self.model_name: str | None = None
        self._model: Any = None

    def load(self, model_name: str) -> None:
        model_name = resolve_embedding_model(model_name)

        if self._model is not None and self.model_name == model_name:
            return

        self.unload()
        self._model = SentenceTransformer(model_name, trust_remote_code=True)
        self.model_name = model_name
        device = str(self._model.device)
        mem = _memory_info(device)
        logger.info(f"Model '{model_name}' loaded on {device}{f' ({mem})' if mem else ''}.")

    def encode(self, texts: list[str], **kwargs) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("No model loaded. Call load() first.")
        return self._model.encode(texts, **kwargs)

    def unload(self) -> None:
        if self._model is None:
            return
        device = str(self._model.device)
        self._model = None
        self.model_name = None
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()
        logger.info("Model unloaded from memory")
