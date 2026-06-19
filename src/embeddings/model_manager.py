import gc
import logging
import os
from typing import Any

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


class ModelManager:
    def __init__(self) -> None:
        self.name: str | None = None
        self.encoder: Any = None

    def load(self, model_name: str) -> None:
        model_name = resolve_embedding_model(model_name)

        if self.encoder is not None and self.name == model_name:
            return

        self.unload()
        self.encoder = SentenceTransformer(model_name, trust_remote_code=True)
        self.name = model_name
        device = str(self.encoder.device)
        mem = _memory_info(device)
        logger.info(f"Model '{model_name}' loaded on {device}{f' ({mem})' if mem else ''}.")

    def unload(self) -> None:
        if self.encoder is None:
            return
        device = str(self.encoder.device)
        self.encoder = None
        self.name = None
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()
        logger.info("Model unloaded from memory")
