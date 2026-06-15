from __future__ import annotations

import gc
import logging
import shutil

import torch

from settings import settings

from .builder import EmbeddingBuilder
from .chroma_manager import collection_name_for_model

logger = logging.getLogger(__name__)


def build_embeddings(
    clear_existing: bool | None = None,
    batch_size: int | None = None,
    model_name: str | None = None,
    models: list | None = None,
    chunking: str | None = None,
):
    if clear_existing is False:
        raise ValueError("Incremental Chroma writes are not supported for full embedding generation.")

    emb = settings.embedding

    MODEL_NAME = model_name or emb.models[0]
    CHUNKING = chunking or emb.default_chunking
    BATCH_SIZE = batch_size if batch_size is not None else emb.batch_size
    CLEAR_EXISTING = clear_existing if clear_existing is not None else True

    if CLEAR_EXISTING:
        chroma_dir = settings.chroma_dir
        if chroma_dir.exists():
            logger.info(f"Fully clearing ChromaDB directory: {chroma_dir.resolve()}")
            try:
                shutil.rmtree(chroma_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Failed to fully remove database directory: {e}")

    builder = EmbeddingBuilder(
        corpus_dir=settings.corpus_dir,
        out_dir=settings.analysis_dir,
        embedding_model=MODEL_NAME,
        chunking=CHUNKING,
        chroma_path=settings.chroma_dir,
        batch_size=BATCH_SIZE,
        chroma_batch_size=emb.chroma_batch_size,
    )

    models_to_run = models or ([MODEL_NAME] if model_name else emb.models or [MODEL_NAME])

    logger.info("Starting embedding generation...")
    logger.info(f"   Source: {settings.corpus_dir}")
    logger.info(f"   Chroma DB: {settings.chroma_dir}")
    logger.info(f"   Results directory: {builder.out_dir}")
    logger.info(f"   Clear collection: {CLEAR_EXISTING}")

    try:
        for model in models_to_run:
            logger.info(f"   Model: {model}")
            logger.info(f"   Model batch size: {BATCH_SIZE}")

            builder.set_model(model)
            logger.info(f"Collection: {collection_name_for_model(model)}")
            builder.save_all_corpus_to_chroma()

    except Exception as e:
        logger.error(f"Embedding generation error: {e}")
        raise
    finally:
        builder.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    logger.info("All embeddings saved to Chroma.")
    logger.info(f"Analysis results: {builder.out_dir}")

    return builder
