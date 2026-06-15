from __future__ import annotations

import gc
import logging

import torch

from settings import settings

from .builder import EmbeddingBuilder
from .chroma_manager import collection_name_for_model, delete_collection

logger = logging.getLogger(__name__)


def build_embeddings(
    batch_size: int | None = None,
    model_name: str | None = None,
    models: list | None = None,
    chunking: str | None = None,
    force: bool = False,
):
    emb = settings.embedding

    MODEL_NAME = model_name or emb.models[0]
    CHUNKING = chunking or emb.default_chunking
    BATCH_SIZE = batch_size if batch_size is not None else emb.batch_size

    builder = EmbeddingBuilder(
        corpus_dir=settings.corpus_dir,
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

    try:
        for model in models_to_run:
            collection_name = collection_name_for_model(model)

            if not force:
                try:
                    collection = builder.chroma_client.get_collection(name=collection_name)
                    count = collection.count()
                    if count > 0:
                        logger.info(f"   Skipping {model}: collection '{collection_name}' already has {count} entries (use --force to regenerate)")
                        continue
                except Exception:
                    pass

            delete_collection(builder.chroma_client, collection_name)
            builder.set_model(model)
            logger.info(f"   Model: {model}")
            logger.info(f"   Model batch size: {builder.batch_size}")
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

    return builder
