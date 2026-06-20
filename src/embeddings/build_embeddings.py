import logging

from model_registry import active_embedding_models, resolve_embedding_model
from settings import settings

from . import chroma_manager
from .builder import EmbeddingBuilder
from .model_manager import EmbeddingEncoder

logger = logging.getLogger(__name__)


def build_embeddings(
    model_name: str | None = None,
    models: list | None = None,
    force: bool = False,
) -> None:
    if model_name:
        models_to_run = [resolve_embedding_model(model_name)]
    else:
        models_to_run = models or active_embedding_models()

    encoder = EmbeddingEncoder()
    encoder.load(models_to_run[0])
    builder = EmbeddingBuilder(encoder)

    logger.info("Starting embedding generation...")
    logger.info(f"   Source: {settings.corpus_dir}")
    logger.info(f"   Embeddings: {settings.embeddings_dir}")

    try:
        for model in models_to_run:
            if force:
                chroma_manager.delete_collection(model)
            else:
                try:
                    coll = chroma_manager.get_collection(model)
                    count = coll.count()
                    expected = (coll.metadata or {}).get("total_chunks")
                    if expected and count >= expected:
                        logger.info(f"   Skipping {model}: collection complete ({count} chunks)")
                        continue
                    if count > 0:
                        logger.info(f"   Resuming {model}: {count} chunks exist")
                except Exception:
                    pass

            encoder.load(model)
            logger.info(f"   Model: {model}")
            logger.info(f"   Model batch size: {builder.batch_size}")
            builder.save_all_corpus_to_chroma()

    except Exception as e:
        logger.error(f"Embedding generation error: {e}")
        raise
    finally:
        encoder.unload()

    logger.info("All embeddings saved to Chroma.")
