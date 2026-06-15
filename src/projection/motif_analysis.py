import hashlib
import json
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from settings import settings

logger = logging.getLogger(__name__)

SUMMARY_PROMPT = (
    "You are a comparative mythology analyst. "
    "Summarize the plot of this text fragment in 2-3 neutral sentences. "
    "Do NOT mention character names, place names, or culture-specific terms. "
    "Replace proper nouns with generic roles (e.g. 'the hero', 'the god', 'the trickster'). "
    "Focus only on the narrative structure: what happens, what transforms, what conflict arises."
)


def _cache_path(output_dir: Path) -> Path:
    return output_dir / "motif_summaries.json"


def _chunk_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _load_cache(path: Path) -> dict[str, str]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupted motif cache, starting fresh")
    return {}


def _save_cache(path: Path, cache: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


def generate_motif_summaries(
    data: list[dict],
    output_dir: Path,
) -> list[str]:
    from graphs.llm_processing import LLMProcessor

    cfg = settings.graphs
    llm = LLMProcessor(
        model_name=cfg.model_name,
        base_url=cfg.base_url,
        temperature=cfg.temperature,
        max_retries=cfg.max_retries,
        retry_backoff_factor=cfg.retry_backoff_factor,
    )

    cache_file = _cache_path(output_dir)
    cache = _load_cache(cache_file)
    summaries: list[str] = []
    new_count = 0

    for item in tqdm(data, desc="Generating motif summaries", unit="chunk"):
        text = item.get("text", "")
        key = _chunk_hash(text)

        if key in cache:
            summaries.append(cache[key])
            continue

        summary = llm.ask_text(SUMMARY_PROMPT, text[:4000])

        cache[key] = summary
        summaries.append(summary)
        new_count += 1

        if new_count % 50 == 0:
            _save_cache(cache_file, cache)

    if new_count > 0:
        _save_cache(cache_file, cache)
        logger.info(f"Generated {new_count} new summaries, {len(summaries) - new_count} from cache")
    else:
        logger.info(f"All {len(summaries)} summaries loaded from cache")

    return summaries


def embed_summaries(summaries: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    logger.info(f"Embedding {len(summaries)} summaries with {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings: np.ndarray = model.encode(
        summaries,
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=False,
    )
    del model
    return embeddings


STORY_EMBEDDING_MODEL = "uhhlt/story-emb"


def run_story_embedding_analysis(
    data: list[dict],
    output_dir: Path,
    story_model: str = STORY_EMBEDDING_MODEL,
    model_name: str | None = None,
    reducer_kwargs: dict | None = None,
) -> None:
    from .visualization import _plot_umap_scatter

    texts = [item.get("text", "") for item in data]
    story_embeddings = embed_summaries(texts, story_model)

    logger.info("Building Story UMAP projection...")
    _plot_umap_scatter(
        data,
        story_embeddings,
        title_prefix="Story UMAP (narrative re-embedding)",
        filename="story_umap_2d.html",
        axis_prefix="Story UMAP",
        save_html=True,
        output_dir=output_dir,
        model_name=model_name,
        reducer_kwargs=reducer_kwargs,
    )
    logger.info("Story UMAP projection saved")


def run_motif_analysis(
    data: list[dict],
    output_dir: Path,
    embedding_model: str,
    model_name: str | None = None,
    reducer_kwargs: dict | None = None,
) -> None:
    from .visualization import _plot_umap_scatter

    summaries = generate_motif_summaries(data, output_dir)

    empty_count = sum(1 for s in summaries if not s.strip())
    if empty_count > len(summaries) * 0.5:
        logger.error(f"Too many empty summaries ({empty_count}/{len(summaries)}), aborting motif UMAP")
        return

    motif_embeddings = embed_summaries(summaries, embedding_model)

    logger.info("Building motif UMAP projection...")
    _plot_umap_scatter(
        data,
        motif_embeddings,
        title_prefix="Motif UMAP (plot summaries, tradition colors)",
        filename="motif_umap_2d.html",
        axis_prefix="Motif UMAP",
        save_html=True,
        output_dir=output_dir,
        model_name=model_name,
        reducer_kwargs=reducer_kwargs,
    )
    logger.info("Motif UMAP projection saved")
