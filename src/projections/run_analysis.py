import logging

from model_registry import resolve_embedding_model

from . import PROJECTION_METHODS
from .analyzer import EmbeddingAnalyzer
from .visualization import CHART_GENERATORS, SCATTER_TRANSFORMS

logger = logging.getLogger(__name__)


def build_projections(
    model_name: str | None = None,
    generate_all_plots: bool = True,
    motif_analysis: bool = False,
    force: bool = False,
) -> EmbeddingAnalyzer | None:
    from embeddings import chroma_manager

    available_models = chroma_manager.get_available_models()

    if not available_models:
        logger.error("ERROR: No available models in the Chroma database!")
        return None

    models_to_analyze = [resolve_embedding_model(model_name)] if model_name else available_models
    logger.info(f"Models queued for analysis: {models_to_analyze}")

    analyzer: EmbeddingAnalyzer | None = None
    for current_model in models_to_analyze:
        logger.info(f"Starting model analysis: {current_model}")

        analyzer = EmbeddingAnalyzer(model_name=current_model)

        if not analyzer.data:
            logger.warning(f"No data found for model {current_model}, skipping...")
            continue

        if generate_all_plots:
            _generate_plots(analyzer, force=force, include_motif=motif_analysis)

    logger.info("Projection analysis complete.")
    return analyzer


def _generate_plots(analyzer: EmbeddingAnalyzer, force: bool = False, include_motif: bool = False) -> None:
    data = analyzer.data
    embeddings = analyzer.embeddings
    out = analyzer.output_dir

    for method in PROJECTION_METHODS:
        key = method["key"]
        chart_type = method["chart_type"]
        label = method["label"]
        output_path = out / f"{key}.json"

        if key == "motif_umap":
            if not include_motif:
                continue
            if not force and output_path.exists():
                logger.info("Skipping %s (already exists)", label)
                continue
            _generate_motif_plot(analyzer)
            continue

        if not force and output_path.exists():
            logger.info("Skipping %s (already exists)", label)
            continue

        logger.info("Generating %s...", label)
        generator = CHART_GENERATORS[chart_type]
        try:
            kwargs = {}
            if chart_type == "scatter":
                kwargs["transform"] = SCATTER_TRANSFORMS[key]
            generator(data, embeddings, output_path, model_name=analyzer.model_name, **kwargs)
        except Exception:
            logger.exception("Error creating %s", label)

    logger.info("Visualizations for %s: %s", analyzer.model_name, out)


def _generate_motif_plot(analyzer: EmbeddingAnalyzer) -> None:
    from .motif_analysis import run_motif_analysis

    logger.info("Generating Motif UMAP projection (LLM summaries)...")
    try:
        run_motif_analysis(
            analyzer.data,
            output_dir=analyzer.output_dir,
            embedding_model=analyzer.model_name,
        )
    except Exception:
        logger.exception("Error creating Motif UMAP plot")
