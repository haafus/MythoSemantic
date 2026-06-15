import logging
from pathlib import Path

from settings import settings

from .analyzer import EmbeddingAnalyzer
from .visualization import (
    add_click_handler_to_html,
    plot_distance_heatmap,
    plot_interactive_2d,
    plot_residual_normalized_umap,
    plot_residual_umap,
    plot_tradition_distribution,
)

logger = logging.getLogger(__name__)


def generate_clickable_plots(output_dir: Path, model_name: str) -> None:
    html_files = list(output_dir.glob("*.html"))

    if not html_files:
        logger.warning(f"No HTML files found in {output_dir} to make clickable.")
        return

    for filepath in html_files:
        add_click_handler_to_html(str(filepath))


def analyze_embeddings(
    model_name: str | None = None,
    generate_all_plots: bool = True,
    motif_analysis: bool = False,
) -> EmbeddingAnalyzer | None:
    try:
        base_analyzer = EmbeddingAnalyzer()
        available_models = base_analyzer.available_models

        if not available_models:
            logger.error("ERROR: No available models in the Chroma database!")
            return None

        models_to_analyze = [model_name] if model_name else available_models
        logger.info(f"Models queued for analysis: {models_to_analyze}")

        analyzer: EmbeddingAnalyzer | None = None
        for current_model in models_to_analyze:
            logger.info(f"Starting model analysis: {current_model}")

            analyzer = EmbeddingAnalyzer(model_name=current_model)

            if not analyzer.data:
                logger.warning(f"No data found for model {current_model}, skipping...")
                continue

            analyzer.print_statistics()
            analyzer.save_summary()
            analyzer.save_models_list()

            if generate_all_plots and analyzer.data:
                _generate_all_plots(analyzer)

            if motif_analysis and analyzer.data:
                _generate_motif_plot(analyzer)

        return analyzer

    except Exception:
        logger.exception("Critical error during embedding analysis")
        return None


def _generate_all_plots(analyzer: EmbeddingAnalyzer) -> None:
    data = analyzer.filter_by_model()
    cfg = settings.projection

    logger.info("Generating UMAP projection...")
    try:
        plot_interactive_2d(
            data,
            output_dir=analyzer.output_dir,
            model_name=analyzer.model_name,
            reducer_kwargs={"n_neighbors": cfg.umap_n_neighbors, "min_dist": cfg.umap_min_dist},
        )
    except Exception:
        logger.exception("Error creating UMAP plot")

    logger.info("Generating Residual UMAP projection...")
    try:
        plot_residual_umap(
            data,
            output_dir=analyzer.output_dir,
            model_name=analyzer.model_name,
            reducer_kwargs={"n_neighbors": cfg.umap_n_neighbors, "min_dist": cfg.umap_min_dist},
        )
    except Exception:
        logger.exception("Error creating Residual UMAP plot")

    logger.info("Generating Residual Normalized UMAP projection...")
    try:
        plot_residual_normalized_umap(
            data,
            output_dir=analyzer.output_dir,
            model_name=analyzer.model_name,
            reducer_kwargs={"n_neighbors": cfg.umap_n_neighbors, "min_dist": cfg.umap_min_dist},
        )
    except Exception:
        logger.exception("Error creating Residual Normalized UMAP plot")

    logger.info("  - Distance heatmap...")
    try:
        plot_distance_heatmap(data, output_dir=analyzer.output_dir, model_name=analyzer.model_name)
    except Exception:
        logger.exception("Error creating heatmap")

    logger.info("  - Tradition distribution chart...")
    try:
        plot_tradition_distribution(data, output_dir=analyzer.output_dir, model_name=analyzer.model_name)
    except Exception:
        logger.exception("Error creating distribution chart")

    logger.info("  - Adding click handlers...")
    try:
        if analyzer.model_name:
            generate_clickable_plots(analyzer.output_dir, analyzer.model_name)
    except Exception:
        logger.exception("Error adding click handlers")

    logger.info(f"\nAll visualizations for {analyzer.model_name} saved to: {analyzer.output_dir}")


def _generate_motif_plot(analyzer: EmbeddingAnalyzer) -> None:
    from .motif_analysis import run_motif_analysis

    data = analyzer.filter_by_model()
    cfg = settings.projection

    logger.info("Generating Motif UMAP projection (LLM summaries)...")
    try:
        run_motif_analysis(
            data,
            output_dir=analyzer.output_dir,
            embedding_model=analyzer.model_name,
            model_name=analyzer.model_name,
            reducer_kwargs={"n_neighbors": cfg.umap_n_neighbors, "min_dist": cfg.umap_min_dist},
        )
    except Exception:
        logger.exception("Error creating Motif UMAP plot")
