import logging
from pathlib import Path

from model_registry import resolve_embedding_model

from .analyzer import EmbeddingAnalyzer
from .visualization import (
    add_click_handler_to_html,
    plot_distance_heatmap,
    plot_interactive_2d,
    plot_residual_normalized_umap,
    plot_residual_umap,
    plot_rlace_umap,
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
    force: bool = False,
) -> EmbeddingAnalyzer | None:
    try:
        from .loader import EmbeddingDataLoader

        available_models = EmbeddingDataLoader().get_available_models()

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
                _generate_all_plots(analyzer, force=force)

            if motif_analysis:
                _generate_motif_plot(analyzer, force=force)

        logger.info("Projection analysis complete.")
        return analyzer

    except Exception:
        logger.exception("Critical error during embedding analysis")
        return None


def _generate_all_plots(analyzer: EmbeddingAnalyzer, force: bool = False) -> None:
    data = analyzer.data
    out = analyzer.output_dir

    umap_plots = [
        ("UMAP projection", "umap_2d_traditions.html", plot_interactive_2d),
        ("Residual UMAP projection", "residual_umap_2d.html", plot_residual_umap),
        ("Residual Normalized UMAP projection", "residual_normalized_umap_2d.html", plot_residual_normalized_umap),
        ("RLACE UMAP projection (INLP concept erasure)", "rlace_umap_2d.html", plot_rlace_umap),
    ]
    simple_plots = [
        ("Distance heatmap", "distance_heatmap.html", plot_distance_heatmap),
        ("Tradition distribution chart", "tradition_distribution.html", plot_tradition_distribution),
    ]

    generated = False
    for label, filename, plot_fn in umap_plots:
        if not force and (out / filename).exists():
            logger.info("Skipping %s (already exists)", label)
            continue
        logger.info("Generating %s...", label)
        try:
            plot_fn(data, output_dir=out, model_name=analyzer.model_name)
            generated = True
        except Exception:
            logger.exception("Error creating %s", label)

    for label, filename, plot_fn in simple_plots:
        if not force and (out / filename).exists():
            logger.info("Skipping %s (already exists)", label)
            continue
        logger.info("Generating %s...", label)
        try:
            plot_fn(data, output_dir=out, model_name=analyzer.model_name)
            generated = True
        except Exception:
            logger.exception("Error creating %s", label)

    if generated:
        try:
            if analyzer.model_name:
                generate_clickable_plots(out, analyzer.model_name)
        except Exception:
            logger.exception("Error adding click handlers")

    logger.info("Visualizations for %s: %s", analyzer.model_name, out)


def _generate_motif_plot(analyzer: EmbeddingAnalyzer, force: bool = False) -> None:
    if not force and (analyzer.output_dir / "motif_umap_2d.html").exists():
        logger.info("Skipping Motif UMAP (already exists)")
        return

    from .motif_analysis import run_motif_analysis

    data = analyzer.data

    logger.info("Generating Motif UMAP projection (LLM summaries)...")
    try:
        run_motif_analysis(
            data,
            output_dir=analyzer.output_dir,
            embedding_model=analyzer.model_name,
            model_name=analyzer.model_name,
        )
    except Exception:
        logger.exception("Error creating Motif UMAP plot")


