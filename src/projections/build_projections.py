import logging

from model_registry import resolve_embedding_model

from . import PROJECTION_METHODS
from .analyzer import ModelData, load_model_data
from .visualization import CHART_GENERATORS, SCATTER_TRANSFORMS

logger = logging.getLogger(__name__)


def build_projections(
    model_name: str | None = None,
    generate_all_plots: bool = True,
    motif_analysis: bool = False,
    force: bool = False,
) -> ModelData | None:
    from embeddings import chroma_manager

    available_models = chroma_manager.get_available_models()

    if not available_models:
        logger.error("ERROR: No available models in the Chroma database!")
        return None

    models_to_analyze = [resolve_embedding_model(model_name)] if model_name else available_models
    logger.info(f"Models queued for analysis: {models_to_analyze}")

    result: ModelData | None = None
    for current_model in models_to_analyze:
        logger.info(f"Starting model analysis: {current_model}")

        model_data = load_model_data(current_model)

        if model_data is None:
            logger.warning(f"No data found for model {current_model}, skipping...")
            continue

        result = model_data
        if generate_all_plots:
            _generate_plots(model_data, force=force, include_motif=motif_analysis)

    logger.info("Projection analysis complete.")
    return result


def _generate_plots(model_data: ModelData, force: bool = False, include_motif: bool = False) -> None:
    for method in PROJECTION_METHODS:
        key = method["key"]
        chart_type = method["chart_type"]
        label = method["label"]
        output_path = model_data.output_dir / f"{key}.json"

        if key == "motif_umap":
            if not include_motif:
                continue
            if not force and output_path.exists():
                logger.info("Skipping %s (already exists)", label)
                continue
            _generate_motif_plot(model_data)
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
            generator(model_data.data, model_data.embeddings, output_path, model_name=model_data.model_name, **kwargs)
        except Exception:
            logger.exception("Error creating %s", label)

    logger.info("Visualizations for %s: %s", model_data.model_name, model_data.output_dir)


def _generate_motif_plot(model_data: ModelData) -> None:
    from .motif_analysis import run_motif_analysis

    logger.info("Generating Motif UMAP projection (LLM summaries)...")
    try:
        run_motif_analysis(
            model_data.data,
            output_dir=model_data.output_dir,
            embedding_model=model_data.model_name,
        )
    except Exception:
        logger.exception("Error creating Motif UMAP plot")
