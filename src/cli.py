import logging
import sys

import click

from log_setup import setup_logging


class _LazyEmbeddingGroup(click.Group):
    """Defers import of embedding.cli until a subcommand is actually invoked."""

    _loaded = False

    def _load(self):
        if not self._loaded:
            from embedding.cli import delete_chroma_collection, generate, query

            for cmd in [generate, query, delete_chroma_collection]:
                self.add_command(cmd)
            self._loaded = True

    def list_commands(self, ctx):
        self._load()
        return super().list_commands(ctx)

    def get_command(self, ctx, cmd_name):
        self._load()
        return super().get_command(ctx, cmd_name)


@click.group()
@click.version_option(package_name="mythosemantic")
def mytho():
    """MythoSemantic — computational framework for comparative mythology."""
    setup_logging()


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------
@mytho.command()
@click.option("--force", is_flag=True, help="Overwrite existing files.")
def corpus(force: bool):
    """Download and build the text corpus (Gutenberg cleanup is automatic)."""
    from corpus.builder import build_corpus

    build_corpus(force=force)
    click.echo(click.style("Corpus build completed.", fg="green"))


# ---------------------------------------------------------------------------
# embeddings — delegate to the existing click group
# ---------------------------------------------------------------------------
@mytho.group(cls=_LazyEmbeddingGroup)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output.")
@click.pass_context
def embeddings(ctx, verbose: bool):
    """Generate, query, and manage embeddings."""
    ctx.ensure_object(dict)
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


# ---------------------------------------------------------------------------
# projection
# ---------------------------------------------------------------------------
@mytho.command()
@click.option("--model", "-m", default=None, help="Embedding model name (all models if omitted).")
@click.option("--no-plots", is_flag=True, help="Skip plot generation, only compute stats.")
def projection(model: str | None, no_plots: bool):
    """Generate UMAP projections and embedding visualizations."""
    from projection.run_analysis import analyze_embeddings

    analyzer = analyze_embeddings(model_name=model, generate_all_plots=not no_plots)
    if analyzer is None:
        click.echo(click.style("No data found — check that embeddings exist.", fg="red"), err=True)
        sys.exit(1)
    click.echo(click.style("Projection analysis completed.", fg="green"))


# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------
@mytho.command()
@click.option("--model", "-m", default=None, help="Embedding model name.")
@click.option(
    "--algorithm", "-a", "clustering_model", default="kmeans",
    help="Clustering algorithm (kmeans, spectral, birch, gmm).",
)
@click.option("--single-model", is_flag=True, help="Run only the selected algorithm.")
@click.option("--no-viz", is_flag=True, help="Skip visualization generation.")
@click.option("--output-dir", default="outputs/analysis", help="Output base directory.")
@click.option("--models-list", multiple=True, help="Explicit list of algorithms to run.")
def cluster(model, clustering_model, single_model, no_viz, output_dir, models_list):
    """Run clustering analysis on embeddings."""
    from clustering.run_clustering import run_all_clustering_models, run_clustering_analysis
    from projection.analyzer import EmbeddingAnalyzer

    analyzer = EmbeddingAnalyzer()
    if not analyzer.available_models:
        click.echo(click.style("No available embedding models in the database.", fg="red"), err=True)
        sys.exit(1)

    models_to_process = [model] if model else analyzer.available_models

    for current_model in models_to_process:
        click.echo(f"Clustering: {current_model}")
        if single_model:
            run_clustering_analysis(
                model_name=current_model,
                clustering_model=clustering_model,
                generate_visualizations=not no_viz,
                output_base_dir=output_dir,
                _analyzer=analyzer,
            )
        else:
            run_all_clustering_models(
                model_name=current_model,
                models_to_run=list(models_list) or None,
                output_base_dir=output_dir,
                _analyzer=analyzer,
            )

    click.echo(click.style("Clustering completed.", fg="green"))


# ---------------------------------------------------------------------------
# graphs
# ---------------------------------------------------------------------------
@mytho.command()
@click.option("--force", is_flag=True, help="Overwrite existing graph outputs.")
def graphs(force: bool):
    """Extract knowledge graphs from corpus texts using an LLM."""
    from graphs.run_graph_generation import run_generate_graphs

    run_generate_graphs(force=force)
    click.echo(click.style("Graph generation completed.", fg="green"))


# ---------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------
@mytho.command()
@click.option("--host", "-h", default=None, help="Bind address (default from config).")
@click.option("--port", "-p", default=None, type=int, help="Port (default from config).")
def server(host: str | None, port: int | None):
    """Start the web UI server."""
    import uvicorn

    from settings import settings

    uvicorn.run(
        "main:app",
        host=host or settings.server.host,
        port=port or settings.server.port,
        reload=False,
    )


# ---------------------------------------------------------------------------
# pipeline — run everything end-to-end
# ---------------------------------------------------------------------------
@mytho.command()
@click.option("--model", "-m", default=None, help="Embedding model (default from config).")
@click.option("--skip-corpus", is_flag=True, help="Skip corpus download.")
@click.option("--skip-embeddings", is_flag=True, help="Skip embedding generation.")
@click.option("--skip-projection", is_flag=True, help="Skip projection generation.")
@click.option("--skip-clustering", is_flag=True, help="Skip clustering analysis.")
@click.option("--skip-graphs", is_flag=True, help="Skip graph extraction.")
def pipeline(model, skip_corpus, skip_embeddings, skip_projection, skip_clustering, skip_graphs):
    """Run the full analysis pipeline end-to-end."""
    steps = [
        ("Corpus", skip_corpus, _pipeline_corpus, {}),
        ("Embeddings", skip_embeddings, _pipeline_embeddings, {"model": model}),
        ("Projection", skip_projection, _pipeline_projection, {"model": model}),
        ("Clustering", skip_clustering, _pipeline_clustering, {"model": model}),
        ("Graphs", skip_graphs, _pipeline_graphs, {}),
    ]

    for name, skip, fn, kwargs in steps:
        if skip:
            click.echo(click.style(f"[skip] {name}", fg="yellow"))
            continue
        click.echo(click.style(f"[start] {name}", fg="cyan", bold=True))
        try:
            fn(**kwargs)
            click.echo(click.style(f"[done]  {name}", fg="green"))
        except Exception as e:
            click.echo(click.style(f"[fail]  {name}: {e}", fg="red"), err=True)
            sys.exit(1)

    click.echo(click.style("\nPipeline finished.", fg="green", bold=True))


def _pipeline_corpus():
    from corpus.builder import build_corpus

    build_corpus()


def _pipeline_embeddings(model: str | None):
    from embedding.build_embeddings import build_embeddings

    build_embeddings(model_name=model)


def _pipeline_projection(model: str | None):
    from projection.run_analysis import analyze_embeddings

    analyze_embeddings(model_name=model)


def _pipeline_clustering(model: str | None):
    from clustering.run_clustering import run_all_clustering_models
    from projection.analyzer import EmbeddingAnalyzer

    analyzer = EmbeddingAnalyzer(model_name=model)
    if analyzer.available_models:
        models = [model] if model else analyzer.available_models
        for m in models:
            run_all_clustering_models(model_name=m, _analyzer=analyzer)


def _pipeline_graphs():
    from graphs.run_graph_generation import run_generate_graphs

    run_generate_graphs()


if __name__ == "__main__":
    mytho()
