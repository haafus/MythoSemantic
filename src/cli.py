import logging
import sys

import click

from log_setup import setup_logging




@click.group()
@click.version_option(package_name="mythoscope")
def mytho():
    """MythoScope — computational framework for comparative mythology."""
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
# embeddings
# ---------------------------------------------------------------------------
@mytho.command()
@click.option("--model", "-m", default=None, help="Embedding model to use.")
@click.option("--force", is_flag=True, help="Regenerate even if collection exists.")
def embeddings(model: str | None, force: bool):
    """Generate embeddings for the corpus."""
    from embedding.build_embeddings import build_embeddings

    build_embeddings(model_name=model, force=force)


# ---------------------------------------------------------------------------
# projection
# ---------------------------------------------------------------------------
@mytho.command()
@click.option("--model", "-m", default=None, help="Embedding model name (all models if omitted).")
@click.option("--no-plots", is_flag=True, help="Skip plot generation, only compute stats.")
@click.option("--motif-analysis", is_flag=True, help="Generate motif UMAP from LLM plot summaries.")
@click.option("--force", is_flag=True, help="Regenerate all plots even if they already exist.")
def projection(model: str | None, no_plots: bool, motif_analysis: bool, force: bool):
    """Generate UMAP projections and embedding visualizations."""
    from projection.run_analysis import analyze_embeddings

    analyzer = analyze_embeddings(model_name=model, generate_all_plots=not no_plots, motif_analysis=motif_analysis, force=force)
    if analyzer is None:
        click.echo(click.style("No data found — check that embeddings exist.", fg="red"), err=True)
        sys.exit(1)
    click.echo(click.style("Projection analysis completed.", fg="green"))


# ---------------------------------------------------------------------------
# graphs
# ---------------------------------------------------------------------------
@mytho.command()
@click.option("--model", "-m", default=None, help="LLM model name from config/models.json registry.")
@click.option("--force", is_flag=True, help="Overwrite existing graph outputs.")
def graphs(model: str | None, force: bool):
    """Extract knowledge graphs from corpus texts using an LLM."""
    from graphs.run_graph_generation import run_generate_graphs

    run_generate_graphs(llm_model=model, force=force)
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
# build — run everything end-to-end
# ---------------------------------------------------------------------------
@mytho.command()
@click.option("--model", "-m", default=None, help="Embedding model (default from config).")
@click.option("--llm-model", default=None, help="LLM model for graphs (from config/models.json).")
@click.option("--force", is_flag=True, help="Force regeneration of all steps.")
@click.option("--skip-corpus", is_flag=True, help="Skip corpus download.")
@click.option("--skip-embeddings", is_flag=True, help="Skip embedding generation.")
@click.option("--skip-projection", is_flag=True, help="Skip projection generation.")
@click.option("--skip-graphs", is_flag=True, help="Skip graph extraction.")
def build(model, llm_model, force, skip_corpus, skip_embeddings, skip_projection, skip_graphs):
    """Run the full analysis pipeline end-to-end."""
    steps = [
        ("Corpus", skip_corpus, _build_corpus, {"force": force}),
        ("Embeddings", skip_embeddings, _build_embeddings, {"model": model, "force": force}),
        ("Projection", skip_projection, _build_projection, {"model": model, "force": force}),
        ("Graphs", skip_graphs, _build_graphs, {"llm_model": llm_model, "force": force}),
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

    click.echo(click.style("\nBuild finished.", fg="green", bold=True))


def _build_corpus(force: bool = False):
    from corpus.builder import build_corpus

    build_corpus(force=force)


def _build_embeddings(model: str | None, force: bool = False):
    from embedding.build_embeddings import build_embeddings

    build_embeddings(model_name=model, force=force)


def _build_projection(model: str | None, force: bool = False):
    from projection.run_analysis import analyze_embeddings

    analyze_embeddings(model_name=model, force=force)


def _build_graphs(llm_model: str | None = None, force: bool = False):
    from graphs.run_graph_generation import run_generate_graphs

    run_generate_graphs(llm_model=llm_model, force=force)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------
PROJECTION_PLOTS = [
    "umap_2d_traditions.html",
    "residual_umap_2d.html",
    "residual_normalized_umap_2d.html",
    "rlace_umap_2d.html",
    "distance_heatmap.html",
    "tradition_distribution.html",
]


@mytho.command()
def status():
    """Show the current state of the data pipeline."""
    import json
    from pathlib import Path

    from settings import settings

    _status_corpus(settings)
    _status_embeddings(settings)
    _status_projections(settings)
    _status_graphs(settings)


def _status_corpus(settings):
    import json

    click.echo(click.style("Corpus:", bold=True))

    config_path = settings.corpus_config_file
    config_count = 0
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                config_count = len(json.load(f))
        except Exception:
            pass

    meta_path = settings.corpus_metadata_path
    meta_count = 0
    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta_count = len(json.load(f))
        except Exception:
            pass

    click.echo(f"  {meta_count} texts built (from {config_count} in config)")
    if config_count > meta_count:
        click.echo(click.style(f"  {config_count - meta_count} missing", fg="yellow"))
    click.echo()


def _status_embeddings(settings):
    click.echo(click.style("Embeddings:", bold=True))

    chroma_path = settings.chroma_dir
    if not chroma_path.exists():
        click.echo("  No ChromaDB found")
        click.echo()
        return

    try:
        import chromadb

        from embedding.chroma_manager import is_model_collection_name

        client = chromadb.PersistentClient(path=str(chroma_path))
        collections = client.list_collections()
        model_collections = []
        for col in collections:
            if is_model_collection_name(col.name):
                model_name = (col.metadata or {}).get("model", col.name)
                model_collections.append((model_name, col.count()))

        if not model_collections:
            click.echo("  No embedding collections")
        else:
            click.echo(f"  {len(model_collections)} models:")
            for model_name, count in sorted(model_collections):
                click.echo(f"    {model_name:<40} {count:>6} chunks")
    except Exception as e:
        click.echo(click.style(f"  Error reading ChromaDB: {e}", fg="red"))

    click.echo()


def _status_projections(settings):
    click.echo(click.style("Projections:", bold=True))

    analysis_dir = settings.analysis_dir
    if not analysis_dir.exists():
        click.echo("  No analysis directory")
        click.echo()
        return

    model_dirs = sorted(d for d in analysis_dir.iterdir() if d.is_dir())
    if not model_dirs:
        click.echo("  No model results")
    else:
        for model_dir in model_dirs:
            existing = [p for p in PROJECTION_PLOTS if (model_dir / p).exists()]
            total = len(PROJECTION_PLOTS)
            count = len(existing)
            color = "green" if count == total else "yellow" if count > 0 else "red"
            mark = "ok" if count == total else f"{count}/{total}"
            click.echo(click.style(f"  {model_dir.name:<40} {mark}", fg=color))

    click.echo()


def _status_graphs(settings):
    click.echo(click.style("Graphs:", bold=True))

    graphs_dir = settings.graphs_dir
    if not graphs_dir.exists():
        click.echo("  No graphs directory")
        click.echo()
        return

    html_files = list(graphs_dir.glob("*.html"))
    click.echo(f"  {len(html_files)} graph files")
    click.echo()


if __name__ == "__main__":
    mytho()
