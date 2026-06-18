import logging
import sys

import click

from log_setup import setup_logging


COMMAND_ORDER = [
    "corpus", "embeddings", "projections", "graphs",
    None,
    "build", "status", "clean", "server",
]


class OrderedGroup(click.Group):
    def format_help(self, ctx, formatter):
        super().format_help(ctx, formatter)
        formatter.write("\nRun 'mytho COMMAND --help' for details on a specific command.\n")

    def format_commands(self, ctx, formatter):
        commands = []
        for name in COMMAND_ORDER:
            if name is None:
                commands.append((None, None))
            elif name in self.commands:
                cmd = self.commands[name]
                help_text = cmd.get_short_help_str(limit=formatter.width)
                commands.append((name, help_text))

        if commands:
            labels = iter(["Pipeline", "Management"])
            section: list[tuple[str, str]] = []
            for name, help_text in commands:
                if name is None:
                    if section:
                        with formatter.section(next(labels)):
                            formatter.write_dl(section)
                        section = []
                else:
                    section.append((name, help_text or ""))
            if section:
                with formatter.section(next(labels)):
                    formatter.write_dl(section)


@click.group(cls=OrderedGroup)
@click.version_option(package_name="mythoscope", prog_name="MythoScope")
def mytho():
    """MythoScope — computational framework for comparative mythology."""
    setup_logging()


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------
@mytho.command()
@click.option("--force", is_flag=True, help="Overwrite existing files.")
def corpus(force: bool):
    """Download and build the text corpus."""
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
# projections
# ---------------------------------------------------------------------------
@mytho.command()
@click.option("--model", "-m", default=None, help="Embedding model name (all models if omitted).")
@click.option("--no-plots", is_flag=True, help="Skip plot generation, only compute stats.")
@click.option("--motif-analysis", is_flag=True, help="Generate motif UMAP from LLM plot summaries.")
@click.option("--force", is_flag=True, help="Regenerate all plots even if they already exist.")
def projections(model: str | None, no_plots: bool, motif_analysis: bool, force: bool):
    """Generate UMAP projections and embedding visualizations."""
    from projections.run_analysis import analyze_embeddings

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

    run_generate_graphs(llm=model, force=force)
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
SAMPLE_MAX_TEXTS = 5


@mytho.command()
@click.option("--model", "-m", default=None, help="Embedding model (default from config).")
@click.option("--llm", default=None, help="LLM model for graphs (from config/models.json).")
@click.option("--force", is_flag=True, help="Force regeneration of all steps.")
@click.option("--sample", is_flag=True, help="Quick run: first embedding model, limited texts.")
def build(model, llm, force, sample):
    """Run the full analysis pipeline end-to-end."""
    if sample:
        from model_registry import active_embedding_models
        model = model or active_embedding_models()[0]
        max_texts = SAMPLE_MAX_TEXTS
        click.echo(click.style(f"[sample] model={model}, max_texts={max_texts}", fg="yellow"))
    else:
        max_texts = None

    steps = [
        ("Corpus", _build_corpus, {"force": force, "max_texts": max_texts}),
        ("Embeddings", _build_embeddings, {"model": model, "force": force}),
        ("Projections", _build_projections, {"model": model, "force": force}),
        ("Graphs", _build_graphs, {"llm": llm, "force": force, "max_texts": max_texts}),
    ]

    for name, fn, kwargs in steps:
        click.echo(click.style(f"[start] {name}", fg="cyan", bold=True))
        try:
            fn(**kwargs)
            click.echo(click.style(f"[done]  {name}", fg="green"))
        except Exception as e:
            click.echo(click.style(f"[fail]  {name}: {e}", fg="red"), err=True)
            sys.exit(1)

    click.echo(click.style("\nBuild finished.", fg="green", bold=True))


def _build_corpus(force: bool = False, max_texts: int | None = None):
    from corpus.builder import build_corpus

    build_corpus(force=force, max_texts=max_texts)


def _build_embeddings(model: str | None, force: bool = False):
    from embedding.build_embeddings import build_embeddings

    build_embeddings(model_name=model, force=force)


def _build_projections(model: str | None, force: bool = False):
    from projections.run_analysis import analyze_embeddings

    analyze_embeddings(model_name=model, force=force)


def _build_graphs(llm: str | None = None, force: bool = False, max_texts: int | None = None):
    from graphs.run_graph_generation import run_generate_graphs

    run_generate_graphs(llm=llm, force=force, max_texts=max_texts)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------
@mytho.command()
def status():
    """Show the current state of the data pipeline."""
    from pipeline_inspect import (
        corpus_status,
        embeddings_status,
        format_size,
        graphs_status,
        projections_status,
    )
    from settings import settings

    total = 0

    # Corpus
    info = corpus_status(settings)
    total += info["total_size"]
    _header("Corpus", info["total_size"])
    built, cfg, missing = info["built_count"], info["config_count"], info["missing_count"]
    click.echo(f"  {built} texts built (from {cfg} in config)")
    if missing:
        click.echo(click.style(f"  {missing} missing", fg="yellow"))
    click.echo()

    # Embeddings
    info = embeddings_status(settings)
    total += info["total_size"]
    _header("Embeddings", info["total_size"])
    if not info["exists"]:
        click.echo("  No embeddings found")
    elif "error" in info:
        click.echo(click.style(f"  Error: {info['error']}", fg="red"))
    elif not info["collections"]:
        click.echo("  No embedding collections")
    else:
        for col in sorted(info["collections"], key=lambda c: c["model"]):
            click.echo(f"  {col['model']:<40} {col['count']:>6} chunks")
    click.echo()

    # Projections
    info = projections_status(settings)
    total += info["total_size"]
    _header("Projections", info["total_size"])
    if not info["exists"]:
        click.echo("  No projections found")
    elif not info["models"]:
        click.echo("  No model results")
    else:
        for m in info["models"]:
            done, tot = m["plots_done"], m["plots_total"]
            color = "green" if done == tot else "yellow" if done > 0 else "red"
            mark = "ok" if done == tot else f"{done}/{tot}"
            click.echo(click.style(f"  {m['name']:<40} {mark:>5}  {format_size(m['size']):>8}", fg=color))
    click.echo()

    # Graphs
    info = graphs_status(settings)
    total += info["total_size"]
    _header("Graphs", info["total_size"])
    if not info["exists"]:
        click.echo("  No graphs directory")
    else:
        click.echo(f"  {info['count']} graph files")
    click.echo()

    click.echo(click.style(f"Total: {format_size(total)}", bold=True))


def _header(name: str, size: int):
    from pipeline_inspect import format_size
    click.echo(click.style(f"{name}:", bold=True) + f"  {format_size(size)}")


# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------
@mytho.command()
@click.option("--apply", is_flag=True, help="Actually delete orphan files (default is dry run).")
def clean(apply: bool):
    """Find and remove orphan files not used by the pipeline."""
    import shutil

    from pipeline_inspect import (
        corpus_orphans,
        embeddings_orphan_chunks,
        embeddings_orphan_collections,
        format_size,
        graphs_orphans,
        projections_orphans,
    )
    from settings import settings

    total_bytes = 0
    total_items = 0

    # Corpus
    orphans = corpus_orphans(settings)
    if orphans:
        _header("Corpus", sum(s for _, s in orphans))
        for path, size in orphans:
            total_bytes += size
            total_items += 1
            click.echo(f"  {path.relative_to(settings.corpus_dir):<50} {format_size(size):>8}")
            if apply:
                path.unlink(missing_ok=True)
        click.echo()

    # Embeddings orphan collections + chunks
    orphan_cols = embeddings_orphan_collections(settings)
    skip_col_names = {c["name"] for c in orphan_cols}
    orphan_chunks = embeddings_orphan_chunks(settings, skip_collections=skip_col_names)
    if orphan_cols or orphan_chunks:
        click.echo(click.style("Embeddings:", bold=True))
        for col in orphan_cols:
            total_items += 1
            click.echo(f"  orphan collection: {col['model']:<30} {col['count']:>6} chunks")
        for info in orphan_chunks:
            n = len(info["orphan_ids"])
            total_items += n
            click.echo(f"  orphan chunks in {info['model']:<30} {n:>6} / {info['total_count']}")
        if apply:
            import chromadb
            from embedding.chroma_manager import delete_collection
            client = chromadb.PersistentClient(path=str(settings.embeddings_dir))
            for col in orphan_cols:
                delete_collection(client, col["name"])
            for info in orphan_chunks:
                collection = client.get_collection(name=info["collection"])
                collection.delete(ids=info["orphan_ids"])
        click.echo()

    # Projections
    orphans = projections_orphans(settings)
    if orphans:
        _header("Projections", sum(m["size"] for m in orphans))
        for m in orphans:
            total_bytes += m["size"]
            total_items += 1
            click.echo(f"  {m['name']:<50} {format_size(m['size']):>8}")
            if apply:
                shutil.rmtree(m["path"])
        click.echo()

    # Graphs
    orphans = graphs_orphans(settings)
    if orphans:
        _header("Graphs", sum(s for _, s in orphans))
        for path, size in orphans:
            total_bytes += size
            total_items += 1
            name = path.name
            click.echo(f"  {name:<50} {format_size(size):>8}")
            if apply:
                shutil.rmtree(path)
        click.echo()

    if total_items == 0:
        click.echo(click.style("No orphans found.", fg="green"))
        return

    summary = f"{total_items} orphan items"
    if total_bytes:
        summary += f", {format_size(total_bytes)} on disk"

    if apply:
        click.echo(click.style(f"Removed: {summary}", fg="green"))
    else:
        click.echo(f"{summary}")
        click.echo(click.style("Dry run. Use --apply to delete.", fg="yellow"))


if __name__ == "__main__":
    mytho()
