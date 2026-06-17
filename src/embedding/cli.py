import time

import click

from model_registry import default_embedding_model
from settings import settings

from .build_embeddings import build_embeddings
from .chroma_manager import collection_name_for_model, delete_collection


@click.command()
@click.option("--model", "-m", default=None, help="Embedding model to use")
@click.option("--force", is_flag=True, help="Regenerate even if collection exists.")
@click.pass_context
def generate(ctx, model: str | None, force: bool):
    t0 = time.monotonic()
    try:
        build_embeddings(model_name=model, force=force)
        elapsed = time.monotonic() - t0
        click.echo(click.style(f"Embeddings generated successfully in {elapsed:.1f}s", fg="green"))
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        raise


@click.command()
@click.argument("query")
@click.option("--top-k", "-k", default=5, type=int, help="Number of results to return")
@click.option("--model", "-m", default=None, help="Model to use for query encoding")
@click.pass_context
def query(ctx, query: str, top_k: int, model: str | None):
    from .builder import EmbeddingBuilder

    builder = EmbeddingBuilder(embedding_model=default_embedding_model(model))

    try:
        results = builder.query_chroma(query, top_k=top_k)
        click.echo(f"\n{'=' * 60}")
        click.echo(click.style(f"Query: {query}", fg="cyan", bold=True))
        click.echo(f"{'=' * 60}\n")

        for i, result in enumerate(results, 1):
            click.echo(click.style(f"[{i}] Score: {1 - result['distance']:.3f}", fg="yellow"))
            click.echo(f"    File: {result['metadata'].get('filename', 'unknown')}")
            click.echo(f"    Tradition: {result['metadata'].get('tradition', 'unknown')}")
            click.echo(f"    Text: {result['document'][:200]}...")
            click.echo()
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)


@click.command("remove")
@click.option("--model", "-m", default=None, help="Model whose collection should be deleted")
@click.option("--yes", is_flag=True, help="Skip confirmation")
@click.pass_context
def delete_chroma_collection(ctx, model: str | None, yes: bool):
    model_name = default_embedding_model(model)
    collection = collection_name_for_model(model_name)

    if not yes:
        click.confirm(f"Delete collection '{collection}' for model '{model_name}'?", abort=True)

    import chromadb

    chroma_dir = settings.chroma_dir
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    try:
        deleted = delete_collection(client, collection)
        if deleted:
            click.echo(click.style(f"Collection '{collection}' deleted", fg="green"))
        else:
            click.echo(click.style(f"Collection '{collection}' does not exist", fg="yellow"))
    except Exception as e:
        click.echo(click.style(f"Error deleting collection: {e}", fg="red"))


