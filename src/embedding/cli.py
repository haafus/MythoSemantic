import time

import click

from .build_embeddings import build_embeddings


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
