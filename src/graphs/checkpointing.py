from pathlib import Path

from json_utils import load_json, save_json

CHECKPOINT_FILE = "checkpoint.json"


def load_checkpoint(book_out_dir: Path) -> dict | None:
    data = load_json(book_out_dir / CHECKPOINT_FILE)
    if isinstance(data, dict) and isinstance(data.get("next_chunk"), int):
        return data
    return None


def save_checkpoint(book_out_dir: Path, next_chunk: int, results: dict) -> None:
    save_json(book_out_dir / CHECKPOINT_FILE, {"next_chunk": next_chunk, **results})


def clear_checkpoint(book_out_dir: Path) -> None:
    (book_out_dir / CHECKPOINT_FILE).unlink(missing_ok=True)
