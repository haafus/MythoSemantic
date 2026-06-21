import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_json_optional(path: Path) -> Any:
    """Load JSON from a file that may not exist or may be corrupt (caches, checkpoints)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
        logger.debug(f"Could not load {path}: {e}")
        return None


def save_json(path: Path, data: Any, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, **kwargs)
