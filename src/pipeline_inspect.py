"""Shared logic for `mytho status` and `mytho clean`."""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Size helpers
# ---------------------------------------------------------------------------

def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def format_size(nbytes: int) -> str:
    if nbytes < 1024:
        return f"{nbytes} B"
    for unit in ("KB", "MB", "GB"):
        nbytes /= 1024
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:.1f} {unit}"
    return f"{nbytes:.1f} GB"


# ---------------------------------------------------------------------------
# Corpus inspection
# ---------------------------------------------------------------------------

def corpus_status(settings) -> dict[str, Any]:
    config_path = settings.corpus_config_file
    config_count = 0
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                config_count = len(json.load(f))
        except Exception:
            pass

    meta_path = settings.corpus_metadata_path
    meta_entries: list[dict] = []
    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta_entries = json.load(f)
        except Exception:
            pass

    known_paths: set[str] = set()
    corpus_dir = Path(settings.corpus_dir)
    for entry in meta_entries:
        rel = entry.get("path")
        if rel:
            known_paths.add(str((corpus_dir / rel).resolve()))

    total_size = dir_size(corpus_dir)

    return {
        "config_count": config_count,
        "built_count": len(meta_entries),
        "missing_count": max(0, config_count - len(meta_entries)),
        "known_paths": known_paths,
        "corpus_dir": corpus_dir,
        "total_size": total_size,
    }


def corpus_orphans(settings) -> list[tuple[Path, int]]:
    if not settings.corpus_metadata_path.exists():
        return []

    info = corpus_status(settings)
    corpus_dir: Path = info["corpus_dir"]
    known: set[str] = info["known_paths"]

    orphans = []
    for txt in corpus_dir.rglob("*.txt"):
        if str(txt.resolve()) not in known:
            orphans.append((txt, file_size(txt)))
    return orphans


# ---------------------------------------------------------------------------
# Embeddings inspection
# ---------------------------------------------------------------------------

def embeddings_status(settings) -> dict[str, Any]:
    chroma_path = Path(settings.embeddings_dir)
    result: dict[str, Any] = {
        "exists": chroma_path.exists(),
        "total_size": dir_size(chroma_path),
        "collections": [],
    }
    if not chroma_path.exists():
        return result

    try:
        from embeddings import chroma_manager

        for col in chroma_manager.list_collections():
            model_name = (col.metadata or {}).get("model", col.name)
            result["collections"].append({
                "name": col.name,
                "model": model_name,
                "count": col.count(),
            })
    except Exception as e:
        result["error"] = str(e)

    return result


def embeddings_orphan_collections(settings) -> list[dict[str, Any]]:
    from model_registry import list_embedding_aliases

    info = embeddings_status(settings)
    if not info["exists"]:
        return []

    known_models = set(list_embedding_aliases().values())

    return [c for c in info["collections"] if c["model"] not in known_models]


def embeddings_orphan_chunks(settings, *, skip_collections: set[str] | None = None) -> list[dict[str, Any]]:
    from corpus.utils import normalize_catalog_id

    chroma_path = Path(settings.embeddings_dir)
    if not chroma_path.exists():
        return []

    if skip_collections is None:
        skip_collections = {c["name"] for c in embeddings_orphan_collections(settings)}

    meta_path = settings.corpus_metadata_path
    if not meta_path.exists():
        return []

    try:
        with open(meta_path, encoding="utf-8") as f:
            meta_entries = json.load(f)
    except Exception:
        return []

    known_text_ids = {normalize_catalog_id(e["id"]) for e in meta_entries if e.get("id")}
    results = []
    try:
        from embeddings import chroma_manager

        for col in chroma_manager.list_collections():
            if col.name in skip_collections:
                continue
            all_meta = col.get(include=["metadatas"])
            orphan_ids = []
            for doc_id, meta in zip(all_meta["ids"], all_meta["metadatas"]):
                text_id = (meta or {}).get("text_id")
                if text_id and text_id not in known_text_ids:
                    orphan_ids.append(doc_id)
            if orphan_ids:
                model = (col.metadata or {}).get("model", col.name)
                results.append({
                    "collection": col.name,
                    "model": model,
                    "orphan_ids": orphan_ids,
                    "total_count": col.count(),
                })
    except Exception:
        logger.exception("Failed to scan chroma for orphan chunks")

    return results


# ---------------------------------------------------------------------------
# Projections inspection
# ---------------------------------------------------------------------------

PROJECTION_PLOTS = [
    "umap_2d_traditions.html",
    "residual_umap_2d.html",
    "residual_normalized_umap_2d.html",
    "rlace_umap_2d.html",
    "distance_heatmap.html",
    "tradition_distribution.html",
]


def projections_status(settings) -> dict[str, Any]:
    proj_dir = Path(settings.projections_dir)
    result: dict[str, Any] = {
        "exists": proj_dir.exists(),
        "total_size": dir_size(proj_dir),
        "models": [],
    }
    if not proj_dir.exists():
        return result

    for model_dir in sorted(d for d in proj_dir.iterdir() if d.is_dir()):
        existing = [p for p in PROJECTION_PLOTS if (model_dir / p).exists()]
        result["models"].append({
            "name": model_dir.name,
            "path": model_dir,
            "plots_done": len(existing),
            "plots_total": len(PROJECTION_PLOTS),
            "size": dir_size(model_dir),
        })

    return result


def projections_orphans(settings) -> list[dict[str, Any]]:
    info = projections_status(settings)
    if not info["exists"] or not info["models"]:
        return []

    emb_info = embeddings_status(settings)
    if "error" in emb_info:
        return []

    from model_registry import model_to_key
    known_dirs = {model_to_key(c["model"]) for c in emb_info["collections"]}

    return [m for m in info["models"] if m["name"] not in known_dirs]


# ---------------------------------------------------------------------------
# Graphs inspection
# ---------------------------------------------------------------------------

def graphs_status(settings) -> dict[str, Any]:
    graphs_dir = Path(settings.graphs_dir)
    result: dict[str, Any] = {
        "exists": graphs_dir.exists(),
        "total_size": dir_size(graphs_dir),
        "count": 0,
    }
    if not graphs_dir.exists():
        return result

    subdirs = [d for d in graphs_dir.iterdir() if d.is_dir()]
    html_files = list(graphs_dir.glob("*.html"))
    result["count"] = len(subdirs) + len(html_files)
    return result


def graphs_orphans(settings) -> list[tuple[Path, int]]:
    """Find orphan graph directories in graphs_dir.

    Matches subdirectory names against normalized text_ids from corpus.
    If corpus metadata is missing, returns [] (can't determine orphans).
    """
    from corpus.utils import normalize_catalog_id

    graphs_dir = Path(settings.graphs_dir)
    if not graphs_dir.exists():
        return []

    meta_path = settings.corpus_metadata_path
    if not meta_path.exists():
        return []

    try:
        with open(meta_path, encoding="utf-8") as f:
            meta_entries = json.load(f)
    except Exception:
        return []

    known_text_ids = {normalize_catalog_id(e["id"]) for e in meta_entries if e.get("id")}

    subdirs = [d for d in graphs_dir.iterdir() if d.is_dir()]

    orphans = []
    for item in subdirs:
        if item.name not in known_text_ids:
            orphans.append((item, dir_size(item)))

    return orphans
