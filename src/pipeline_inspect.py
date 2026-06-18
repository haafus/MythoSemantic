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
    info = corpus_status(settings)
    corpus_dir: Path = info["corpus_dir"]
    known: set[str] = info["known_paths"]

    orphans = []
    for txt in corpus_dir.rglob("*.txt"):
        if str(txt.resolve()) not in known:
            orphans.append((txt, file_size(txt)))
    return orphans


# ---------------------------------------------------------------------------
# Chroma inspection
# ---------------------------------------------------------------------------

def chroma_status(settings) -> dict[str, Any]:
    chroma_path = Path(settings.chroma_dir)
    result: dict[str, Any] = {
        "exists": chroma_path.exists(),
        "total_size": dir_size(chroma_path),
        "collections": [],
    }
    if not chroma_path.exists():
        return result

    try:
        import chromadb
        from embedding.chroma_manager import is_model_collection_name

        client = chromadb.PersistentClient(path=str(chroma_path))
        for col in client.list_collections():
            if is_model_collection_name(col.name):
                model_name = (col.metadata or {}).get("model", col.name)
                result["collections"].append({
                    "name": col.name,
                    "model": model_name,
                    "count": col.count(),
                })
    except Exception as e:
        result["error"] = str(e)

    return result


def chroma_orphan_collections(settings) -> list[dict[str, Any]]:
    from embedding.chroma_manager import collection_name_for_model
    from model_registry import list_embedding_aliases

    info = chroma_status(settings)
    if not info["exists"]:
        return []

    aliases = list_embedding_aliases()
    known_names = {collection_name_for_model(full_name) for full_name in aliases.values()}

    return [c for c in info["collections"] if c["name"] not in known_names]


def chroma_orphan_chunks(settings) -> list[dict[str, Any]]:
    from corpus.corpus_iterator import normalize_catalog_id

    chroma_path = Path(settings.chroma_dir)
    if not chroma_path.exists():
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
    orphan_collection_names = {c["name"] for c in chroma_orphan_collections(settings)}

    results = []
    try:
        import chromadb
        from embedding.chroma_manager import is_model_collection_name

        client = chromadb.PersistentClient(path=str(chroma_path))
        for col in client.list_collections():
            if not is_model_collection_name(col.name):
                continue
            if col.name in orphan_collection_names:
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
# Analysis inspection
# ---------------------------------------------------------------------------

PROJECTION_PLOTS = [
    "umap_2d_traditions.html",
    "residual_umap_2d.html",
    "residual_normalized_umap_2d.html",
    "rlace_umap_2d.html",
    "distance_heatmap.html",
    "tradition_distribution.html",
]


def analysis_status(settings) -> dict[str, Any]:
    analysis_dir = Path(settings.analysis_dir)
    result: dict[str, Any] = {
        "exists": analysis_dir.exists(),
        "total_size": dir_size(analysis_dir),
        "models": [],
    }
    if not analysis_dir.exists():
        return result

    for model_dir in sorted(d for d in analysis_dir.iterdir() if d.is_dir()):
        existing = [p for p in PROJECTION_PLOTS if (model_dir / p).exists()]
        result["models"].append({
            "name": model_dir.name,
            "path": model_dir,
            "plots_done": len(existing),
            "plots_total": len(PROJECTION_PLOTS),
            "size": dir_size(model_dir),
        })

    return result


def analysis_orphans(settings) -> list[dict[str, Any]]:
    info = analysis_status(settings)
    if not info["exists"]:
        return []

    chroma_info = chroma_status(settings)
    if not chroma_info["exists"] or not chroma_info["collections"]:
        return []

    known_dirs = {settings.safe_model_name(c["model"]) for c in chroma_info["collections"]}

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
    """Find orphan graph items in graphs_dir.

    New format: <text_id>/characters.html (dirs matched by text_id).
    Legacy format: web_*-characters.html (flat files, no corpus match possible).
    """
    from corpus.corpus_iterator import normalize_catalog_id

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

    orphans = []
    for item in graphs_dir.iterdir():
        if item.is_dir():
            if item.name not in known_text_ids:
                orphans.append((item, dir_size(item)))

    return orphans
