import json
import logging
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import LabelEncoder, Normalizer

from settings import settings

from .utils import reduce_dimensions

MAX_TEXT_PREVIEW_LEN = 200

logger = logging.getLogger(__name__)


def _reduce_dimensions_safe(
    embeddings: np.ndarray, n_components: int = 2,
) -> np.ndarray | None:
    cfg = settings.projection
    try:
        return reduce_dimensions(
            embeddings, n_components=n_components,
            n_neighbors=cfg.umap_n_neighbors, min_dist=cfg.umap_min_dist,
        )
    except Exception:
        logger.exception("Dimension reduction failed")
        return None


# ---------------------------------------------------------------------------
# Embedding transforms (scatter methods differ only in how they transform
# the raw embeddings before UMAP reduction)
# ---------------------------------------------------------------------------

def _identity(data: list[dict], embeddings: np.ndarray) -> np.ndarray:
    return embeddings


def _compute_tradition_residuals(data: list[dict], embeddings: np.ndarray) -> np.ndarray:
    traditions = [item.get("tradition", "unknown") for item in data]
    unique_traditions = set(traditions)
    centroids = {}
    for trad in unique_traditions:
        mask = [i for i, t in enumerate(traditions) if t == trad]
        centroids[trad] = embeddings[mask].mean(axis=0)
    residuals = np.empty_like(embeddings)
    for i, trad in enumerate(traditions):
        residuals[i] = embeddings[i] - centroids[trad]
    return residuals


def _residual_normalized(data: list[dict], embeddings: np.ndarray) -> np.ndarray:
    residuals = _compute_tradition_residuals(data, embeddings)
    return Normalizer(norm="l2").fit_transform(residuals)


def _concept_erasure(data: list[dict], embeddings: np.ndarray, max_iters: int = 35) -> np.ndarray:
    """INLP: iteratively project out tradition-predictive directions (Ravfogel et al., ACL 2020)."""
    traditions = [item.get("tradition", "unknown") for item in data]
    le = LabelEncoder()
    y = le.fit_transform(traditions)
    n_classes = len(le.classes_)
    chance = 1.0 / n_classes

    X = embeddings.astype(np.float64)
    d = X.shape[1]
    P = np.eye(d, dtype=np.float64)

    for i in range(max_iters):
        X_proj = X @ P
        clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0)
        clf.fit(X_proj, y)
        acc = clf.score(X_proj, y)
        print(f"  INLP iteration {i + 1}/{max_iters}: accuracy = {acc:.3f} (chance = {chance:.3f})", end="\r", flush=True)

        if acc < chance + 0.03:
            print()
            logger.info(f"  INLP converged after {i + 1} iterations")
            break

        W = clf.coef_
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        basis = Vt[S > 1e-10]
        rowspace_proj = basis.T @ basis
        P = P @ (np.eye(d, dtype=np.float64) - rowspace_proj)
    else:
        print()

    return (X @ P).astype(embeddings.dtype)


SCATTER_TRANSFORMS = {
    "umap": _identity,
    "residual_umap": _compute_tradition_residuals,
    "residual_normalized_umap": _residual_normalized,
    "rlace_umap": _concept_erasure,
    "motif_umap": _identity,
}


# ---------------------------------------------------------------------------
# Generators by chart type
# ---------------------------------------------------------------------------

def generate_scatter(
    data: list[dict],
    embeddings: np.ndarray,
    output_path: Path,
    model_name: str | None = None,
    transform=_identity,
) -> None:
    transformed = transform(data, embeddings)
    embedding_2d = _reduce_dimensions_safe(transformed, n_components=2)
    if embedding_2d is None:
        return

    points = []
    for i, item in enumerate(data):
        text_preview = item.get("text", "")[:MAX_TEXT_PREVIEW_LEN]
        if len(item.get("text", "")) > MAX_TEXT_PREVIEW_LEN:
            text_preview += "..."
        points.append({
            "id": item["id"],
            "tradition": item.get("tradition", "unknown"),
            "chunk_index": item["chunk_index"],
            "text": text_preview,
            "x": round(float(embedding_2d[i, 0]), 6),
            "y": round(float(embedding_2d[i, 1]), 6),
        })

    output_path.write_text(
        json.dumps({"model": model_name or "", "points": points}),
        encoding="utf-8",
    )
    logger.info(f"Scatter saved: {output_path}")


def generate_heatmap(
    data: list[dict],
    embeddings: np.ndarray,
    output_path: Path,
    model_name: str | None = None,
    **_kwargs,
) -> None:
    traditions_data: dict[str, list] = {}
    for item, emb in zip(data, embeddings, strict=True):
        trad = item.get("tradition", "unknown")
        if trad not in traditions_data:
            traditions_data[trad] = []
        traditions_data[trad].append(emb)

    centroids = {}
    for trad, embs in traditions_data.items():
        centroids[trad] = np.mean(embs, axis=0)

    trad_list = sorted(centroids.keys())
    centroid_matrix = np.array([centroids[trad] for trad in trad_list])
    distance_matrix = cosine_distances(centroid_matrix, centroid_matrix)

    result = {
        "model": model_name or "",
        "traditions": trad_list,
        "distances": [[round(float(v), 6) for v in row] for row in distance_matrix],
    }

    output_path.write_text(json.dumps(result), encoding="utf-8")
    logger.info(f"Heatmap saved: {output_path}")


def generate_distribution(
    data: list[dict],
    embeddings: np.ndarray,
    output_path: Path,
    model_name: str | None = None,
    **_kwargs,
) -> None:
    tradition_counts: dict[str, int] = {}
    tradition_docs: dict[str, set] = {}
    for item in data:
        trad = item.get("tradition", "unknown")
        tradition_counts[trad] = tradition_counts.get(trad, 0) + 1
        tradition_docs.setdefault(trad, set()).add(item["id"])

    sorted_traditions = sorted(tradition_counts.items(), key=lambda x: -x[1])
    total_chunks = sum(c for _, c in sorted_traditions)

    traditions = []
    for name, count in sorted_traditions:
        traditions.append({
            "name": name,
            "chunks": count,
            "percentage": round(count / total_chunks * 100, 2) if total_chunks else 0,
            "doc_count": len(tradition_docs.get(name, set())),
        })

    result = {"model": model_name or "", "total_chunks": total_chunks, "traditions": traditions}

    output_path.write_text(json.dumps(result), encoding="utf-8")
    logger.info(f"Distribution saved: {output_path}")


CHART_GENERATORS = {
    "scatter": generate_scatter,
    "heatmap": generate_heatmap,
    "distribution": generate_distribution,
}
