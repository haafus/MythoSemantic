import logging
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer

logger = logging.getLogger(__name__)

_UMAP_AVAILABLE = None


def _check_umap() -> bool:
    global _UMAP_AVAILABLE
    if _UMAP_AVAILABLE is None:
        try:
            import umap  # noqa: F401

            _UMAP_AVAILABLE = True
        except ImportError:
            _UMAP_AVAILABLE = False
    return _UMAP_AVAILABLE


def reduce_dimensions(
    embeddings: np.ndarray,
    n_components: int = 2,
    normalize: bool = False,
    fallback_on_error: bool = False,
    **kwargs: Any,
) -> np.ndarray:
    if len(embeddings) == 0:
        return np.array([])

    if len(embeddings) < 3:
        logger.warning(f"Too few points ({len(embeddings)}) for dimensionality reduction")
        return np.zeros((len(embeddings), n_components))

    data = Normalizer(norm="l2").fit_transform(embeddings) if normalize else embeddings

    try:
        return _run_umap(data, n_components, **kwargs)
    except Exception:
        if not fallback_on_error:
            raise
        logger.exception("UMAP failed, falling back to PCA")
        return _run_pca(data, n_components, **kwargs)


def _run_umap(data: np.ndarray, n_components: int, **kwargs: Any) -> np.ndarray:
    if not _check_umap():
        raise ImportError("umap-learn is not installed")
    import umap

    random_state = kwargs.get("random_state", 42)
    n_neighbors = kwargs.get("n_neighbors", min(15, len(data) - 1))
    result: np.ndarray = umap.UMAP(
        n_components=n_components,
        n_neighbors=max(2, n_neighbors),
        min_dist=kwargs.get("min_dist", 0.1),
        metric=kwargs.get("metric", "cosine"),
        random_state=random_state,
        n_jobs=-1,
    ).fit_transform(data)
    return result


def _run_pca(data: np.ndarray, n_components: int, **kwargs: Any) -> np.ndarray:
    from sklearn.preprocessing import StandardScaler

    random_state = kwargs.get("random_state", 42)
    actual_components = min(n_components, data.shape[1], len(data) - 1)
    scaled: np.ndarray = StandardScaler().fit_transform(data)
    result: np.ndarray = PCA(n_components=max(1, actual_components), random_state=random_state).fit_transform(scaled)
    return result
