import logging
from typing import Any

import numpy as np
from sklearn.preprocessing import Normalizer

logger = logging.getLogger(__name__)


def reduce_dimensions(
    embeddings: np.ndarray,
    n_components: int = 2,
    normalize: bool = False,
    **kwargs: Any,
) -> np.ndarray:
    if len(embeddings) == 0:
        return np.array([])

    if len(embeddings) < 3:
        logger.warning(f"Too few points ({len(embeddings)}) for dimensionality reduction")
        return np.zeros((len(embeddings), n_components))

    data = Normalizer(norm="l2").fit_transform(embeddings) if normalize else embeddings

    import umap

    random_state = kwargs.get("random_state", 42)
    n_neighbors = kwargs.get("n_neighbors", min(15, len(data) - 1))
    result: np.ndarray = umap.UMAP(
        n_components=n_components,
        n_neighbors=max(2, n_neighbors),
        min_dist=kwargs.get("min_dist", 0.1),
        metric=kwargs.get("metric", "cosine"),
        random_state=random_state,
        n_jobs=1,
    ).fit_transform(data)
    return result
