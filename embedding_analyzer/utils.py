import json
import os
from typing import Dict, Tuple, Optional

import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def load_corpus_metadata(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return {str(item["id"]): item["tradition"] for item in metadata}


def reduce_dimensions(
        embeddings: np.ndarray,
        method: str = 'umap',
        n_components: int = 2,
        **kwargs
) -> np.ndarray:
    if method == 'umap':
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=kwargs.get('n_neighbors', 15),
            min_dist=kwargs.get('min_dist', 0.1),
            metric=kwargs.get('metric', 'cosine'),
            random_state=kwargs.get('random_state', 42),
            n_jobs=-1
        )
        return reducer.fit_transform(embeddings)

    elif method == 'pca':
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        reducer = PCA(
            n_components=n_components,
            random_state=kwargs.get('random_state', 42)
        )
        return reducer.fit_transform(embeddings_scaled)

    elif method == 'tsne':
        reducer = TSNE(
            n_components=n_components,
            perplexity=kwargs.get('perplexity', min(30, len(embeddings) - 1)),
            random_state=kwargs.get('random_state', 42),
            max_iter=kwargs.get('max_iter', kwargs.get('n_iter', 1000))
        )
        return reducer.fit_transform(embeddings)

    else:
        raise ValueError(f"Unknown method: {method}")


def safe_numpy_array(emb) -> np.ndarray:
    return np.array(emb) if isinstance(emb, list) else emb
