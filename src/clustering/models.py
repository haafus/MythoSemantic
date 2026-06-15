import logging
from typing import Any

import numpy as np
from sklearn.cluster import HDBSCAN, OPTICS, Birch, KMeans, MeanShift, SpectralClustering, estimate_bandwidth
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import Normalizer

logger = logging.getLogger(__name__)


class BaseClusteringModel:
    def __init__(
        self,
        name: str,
        min_samples_required: int = 2,
        needs_dim_reduction: bool = False,
        n_components: int = 15,
        **params,
    ):
        self.name = name
        self.min_samples_required = min_samples_required
        self.needs_dim_reduction = needs_dim_reduction
        self.n_components = n_components
        self.params = params
        self.model: Any = None
        self.scaler = Normalizer(norm="l2")
        self.processed_embeddings: np.ndarray | None = None

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        if len(embeddings) < self.min_samples_required:
            self.processed_embeddings = embeddings
            return np.zeros(len(embeddings), dtype=int)

        data_ready = self.scaler.fit_transform(embeddings)

        if self.needs_dim_reduction and data_ready.shape[1] > self.n_components:
            data_ready = self._reduce_dimensions(data_ready)

        self.processed_embeddings = data_ready

        return self._do_fit_predict(data_ready)

    def _reduce_dimensions(self, data: np.ndarray) -> np.ndarray:
        from projection.utils import reduce_dimensions

        actual_components = max(2, min(self.n_components, len(data) - 2))
        return reduce_dimensions(
            data,
            n_components=actual_components,
            metric="cosine",
            fallback_on_error=True,
        )

    def _do_fit_predict(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class KMeansClustering(BaseClusteringModel):
    def __init__(self, n_clusters: int = 2, **kwargs):
        super().__init__("kmeans", **kwargs)
        self.n_clusters = n_clusters

    def _do_fit_predict(self, data: np.ndarray) -> np.ndarray:
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        result: np.ndarray = self.model.fit_predict(data)
        return result


class HDBSCANClustering(BaseClusteringModel):
    def __init__(self, min_cluster_size: int = 200, **kwargs):
        super().__init__("hdbscan", needs_dim_reduction=True, **kwargs)
        self.min_cluster_size = min_cluster_size

    def _do_fit_predict(self, data: np.ndarray) -> np.ndarray:
        self.model = HDBSCAN(min_cluster_size=self.min_cluster_size, metric="euclidean")
        result: np.ndarray = self.model.fit_predict(data)
        return result


class SpectralClusteringModel(BaseClusteringModel):
    def __init__(self, n_clusters: int = 2, **kwargs):
        super().__init__("spectral", min_samples_required=3, **kwargs)
        self.n_clusters = n_clusters

    def _do_fit_predict(self, data: np.ndarray) -> np.ndarray:
        self.model = SpectralClustering(n_clusters=self.n_clusters, random_state=42, affinity="nearest_neighbors")
        result: np.ndarray = self.model.fit_predict(data)
        return result


class BirchClustering(BaseClusteringModel):
    def __init__(self, n_clusters: int = 2, threshold: float = 0.5, **kwargs):
        super().__init__("birch", **kwargs)
        self.n_clusters = n_clusters
        self.threshold = threshold

    def _do_fit_predict(self, data: np.ndarray) -> np.ndarray:
        self.model = Birch(n_clusters=self.n_clusters, threshold=self.threshold)
        result: np.ndarray = self.model.fit_predict(data)
        return result


class GMMClustering(BaseClusteringModel):
    def __init__(self, n_components: int = 2, **kwargs):
        super().__init__("gmm", needs_dim_reduction=True, **kwargs)
        self.n_components = n_components

    def _do_fit_predict(self, data: np.ndarray) -> np.ndarray:
        self.model = GaussianMixture(n_components=self.n_components, random_state=42)
        result: np.ndarray = self.model.fit_predict(data)
        return result


class MeanShiftClustering(BaseClusteringModel):
    def __init__(self, **kwargs):
        super().__init__("meanshift", needs_dim_reduction=True, **kwargs)

    def _do_fit_predict(self, data: np.ndarray) -> np.ndarray:

        bandwidth = estimate_bandwidth(data, quantile=0.05, n_samples=2000, random_state=42)

        if bandwidth <= 0:
            bandwidth = 1.0
            logger.warning("Estimated bandwidth <= 0, using 1.0")

        logger.info(f"MeanShift will use bandwidth: {bandwidth:.4f}")

        self.model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        result: np.ndarray = self.model.fit_predict(data)
        return result


class OPTICSClustering(BaseClusteringModel):
    def __init__(self, min_samples: int = 200, **kwargs):
        super().__init__("optics", needs_dim_reduction=True, **kwargs)
        self.min_samples = min_samples

    def _do_fit_predict(self, data: np.ndarray) -> np.ndarray:
        self.model = OPTICS(min_samples=self.min_samples)
        result: np.ndarray = self.model.fit_predict(data)
        return result


def get_clustering_model(model_name: str, **params) -> BaseClusteringModel:
    models = {
        "kmeans": KMeansClustering,
        "hdbscan": HDBSCANClustering,
        "spectral": SpectralClusteringModel,
        "birch": BirchClustering,
        "gmm": GMMClustering,
        "meanshift": MeanShiftClustering,
        "optics": OPTICSClustering,
    }

    if model_name not in models:
        available = list(models.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")

    model_cls = models[model_name]
    result: BaseClusteringModel = model_cls(**params)
    return result


def list_available_models() -> list[str]:
    return ["kmeans", "hdbscan", "spectral", "birch", "gmm", "meanshift", "optics"]
