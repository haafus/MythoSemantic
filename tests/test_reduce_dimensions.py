import numpy as np
import pytest

from projection.utils import reduce_dimensions


@pytest.fixture
def random_embeddings():
    rng = np.random.default_rng(42)
    return rng.standard_normal((50, 128))


@pytest.fixture
def small_embeddings():
    rng = np.random.default_rng(42)
    return rng.standard_normal((10, 32))


class TestReduceDimensionsUMAP:

    def test_basic_2d(self, random_embeddings):
        result = reduce_dimensions(random_embeddings, n_components=2)
        assert result.shape == (50, 2)

    def test_custom_params(self, random_embeddings):
        result = reduce_dimensions(random_embeddings, n_components=2, n_neighbors=5, min_dist=0.5)
        assert result.shape == (50, 2)

    def test_no_nans(self, random_embeddings):
        result = reduce_dimensions(random_embeddings, n_components=2)
        assert not np.isnan(result).any()

    def test_with_normalization(self, random_embeddings):
        result = reduce_dimensions(random_embeddings, n_components=2, normalize=True)
        assert result.shape == (50, 2)


class TestEdgeCases:
    def test_empty_array(self):
        result = reduce_dimensions(np.array([]).reshape(0, 10))
        assert len(result) == 0

    def test_two_points_returns_zeros(self):
        data = np.random.default_rng(42).standard_normal((2, 10))
        result = reduce_dimensions(data, n_components=2)
        assert result.shape == (2, 2)
        assert np.allclose(result, 0)

    def test_one_point_returns_zeros(self):
        data = np.random.default_rng(42).standard_normal((1, 10))
        result = reduce_dimensions(data, n_components=2)
        assert result.shape == (1, 2)
        assert np.allclose(result, 0)

    def test_normalize_does_not_change_shape(self, random_embeddings):
        result_norm = reduce_dimensions(random_embeddings, n_components=2, normalize=True)
        result_raw = reduce_dimensions(random_embeddings, n_components=2, normalize=False)
        assert result_norm.shape == result_raw.shape
