"""Integration smoke test: downloads and loads every registered embedding model.

Heavy (gigabytes of model weights) — auto-skipped when ML dependencies are absent.
Run explicitly: pytest tests/test_embedding_models_integration.py
"""

import time
import unittest

import numpy as np
import pytest

from model_registry import resolve_embedding_model
from settings import settings

SentenceTransformer = pytest.importorskip("sentence_transformers").SentenceTransformer

try:
    import torch
except ImportError:
    torch = None

MODELS = [resolve_embedding_model(m) for m in settings.embedding.models]


class TestEmbeddingModels(unittest.TestCase):
    """Test class for embedding models"""

    results: dict[str, dict] = {}

    @classmethod
    def setUpClass(cls):
        if torch and torch.cuda.is_available():
            print(f"GPU available: {torch.cuda.get_device_name(0)}")
        elif torch:
            print("GPU not detected; using CPU (tests will be slower)")
        else:
            print("PyTorch is not installed; skipping GPU check")

        cls.results = {}

    def test_all_models_can_load_and_encode(self):
        failed_models = []

        for model_name in MODELS:
            print(f"Testing model: {model_name}")

            start_time = time.time()

            try:
                device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
                model = SentenceTransformer(model_name, device=device)

                load_time = time.time() - start_time
                print(f"  Model loaded in {load_time:.2f} s")

                try:
                    test_texts = ["Hello world", "Hello friend", "Bonjour le monde"]

                    encode_start = time.time()
                    embeddings = model.encode(test_texts, normalize_embeddings=True, show_progress_bar=False)
                    encode_time = time.time() - encode_start

                    actual_dim = embeddings.shape[1] if len(embeddings.shape) > 1 else len(embeddings)
                    expected_dim = model.get_sentence_embedding_dimension()

                    print(f"  Encoding completed in {encode_time:.2f} s, dim={actual_dim}")

                    self.assertEqual(actual_dim, expected_dim)
                    self.assertEqual(len(embeddings), len(test_texts))
                    self.assertFalse(np.isnan(embeddings).any(), "Embeddings contain NaN values")

                    if len(embeddings) >= 2:
                        from sklearn.metrics.pairwise import cosine_similarity

                        sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                        print(f"  Cosine similarity ('Hello world' / 'Hello friend'): {sim:.4f}")

                    self.results[model_name] = {
                        "status": "PASS",
                        "load_time": load_time,
                        "encode_time": encode_time,
                        "dim": actual_dim,
                    }

                except Exception as e:
                    print(f"  Encoding error: {e}")
                    failed_models.append((model_name, f"Encoding error: {e}"))
                    self.results[model_name] = {"status": "FAIL", "error": str(e)}

            except Exception as e:
                print(f"  Model load error: {e}")
                failed_models.append((model_name, f"Load error: {e}"))
                self.results[model_name] = {"status": "FAIL", "error": str(e)}

            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(1)

        if failed_models:
            self.fail(f"\n{len(failed_models)} models failed testing")
