from fastapi.testclient import TestClient

from server.run_server import create_app

client = TestClient(create_app())


class TestSPAFallback:
    def test_spa_serves_index(self):
        response = client.get("/nonexistent/page")
        assert response.status_code == 200

    def test_missing_index_returns_404(self, tmp_path, monkeypatch):
        from settings import settings

        monkeypatch.setattr(settings, "web_root", tmp_path / "web")
        test_client = TestClient(create_app())
        response = test_client.get("/nonexistent/page")
        assert response.status_code == 404


class TestModelsEndpoint:
    def test_list_models(self):
        from unittest.mock import patch

        with patch("server.api.models.ChromaStore") as MockStore:
            MockStore.return_value.get_available_models.return_value = []
            response = client.get("/api/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)


class TestCorpusCatalog:
    def test_catalog_returns_list(self):
        response = client.get("/api/corpus/catalog")
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert "total" in data
        assert isinstance(data["documents"], list)
        assert data["total"] == len(data["documents"])

    def test_catalog_ignores_unknown_params(self):
        response = client.get("/api/corpus/catalog?source=anything")
        assert response.status_code == 200


class TestGeographyEndpoint:
    def test_traditions(self):
        response = client.get("/api/geography/traditions")
        assert response.status_code == 200
        data = response.json()
        assert "traditions" in data
        assert "total" in data


class TestCorpusDocumentEndpoint:
    def test_missing_params(self):
        response = client.get("/api/corpus/documents")
        assert response.status_code == 422

    def test_nonexistent_document(self):
        response = client.get(
            "/api/corpus/documents",
            params={
                "id": "nonexistent_xyz",
                "major_tradition": "none",
                "tradition": "none",
            },
        )
        assert response.status_code in (403, 404)


class TestSimilarityEndpoints:
    def test_search_validation(self):
        response = client.post("/api/similarity/search", json={"query": "", "model": "m"})
        assert response.status_code == 422

    def test_search_job_not_found(self):
        response = client.get("/api/similarity/search/jobs/nonexistent_id")
        assert response.status_code == 404

    def test_projection_not_found(self):
        response = client.get("/api/similarity/projections/fake_model/umap")
        assert response.status_code == 404
