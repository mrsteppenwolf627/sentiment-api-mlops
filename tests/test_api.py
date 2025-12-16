"""
Tests para los endpoints de la API
Usa TestClient de FastAPI para simular requests HTTP
sin necesidad de correr servidor real.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.config import settings

# Test client de FastAPI
client = TestClient(app)

class TestRootEndpoint:
    """Tests del endpoint raíz"""
    def test_root_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs" in data

class TestHealthEndpoint:
    """Tests del health check"""
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

class TestAnalyzeEndpoint:
    """Tests del endpoint de análisis"""
    def test_analyze_positive_text(self):
        response = client.post(
            "/analyze",
            json={"text": "I love this product! It's amazing."}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["sentiment"] == "positive"
        assert data["confidence"] > 0.5
        # Verificar que devuelve la estructura completa
        assert "processing_time_ms" in data
        assert "cost_estimate_usd" in data

    def test_analyze_negative_text(self):
        response = client.post(
            "/analyze",
            json={"text": "This is terrible and awful. I hate it."}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["sentiment"] == "negative"

    def test_analyze_empty_text_fails(self):
        """Test que texto vacío es rechazado (Validación Pydantic)"""
        response = client.post(
            "/analyze",
            json={"text": ""}
        )
        # Pydantic debe devolver error 422 (Unprocessable Entity)
        assert response.status_code == 422

class TestMetricsEndpoint:
    """Tests del endpoint de métricas"""
    def test_metrics_returns_200(self):
        response = client.get("/metrics")
        assert response.status_code == 200
        # Verificar formato Prometheus
        assert "text/plain" in response.headers["content-type"]
        assert "sentiment_api_requests_total" in response.text