from fastapi.testclient import TestClient

from app.main import app


def test_health_without_model_is_ok():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is False


def test_ready_without_model_is_unavailable():
    response = TestClient(app).get("/ready")
    assert response.status_code == 503

