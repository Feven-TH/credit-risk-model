from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    """Verify the API is reachable"""
    response = client.get("/docs") 
    assert response.status_code == 200

def test_prediction_endpoint():
    """Verify the model returns a valid JSON structure"""
    payload = {
        "txn_count": 10,
        "amount_sum": 100,
        "amount_mean": 10,
        "amount_std": 2,
        "recency_days": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "is_high_risk" in response.json()
    assert "probability" in response.json()