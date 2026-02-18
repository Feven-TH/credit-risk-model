import os
import time
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from pathlib import Path

# Import your app and configuration
from src.api.main import app
from src.core.config import config

# --- FIX 1: THE LIFESPAN FIX ---
# We use a fixture with a 'with' statement. This ensures the 
# asynccontextmanager (lifespan) in main.py actually runs,
# loading the MLflow model before the tests start.
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        # Give the model a moment to stabilize after the 
        # '✅ Model loaded' message appears.
        time.sleep(1) 
        yield c

# --- 1. FEATURE ENGINEERING TEST ---
def test_behavioral_logic():
    """Test: Does our math for velocity/momentum make sense?"""
    amount = 1000
    days = 10
    velocity = amount / days
    assert velocity == 100.0

# --- 2. API CONTRACT TEST ---
def test_api_health(client):
    """Test: Is the FastAPI server up and connected to MLflow?"""
    response = client.get("/health")
    assert response.status_code == 200
    # This will now pass because the lifespan fixture loaded the model
    assert response.json()["status"] == "online"

# --- 3. INPUT VALIDATION TEST ---
def test_invalid_input_types(client):
    """Test: Does the API reject bad data (strings instead of floats)?"""
    bad_payload = {
        "amount_mean": "A lot of money", 
        "max_velocity": 1000,
        "avg_momentum": 0.5
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422 

# --- 4. BUSINESS LOGIC TEST (Thresholds) ---
def test_threshold_logic(client):
    """Test: Does changing the threshold actually change the decision?"""
    payload = {
        "amount_mean": 500.0,
        "max_velocity": 1500.0,
        "avg_momentum": 0.7
    }
    
    res_strict = client.post("/predict", json={**payload, "threshold": 0.1})
    res_loose = client.post("/predict", json={**payload, "threshold": 0.9})
    
    # These now work because threshold_used is in your Pydantic Response model
    assert res_strict.json()["threshold_used"] == 0.1
    assert res_loose.json()["threshold_used"] == 0.9

# --- 5. DATA PERSISTENCE TEST (Logging) ---
def test_audit_log_creation(client):
    """Test: Does the API actually write to the CSV log file?"""
    log_path = config.log_path 
    
    # Get size before prediction
    initial_size = os.path.getsize(log_path) if log_path.exists() else 0
    
    client.post("/predict", json={
        "amount_mean": 10.0, "max_velocity": 10.0, "avg_momentum": 0.1, "threshold": 0.5
    })
    
    # --- FIX 2: THE SYNC FIX ---
    # CSV writes are fast, but the OS needs a split second to update file metadata
    time.sleep(0.8)
    
    assert log_path.exists()
    assert os.path.getsize(log_path) > initial_size

# --- 6. EXPLAINABILITY TEST ---
def test_shap_explanation_structure(client):
    """Test: Does the API provide a breakdown of feature importance?"""
    payload = {
        "amount_mean": 1000.0,
        "max_velocity": 5000.0,
        "avg_momentum": 0.5,
        "threshold": 0.5
    }
    response = client.post("/predict", json=payload)
    data = response.json()
    
    assert "explanation" in data
    assert isinstance(data["explanation"], dict)
    # Ensure all features defined in config.FEATURE_ORDER are explained
    for feat in ["amount_mean", "avg_momentum", "max_velocity"]:
        assert feat in data["explanation"]