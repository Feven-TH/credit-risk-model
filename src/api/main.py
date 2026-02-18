import sys
import datetime
import csv
import numpy as np
import pandas as pd
import mlflow.pyfunc
import shap
from pathlib import Path
from fastapi import FastAPI, HTTPException
from typing import Dict, Any
from src.core.config import config
from src.api.pydantic_models import PredictionRequest, PredictionResponse
from contextlib import asynccontextmanager



# 1. Setup Tracking & App
mlflow.set_tracking_uri(config.db_uri)
# app = FastAPI(title="Bati Bank Credit Risk API")

# 2. Global Model & Explainer Objects
MODEL_URI = f"models:/{config.MODEL_NAME}@{config.MODEL_ALIAS}" 
model = None
explainer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events."""
    global model, explainer
    try:
        print(f"📡 Connecting to MLflow at: {config.db_uri}")
        mlflow.set_tracking_uri(config.db_uri)
        
        # Load assets
        mlflow_model = mlflow.pyfunc.load_model(MODEL_URI)
        raw_model = mlflow_model._model_impl.sklearn_model
        
        model = mlflow_model
        explainer = shap.TreeExplainer(raw_model)
        print("✅ Model & SHAP Explainer loaded successfully!")
    except Exception as e:
        print(f"❌ Initialization Error: {e}")
    
    yield  # The API is now "online"
    
    # Logic here (if any) for shutdown (e.g., closing DB connections)
    print("🔌 Shutting down...")

# Initialize FastAPI with the lifespan handler
app = FastAPI(title="Bati Bank Credit Risk API", lifespan=lifespan)

# --- ENDPOINT 1: HEALTH CHECK (GET) ---
@app.get("/health")
def health_check():
    try:
        # Check if model is loaded
        if model is None:
            return {"status": "unhealthy", "reason": "model not loaded"}
        return {"status": "online"}
    except Exception as e:
        print(f"HEALTH CHECK ERROR: {e}") # This will show in your terminal
        return {"status": "unhealthy", "detail": str(e)}

# --- ENDPOINT 2: PREDICTION (POST) ---
@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest) -> Dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model unavailable")
    
    try:
        # Prepare Data
        input_dict = data.model_dump()
        features_dict = {k: v for k, v in input_dict.items() if k != 'threshold'}
        input_df = pd.DataFrame([features_dict])
        input_df = input_df[config.FEATURE_ORDER]
        
        # 1. Logic: Probability
        # Some MLflow wrappers return a numpy array, some return a list.
        # We use [0, 1] to get the first row, second column (High Risk probability)
        y_proba = model._model_impl.predict_proba(input_df)
        
        # .item() is the key fix here for the "length-1 array" error
        probability = float(y_proba[0][1].item() if hasattr(y_proba[0][1], "item") else y_proba[0][1])
        
        is_high_risk = 1 if probability >= data.threshold else 0
        
        # 2. Logic: SHAP Explanation
        # 2. Logic: SHAP Explanation
        explanation = None
        if explainer is not None:
            shap_output = explainer.shap_values(input_df)
            
            # 1. Handle Class Selection (Index 1 for High Risk)
            if isinstance(shap_output, list):
                sv = shap_output[1]
            else:
                sv = shap_output
                
            # 2. THE FIX: Flatten the array to ensure it's 1D
            # This converts [[val1, val2, val3]] -> [val1, val2, val3]
            sv_flattened = np.array(sv).flatten()

            # 3. Map to feature names safely
            explanation = {
                feat: round(float(sv_flattened[i]), 4) 
                for i, feat in enumerate(config.FEATURE_ORDER)
            }
        
        # 3. Construct Response
        response = {
            "is_high_risk": is_high_risk,
            "probability": round(probability, 4),
            "threshold_used": data.threshold,
            "explanation": explanation
        }
        
        log_prediction(input_dict, response)
        return response
        
    except Exception as e:
        # This will print the exact line where it fails in your terminal
        import traceback
        print("--- FULL ERROR TRACEBACK ---")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Runtime Error: {str(e)}")

def log_prediction(input_data: dict, response_data: dict) -> None:
    """Saves API interaction to a CSV for auditing."""
    file_exists = config.log_path.exists()
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        **input_data,
        "is_high_risk": response_data["is_high_risk"],
        "probability": response_data["probability"]
    }
    
    with open(config.log_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(log_entry.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)