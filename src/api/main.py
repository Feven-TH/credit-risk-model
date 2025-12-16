import mlflow.pyfunc
import pandas as pd
import os
from fastapi import FastAPI, HTTPException
from .pydantic_models import PredictionRequest, PredictionResponse
from sklearn.base import BaseEstimator, TransformerMixin

# 1. Custom Transformer must be defined exactly as it was during training
class DataFrameConverter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)

app = FastAPI(title="Credit Risk API")

# 2. Database Path Logic (Docker-friendly)
# Defaults to local path if MLFLOW_TRACKING_URI environment variable isn't set
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///src/mlflow.db")
mlflow.set_tracking_uri(TRACKING_URI)

# 3. Model Configuration
MODEL_NAME = "Credit_Risk_Champion_Model"
MODEL_VERSION = 1
model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"


model = None

try:
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"✅ Successfully loaded model: {MODEL_NAME} v{MODEL_VERSION}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# 4. Health Check Endpoint (Critical for CI/CD and Docker)
@app.get("/health")
def health_check():
    if model is None:
        return {"status": "unhealthy", "error": "Model not loaded"}
    return {"status": "online", "model": MODEL_NAME, "version": MODEL_VERSION}

# 5. Prediction Endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server")
    
    try:
        input_df = pd.DataFrame([data.dict()])
        cols_order = ['txn_count', 'amount_sum', 'amount_mean', 'amount_std', 'recency_days']
        input_df = input_df[cols_order]
        prediction = model.predict(input_df)[0]
        
        try:
            y_proba = model._model_impl.predict_proba(input_df)
            probability = float(y_proba[0][1])
        except AttributeError:
            probability = 1.0 if prediction == 1 else 0.0
        
        return {
            "is_high_risk": int(prediction),
            "probability": round(probability, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")