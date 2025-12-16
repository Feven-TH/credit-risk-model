from pydantic import BaseModel

class PredictionRequest(BaseModel):
    txn_count: int
    amount_sum: float
    amount_mean: float
    amount_std: float
    recency_days: float

class PredictionResponse(BaseModel):
    is_high_risk: int
    probability: float