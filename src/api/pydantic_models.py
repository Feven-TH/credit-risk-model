from pydantic import BaseModel, Field
from typing import Dict, Optional

class PredictionRequest(BaseModel):
    amount_mean: float
    max_velocity: float
    avg_momentum: float
    # Default to 0.5, but allow the user to change it
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    is_high_risk: int
    probability: float
    threshold_used: float
    explanation: Optional[Dict[str, float]] = None