from dataclasses import dataclass, field
import os
from pathlib import Path

@dataclass(frozen=True)
class APIConfig:
    # Model Metadata
    MODEL_NAME: str = "Bati_Bank_Credit_Model"
    MODEL_ALIAS: str = "champion"
    
    # Feature Engineering (The exact order your model expects)
    FEATURE_ORDER: list[str] = field(default_factory=lambda: [
        "amount_mean", 
        "avg_momentum", 
        "max_velocity"
    ])
    
    # Infrastructure Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DB_NAME: str = "mlflow.db"
    LOG_NAME: str = "predictions_log.csv"
    
    @property
    def db_uri(self) -> str:
        # UPDATE: Check if an environment variable exists first (useful for Docker/CI)
        # Otherwise, fall back to the local absolute path.
        env_uri = os.getenv("MLFLOW_TRACKING_URI")
        if env_uri:
            return env_uri
        return f"sqlite:///{self.BASE_DIR / self.DB_NAME}"
    
    @property
    def log_path(self) -> Path:
        return self.BASE_DIR / self.LOG_NAME

# Create a single instance to be imported
config = APIConfig()