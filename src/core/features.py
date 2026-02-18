import pandas as pd
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class FeatureConfig:
    REFERENCE_DATE: pd.Timestamp = pd.Timestamp.now(tz='UTC')
    VELOCITY_WINDOW: str = '30D' 

class FeatureEngineer:
    def __init__(self, config: FeatureConfig = FeatureConfig()):
        self.config = config

    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates spending velocity and momentum using time-based rolling windows.
        Fixed for Linux/Pandas compatibility using index-based rolling.
        """
        # Ensure data is sorted by time
        df = df.sort_values('TransactionStartTime')
        
        # Set TransactionStartTime as index temporarily for the rolling window logic
        df = df.set_index('TransactionStartTime')
        
        # 1. Spending Momentum (30-day sum)
        # We group by AccountId and then roll over the time-index
        df['spending_momentum_30d'] = df.groupby('AccountId')['Amount'].transform(
            lambda x: x.rolling(window=self.config.VELOCITY_WINDOW).sum()
        )
        
        # 2. Transaction Velocity (30-day count)
        df['txn_velocity_30d'] = df.groupby('AccountId')['Amount'].transform(
            lambda x: x.rolling(window=self.config.VELOCITY_WINDOW).count()
        )
        
        # Reset index to bring TransactionStartTime back as a column
        return df.reset_index()

    def get_aggregate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Groups data into a per-customer summary."""
        agg = df.groupby('AccountId').agg(
            txn_count=('TransactionId', 'count'),
            amount_sum=('Amount', 'sum'),
            amount_mean=('Amount', 'mean'),
            last_txn=('TransactionStartTime', 'max'),
            avg_momentum=('spending_momentum_30d', 'mean'),
            max_velocity=('txn_velocity_30d', 'max')
        ).reset_index()
        
        agg['recency_days'] = (self.config.REFERENCE_DATE - agg['last_txn']).dt.days
        return agg.drop(columns=['last_txn'])