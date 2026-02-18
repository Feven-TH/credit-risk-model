from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

class RiskLabeler:
    def __init__(self, n_clusters: int = 3):
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.high_risk_cluster = None
        self.feature_names = ['recency_days', 'txn_count', 'amount_sum']

    def fit_labels(self, rfm_df: pd.DataFrame):
        """Identifies the 'High Risk' cluster based on low transaction frequency."""
        # Ensure we only use the 3 RFM columns for scaling/fitting
        data_to_fit = rfm_df[self.feature_names]
        
        scaled_data = self.scaler.fit_transform(data_to_fit)
        clusters = self.kmeans.fit_predict(scaled_data)
        
        # Temporary column for analysis only
        temp_df = data_to_fit.copy()
        temp_df['cluster'] = clusters
        
        # Identify high risk: usually the group with the lowest transaction count
        analysis = temp_df.groupby('cluster')['txn_count'].mean()
        self.high_risk_cluster = analysis.idxmin()
        return self

    def predict_risk(self, rfm_df: pd.DataFrame) -> pd.Series:
        """Assigns risk labels to customers using only the RFM features."""
        # Filter to only use the features the scaler expects
        data_to_predict = rfm_df[self.feature_names]
        
        scaled_data = self.scaler.transform(data_to_predict)
        clusters = self.kmeans.predict(scaled_data)
        return (clusters == self.high_risk_cluster).astype(int)