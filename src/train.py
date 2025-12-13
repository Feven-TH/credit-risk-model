from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import data_processing as dp


df = pd.read_csv("../data/raw/data.csv") 

df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], utc=True)
cust = dp.aggregate_customer_features(df)
cust = dp.add_recency(cust)
cust = dp.finalize_dataset(cust)
target_col = 'is_high_risk' 

print("--- Task 4: Target Variable Creation (K-Means Clustering) ---")

rfm_features = cust[['recency_days', 'txn_count', 'amount_sum']].copy()
rfm_features.columns = ['Recency', 'Frequency', 'Monetary'] # Renamed for clarity

# Pre-process (Scale RFM Features)

scaler_rfm = StandardScaler()
rfm_scaled = scaler_rfm.fit_transform(rfm_features)


# Cluster Customers (K-Means)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10) 
cust['Cluster'] = kmeans.fit_predict(rfm_scaled)


#Define and Assign the "High-Risk" Label
# Analyze Clusters: Calculate mean RFM values for each cluster to identify the high-risk group

cluster_analysis = cust.groupby('Cluster')[['recency_days', 'txn_count', 'amount_sum']].mean()
print("\nCluster Analysis (Mean RFM Values):")
print(cluster_analysis)

high_risk_cluster_id = cluster_analysis['txn_count'].idxmin() 
cust[target_col] = (cust['Cluster'] == high_risk_cluster_id).astype(int)

print(f"\nHigh-Risk Cluster ID Identified: {high_risk_cluster_id}")
print(f"Target Variable '{target_col}' Created Successfully.")
print("Target Distribution (0=Low Risk, 1=High Risk):")
print(cust[target_col].value_counts())
print("--- Task 4 Complete. Target variable is ready. ---")
