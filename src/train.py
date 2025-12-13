from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import data_processing as dp

# --- Task 3: Data Loading and Aggregation (KEEP) ---
df = pd.read_csv("../data/raw/data.csv") 


# DATA EXECUTION (KEEP - This creates 'cust')
df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], utc=True)
cust = dp.aggregate_customer_features(df)
cust = dp.add_recency(cust)
cust = dp.finalize_dataset(cust)
target_col = 'is_high_risk' 

print("--- Task 4: Target Variable Creation (K-Means Clustering) ---")

# Calculate RFM Metrics (Already done in aggregation, just select them)

# Select the required RFM features from the customer DataFrame (cust)
rfm_features = cust[['recency_days', 'txn_count', 'amount_sum']].copy()
rfm_features.columns = ['Recency', 'Frequency', 'Monetary'] # Renamed for clarity

# -------------------------------------------
# 2. Pre-process (Scale RFM Features)
# -------------------------------------------

# Concept: Scaling is essential for K-Means so that large monetary values don't dominate the distance calculation.
scaler_rfm = StandardScaler()
rfm_scaled = scaler_rfm.fit_transform(rfm_features)

# -------------------------------------------
# 3. Cluster Customers (K-Means)
# -------------------------------------------

# Concept: K-Means segments the customers into 3 groups based on their scaled RFM profiles.
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10) 
cust['Cluster'] = kmeans.fit_predict(rfm_scaled)


# -------------------------------------------
# 4. Define and Assign the "High-Risk" Label
# -------------------------------------------

# Analyze Clusters: Calculate mean RFM values for each cluster to identify the high-risk group.
cluster_analysis = cust.groupby('Cluster')[['recency_days', 'txn_count', 'amount_sum']].mean()
print("\nCluster Analysis (Mean RFM Values):")
print(cluster_analysis)

# IDENTIFY THE HIGH-RISK CLUSTER:
# High-Risk = High Recency, Low Frequency, Low Monetary.
# We find the cluster ID (0, 1, or 2) that has the lowest mean transaction count (Frequency).
high_risk_cluster_id = cluster_analysis['txn_count'].idxmin() 

# Create the new binary target column named is_high_risk.
# Assign a value of 1 to customers in the high-risk cluster and 0 to all others.
cust[target_col] = (cust['Cluster'] == high_risk_cluster_id).astype(int)

# Integrate the Target Variable (Target is now integrated into the 'cust' DataFrame)

print(f"\nHigh-Risk Cluster ID Identified: {high_risk_cluster_id}")
print(f"Target Variable '{target_col}' Created Successfully.")
print("Target Distribution (0=Low Risk, 1=High Risk):")
print(cust[target_col].value_counts())
print("--- Task 4 Complete. Target variable is ready. ---")

# At this point, the 'cust' DataFrame has the new 'is_high_risk' column.