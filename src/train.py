import joblib
import mlflow
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import pandas as pd
import data_processing as dp
from xverse.transformer import WOE
import pathlib


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


# Task 5: Final Feature Pipeline Setup ---

target_col = 'is_high_risk'

# 1. DYNAMIC COLUMN DETECTION (Now including the target variable)
# We drop 'fraud_flag' as we are using the new 'is_high_risk' proxy target.
ignore_cols = [target_col, 'AccountId', 'Cluster', 'fraud_flag'] 

num_cols = cust.select_dtypes(include=['float64', 'int64']).columns.tolist()
num_cols = [c for c in num_cols if c not in ignore_cols]  

cat_cols = cust.select_dtypes(include=['object', 'category']).columns.tolist()
cat_cols = [c for c in cat_cols if c not in ignore_cols]

# 2. DATAFRAME CONVERTER (For xverse compatibility - kept as a utility class)
class DataFrameConverter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)

# 3. PIPELINES AND COLUMN TRANSFORMER (Same as Task 3, but now using the final column lists)
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('df_converter', DataFrameConverter()),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ('woe', WOE()) 
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])
full_pipeline = Pipeline([
    ('preprocessor', preprocessor)
])


#  Task 5: Data Preparation and Splitting
X = cust.drop(columns=ignore_cols)
y = cust[target_col]          # Target: the new 'is_high_risk' column

# SPLIT DATA
# stratify=y ensures the training and testing sets have the same proportion of high-risk customers.
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nData Split Complete. Training set size: {X_train_raw.shape[0]} samples.")

# TRANSFORM FEATURES
# Fit the pipeline ONLY on the training data (to prevent data leakage)

X_train = full_pipeline.fit_transform(X_train_raw, y_train)
X_test = full_pipeline.transform(X_test_raw)

print("Features successfully transformed (Scaled and WoE-encoded).")

# Task 5: MLflow Experiment Tracking and Training

# 1. Set MLflow Experiment
mlflow.set_experiment("Credit Risk Proxy Model")

def train_and_log_model(model, X_train, y_train, X_test, y_test, model_name, params):
    """Trains a model and logs metrics and artifacts to MLflow."""

    with mlflow.start_run(run_name=model_name) as run:
        
        # Log parameters
        mlflow.log_params(params)
        
        # Train Model
        model.fit(X_train, y_train)
        
        # Predict and Evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate Metrics
        roc_auc = roc_auc_score(y_test, y_proba)
        
        # Detailed metrics from classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Log Metrics
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("accuracy", report['accuracy'])
        mlflow.log_metric("precision_1", report['1']['precision'])
        mlflow.log_metric("recall_1", report['1']['recall'])
        mlflow.log_metric("f1_score_1", report['1']['f1-score'])
        
        # Log the model artifact
        mlflow.sklearn.log_model(model, "model")
        
        artifact_dir = pathlib.Path("artifacts")
        artifact_dir.mkdir(exist_ok=True) 

        artifact_path = artifact_dir / f"full_pipeline_{model_name}.pkl"

        joblib.dump(full_pipeline, artifact_path)
        mlflow.log_artifact(artifact_path, "feature_pipeline")

        # Log the full feature pipeline as an artifact
        joblib.dump(full_pipeline, f"artifacts/full_pipeline_{model_name}.pkl")
        mlflow.log_artifact(f"artifacts/full_pipeline_{model_name}.pkl", "feature_pipeline")
        
        print(f"\n--- {model_name} Results ---")
        print(f"ROC AUC: {roc_auc:.4f}")
        print("Classification Report:\n", classification_report(y_test, y_pred))
        
        return model, roc_auc

# Train Logistic Regression (Model 1)
logreg_params = {
    'solver': 'liblinear', 
    'class_weight': 'balanced', 
    'random_state': 42
}
logreg_model = LogisticRegression(**logreg_params)

train_and_log_model(
    logreg_model, 
    X_train, 
    y_train, 
    X_test, 
    y_test, 
    "Logistic_Regression_V1", 
    logreg_params
)

# Train Random Forest (Model 2)
rf_params = {
    'n_estimators': 100, 
    'max_depth': 10, 
    'random_state': 42, 
    'class_weight': 'balanced'
}
rf_model = RandomForestClassifier(**rf_params)

train_and_log_model(
    rf_model, 
    X_train, 
    y_train, 
    X_test, 
    y_test, 
    "Random_Forest_V1", 
    rf_params
)

print("\n--- Task 5 Model Training Complete. Check MLflow UI for comparison. ---")