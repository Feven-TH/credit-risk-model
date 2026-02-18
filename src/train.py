import pandas as pd
import pathlib
import joblib
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from core.features import FeatureEngineer
from core.labeling import RiskLabeler

def run_production_training():
    mlflow.set_experiment("Bati_Bank_Credit_Risk_V2")
    
    # Load Data
    df = pd.read_csv("data/raw/data.csv")
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], utc=True)

    # 1. Feature Engineering (Includes Temporal Velocity)
    engineer = FeatureEngineer()
    df_temporal = engineer.extract_temporal_features(df)
    cust = engineer.get_aggregate_features(df_temporal)

    # 2. Stable Risk Labeling
    labeler = RiskLabeler()
    rfm_data = cust[['recency_days', 'txn_count', 'amount_sum']].copy()
    labeler.fit_labels(rfm_data)
    cust['is_high_risk'] = labeler.predict_risk(rfm_data)

    # 3. Model Training
    leaky_features = ['recency_days', 'txn_count', 'amount_sum']
    X = cust.drop(columns=['AccountId', 'is_high_risk'] + leaky_features)
    y = cust['is_high_risk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    with mlflow.start_run(run_name="Random_Forest_Production"):
        model = RandomForestClassifier(n_estimators=100, class_weight='balanced')
        model.fit(X_train, y_train)
        
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        mlflow.log_metric("roc_auc", auc)
        
        # Save Artifacts
        pathlib.Path("models").mkdir(exist_ok=True)
        joblib.dump(model, "models/credit_risk_model_v2.pkl")
        joblib.dump(labeler, "models/risk_labeler.pkl")
        mlflow.sklearn.log_model(model, "model")
        
        print(f"✅ Training Complete! AUC: {auc:.4f}")

if __name__ == "__main__":
    run_production_training()