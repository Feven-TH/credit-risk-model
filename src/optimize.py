import pandas as pd
import optuna
import shap
import mlflow
import matplotlib.pyplot as plt
import pathlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from core.features import FeatureEngineer
from core.labeling import RiskLabeler
from mlflow.tracking import MlflowClient # <--- Added for Aliasing
import joblib
import os

def prepare_data():
    df = pd.read_csv("data/raw/data.csv")
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], utc=True)
    
    engineer = FeatureEngineer()
    df_temp = engineer.extract_temporal_features(df)
    cust = engineer.get_aggregate_features(df_temp)
    
    labeler = RiskLabeler().fit_labels(cust[['recency_days', 'txn_count', 'amount_sum']])
    cust['is_high_risk'] = labeler.predict_risk(cust)

    leaky_cols = ['recency_days', 'txn_count', 'amount_sum']
    X = cust.drop(columns=['AccountId', 'is_high_risk'] + leaky_cols)
    print(f"DEBUG: Training features are: {X.columns.tolist()}")
    y = cust['is_high_risk']
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def objective(trial):
    X_train, X_test, y_train, y_test = prepare_data()
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'class_weight': 'balanced',
        'random_state': 42
    }
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, preds)

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(root_dir, "mlflow.db")
    
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("Bati_Bank_Credit_Risk_V2")
    with mlflow.start_run(run_name="Optuna_Hyperparameter_Tuning"):
        print("Starting Hyperparameter Optimization...")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=5) 

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_roc_auc", study.best_value)

        X_train, X_test, y_train, y_test = prepare_data()
        best_model = RandomForestClassifier(**study.best_params, class_weight='balanced')
        best_model.fit(X_train, y_train)

        # Explainability
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test)
        display_shap_values = shap_values[1] if isinstance(shap_values, list) else shap_values
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(display_shap_values, X_test, show=False)
        plot_path = "reports/figures/shap_summary.png"
        pathlib.Path("reports/figures").mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, bbox_inches='tight')
        mlflow.log_artifact(plot_path) 
        plt.close()

        # Registration logic
        model_name = "Bati_Bank_Credit_Model"
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name=model_name
        )

        # 👑 ALIASING: Set this version as the 'champion'
        client = MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version
        client.set_registered_model_alias(model_name, "champion", latest_version)

        joblib.dump(best_model, "models/final_credit_model.pkl")
        print(f"👑 Version {latest_version} promoted to '@champion'")
        print(f"DONE. Model saved and registered.")