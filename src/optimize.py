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

def prepare_data():
    df = pd.read_csv("data/raw/data.csv")
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], utc=True)
    
    engineer = FeatureEngineer()
    df_temp = engineer.extract_temporal_features(df)
    cust = engineer.get_aggregate_features(df_temp)
    
    labeler = RiskLabeler().fit_labels(cust[['recency_days', 'txn_count', 'amount_sum']])
    cust['is_high_risk'] = labeler.predict_risk(cust)

    X = cust.drop(columns=['AccountId', 'is_high_risk'])
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
    mlflow.set_experiment("Bati_Bank_Credit_Risk_V2")

    with mlflow.start_run(run_name="Optuna_Hyperparameter_Tuning"):
        print("Starting Hyperparameter Optimization...")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=5) 

        # Log the best parameters found by Optuna
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_roc_auc", study.best_value)

        # Train Final Model with Best Params
        X_train, X_test, y_train, y_test = prepare_data()
        best_model = RandomForestClassifier(**study.best_params, class_weight='balanced')
        best_model.fit(X_train, y_train)

        print("Generating SHAP Plot and Logging to MLflow...")
        explainer = shap.TreeExplainer(best_model)
        
    
        shap_values = explainer.shap_values(X_test)

        if isinstance(shap_values, list):
            display_shap_values = shap_values[1]
        else:
            display_shap_values = shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values

        plt.figure(figsize=(12, 8))
        
        shap.summary_plot(
            display_shap_values, 
            X_test, 
            feature_names=X_test.columns.tolist(),
            show=False
        )
        
        
        # Save locally and then log to MLflow
        plot_path = "reports/figures/shap_summary.png"
        pathlib.Path("reports/figures").mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, bbox_inches='tight')
        mlflow.log_artifact(plot_path) 
        plt.close()

        # Save the best model
        mlflow.sklearn.log_model(best_model, "best_credit_model")
        
        print(f" DONE. View results by running 'mlflow ui' and checking the Bati_Bank_Optimization experiment.")