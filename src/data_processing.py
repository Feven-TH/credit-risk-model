import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from xverse.transformer import WOE
import joblib

# FUNCTIONS (Safe to import by pytest) 

def aggregate_customer_features(df):
    agg = df.groupby('AccountId').agg(
        txn_count=('TransactionId', 'count'),
        amount_sum=('Amount', 'sum'),
        amount_mean=('Amount', 'mean'),
        amount_std=('Amount', 'std'),
        last_txn=('TransactionStartTime', 'max'),
        fraud_flag=('FraudResult', 'max')
    ).reset_index()
    return agg

def add_recency(df):
    reference_date = pd.Timestamp.now(tz='UTC')
    df['recency_days'] = (reference_date - df['last_txn']).dt.days
    return df

def finalize_dataset(df):
    # Check if column exists before dropping to avoid errors in tests
    if 'last_txn' in df.columns:
        df = df.drop(columns=['last_txn'])
    return df

class DataFrameConverter(BaseEstimator, TransformerMixin):
    """Converts NumPy array back to DataFrame for xverse compatibility."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)

#THE EXECUTION BLOCK (Only runs when you run this file directly)

if __name__ == "__main__":
    df = pd.read_csv("data/raw/data.csv") 
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], utc=True)
    
    # Feature Engineering
    cust = aggregate_customer_features(df)
    cust = add_recency(cust)
    cust = finalize_dataset(cust)

    # Dynamic Column Detection
    ignore_cols = ['AccountId', 'fraud_flag'] 
    num_cols = cust.select_dtypes(include=['float64', 'int64']).columns.tolist()
    num_cols = [c for c in num_cols if c not in ignore_cols]  

    cat_cols = cust.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ignore_cols]

    # Pipelines
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

    # Execute Transformation
    X = cust.drop(columns=['AccountId', 'fraud_flag']) 
    y_woe_target = cust['fraud_flag'] 

    X_transformed = full_pipeline.fit_transform(X, y_woe_target)

    feature_names = num_cols + [f'woe_{c}' for c in cat_cols]
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

    # Save outputs
    X_transformed_df.to_csv('data/processed/transformed_features_task3.csv', index=False)
    joblib.dump(full_pipeline, 'models/feature_pipeline_task3.pkl')

    print("--- Task 3: Feature Engineering Complete ---")
    print(f"Features transformed: {X_transformed_df.shape}")