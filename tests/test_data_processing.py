import pandas as pd
import numpy as np
import pytest


from research.data_processing import aggregate_customer_features, add_recency

@pytest.fixture
def sample_raw_data():
    return pd.DataFrame({
        'AccountId': ['A001', 'A001', 'A002', 'A003', 'A003'],
        'TransactionId': [1, 2, 3, 4, 5],
        'Amount': [10.0, 5.0, 100.0, 20.0, -5.0],
        'TransactionStartTime': pd.to_datetime([
            '2025-12-01 10:00:00',
            '2025-12-05 15:00:00',
            '2025-11-20 08:00:00',
            '2025-12-10 12:00:00',
            '2025-12-10 12:00:00'
        ], utc=True),
        'FraudResult': [0, 0, 1, 0, 0]
    })

# Test 1: Check if the aggregated features have the correct values (Count/Sum)
def test_aggregated_values(sample_raw_data):
    # Act
    agg_df = aggregate_customer_features(sample_raw_data)
    agg_df = agg_df.set_index('AccountId')
    
    # Assert
    # Check A001: 2 transactions, sum = 15.0
    assert agg_df.loc['A001', 'txn_count'] == 2
    assert agg_df.loc['A001', 'amount_sum'] == 15.0
    
    # Check A003: 2 transactions, sum = 15.0 (20.0 + (-5.0))
    assert agg_df.loc['A003', 'txn_count'] == 2
    assert agg_df.loc['A003', 'amount_sum'] == 15.0
    
# Test 2: Check if the required Recency column is created and is positive
def test_recency_calculation(sample_raw_data):
    agg_df = aggregate_customer_features(sample_raw_data)
    rec_df = add_recency(agg_df)
    assert 'recency_days' in rec_df.columns
    assert (rec_df['recency_days'] >= 0).all()
