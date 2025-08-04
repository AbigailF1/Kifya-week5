# tests/test_data_processing.py

import pandas as pd
import pytest
from io import StringIO
from src.data_processing import load_and_prepare_transactions, create_customer_level_features

# --- Test for the first helper function ---
def test_load_and_prepare_transactions():
    """
    Tests that initial data loading and cleaning works as expected.
    - Drops rows with missing CustomerId.
    - Creates TransactionHour column.
    """
    # Create a sample CSV in memory
    csv_data = """TransactionId,CustomerId,TransactionStartTime,Value
1,C1,2025-01-15 10:00:00,100
2,,2025-01-15 11:00:00,200
3,C2,2025-01-16 14:00:00,300
"""
    # Use a dummy file path; the function will read the string
    df = load_and_prepare_transactions(StringIO(csv_data))
    
    # Assertion 1: Check that the row with the missing CustomerId was dropped
    assert df.shape[0] == 2
    assert df['CustomerId'].isnull().sum() == 0
    
    # Assertion 2: Check that the 'TransactionHour' column was created correctly
    assert 'TransactionHour' in df.columns
    assert df.loc[df['CustomerId'] == 'C2', 'TransactionHour'].iloc[0] == 14

# --- Test for the second helper function ---
def test_create_customer_level_features():
    """
    Tests that customer-level aggregation is calculated correctly.
    - Checks Recency, Frequency, and Monetary values.
    """
    # Create a sample DataFrame of transactions
    data = {
        'TransactionId': ['T1', 'T2', 'T3'],
        'CustomerId': ['C1', 'C2', 'C1'],
        'TransactionStartTime': pd.to_datetime(['2025-06-01', '2025-06-05', '2025-06-10']),
        'Value': [100, 50, 200],
        'ChannelId': ['a', 'b', 'a'] # Dummy ChannelId
    }
    df = pd.DataFrame(data)
    
    # Run the aggregation function
    customer_df = create_customer_level_features(df)
    
    # Get the data for customer 'C1'
    customer_c1 = customer_df[customer_df['CustomerId'] == 'C1'].iloc[0]
    
    # Assertions for Customer C1
    # Snapshot date is 2025-06-11. Last transaction is 2025-06-10. Recency should be 1.
    assert customer_c1['Recency'] == 1
    # Customer C1 has 2 transactions
    assert customer_c1['Frequency'] == 2
    # Customer C1 spent 100 + 200 = 300
    assert customer_c1['Monetary'] == 300