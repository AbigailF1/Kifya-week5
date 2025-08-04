# src/data_processing.py

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
import logging
import sys

# --- Basic Setup ---
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_prepare_transactions(raw_data_path):
    """
    Loads raw data and performs transaction-level feature extraction.
    """
    logging.info(f"Loading raw data from {raw_data_path}...")
    df = pd.read_csv(raw_data_path)

    # Removal of rows with missing CustomerId
    df.dropna(subset=['CustomerId'], inplace=True)
    
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

    # Extract Time-Based Features
    df['TransactionHour'] = df['TransactionStartTime'].dt.hour
    df['TransactionDay'] = df['TransactionStartTime'].dt.day
    df['TransactionMonth'] = df['TransactionStartTime'].dt.month
    df['TransactionYear'] = df['TransactionStartTime'].dt.year
    
    logging.info("Transaction-level feature extraction complete.")
    return df


def create_customer_level_features(df):
    """
    Aggregates transaction data to create customer-level features, including RFM.
    """
    logging.info("Aggregating data to create customer-level features...")
    
    df['ChannelName'] = df['ChannelId'].str.split('_').str[0] # Simplified mapping

    # --- THIS IS THE KEY EDIT ---
    # Define a snapshot date for Recency calculation before aggregation.
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    
    customer_df = df.groupby('CustomerId').agg(
        # --- RFM Metrics (Now Included) ---
        Recency=('TransactionStartTime', lambda date: (snapshot_date - date.max()).days),
        Frequency=('TransactionId', 'count'),
        Monetary=('Value', 'sum'),
        
        # --- Other Aggregate Features ---
        AverageTransactionAmount=('Value', 'mean'),
        StdDevTransactionAmount=('Value', 'std'),
        AvgTransactionHour=('TransactionHour', 'mean'),
        MostFrequentMonth=('TransactionMonth', lambda x: x.mode()[0]),
        MostFrequentChannel=('ChannelName', lambda x: x.mode()[0])
    ).reset_index()

    # Handle missing values that can arise from aggregation
    customer_df['StdDevTransactionAmount'].fillna(0, inplace=True)

    logging.info("Customer-level feature aggregation complete with RFM metrics.")
    return customer_df


def build_preprocessing_pipeline():
    """
    Builds a scikit-learn pipeline for final feature preprocessing.
    This function remains the same as it adapts to the columns it's given.
    """
    logging.info("Building the feature preprocessing pipeline...")

    numerical_features = [
        'Recency', 'Frequency', 'Monetary', 'AverageTransactionAmount', 
        'StdDevTransactionAmount', 'AvgTransactionHour', 'MostFrequentMonth'
    ]
    categorical_features = ['MostFrequentChannel']

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor


# --- Main execution block ---
if __name__ == '__main__':
    import os
    RAW_DATA_PATH = os.getcwd()+'/../data/data.csv'
    PROCESSED_DATA_PATH = os.getcwd()+'/../data/processed/customer_features_base.csv'
    PIPELINE_PATH = os.getcwd()+'/../models/base_preprocessing_pipeline.joblib'

    transactions_df = load_and_prepare_transactions(RAW_DATA_PATH)
    customer_features_df = create_customer_level_features(transactions_df)
    
    # Save the customer-level data, which now includes RFM columns
    customer_features_df.to_csv(PROCESSED_DATA_PATH, index=False)
    
    pipeline = build_preprocessing_pipeline()
    
    X = customer_features_df.drop('CustomerId', axis=1)
    pipeline.fit(X)

    joblib.dump(pipeline, PIPELINE_PATH)
    logging.info(f"Base preprocessing pipeline saved to {PIPELINE_PATH}")
    logging.info("Updated data processing script finished successfully.")