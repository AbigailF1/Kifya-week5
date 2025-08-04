# src/create_target_variable.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
import joblib
import logging
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# --- Basic Setup ---
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_target_variable(base_features_path, final_data_path):
    """
    Creates the 'is_high_risk' proxy target variable using RFM and K-Means clustering.
    """
    logging.info(f"Loading base customer features from {base_features_path}...")
    try:
        df = pd.read_csv(base_features_path)
    except FileNotFoundError:
        logging.error(f"Base features file not found at {base_features_path}. Exiting.")
        sys.exit(1)

    # --- Step 1: Prepare RFM Metrics for Clustering ---
    rfm_features = df[['Recency', 'Frequency', 'Monetary']]

    # Log transform for skewed Frequency and Monetary data, then scale all features.
    # This pipeline ensures the data is properly pre-processed for K-Means.
    preprocessing_pipeline = Pipeline([
        ('log_transform', FunctionTransformer(np.log1p)),
        ('scaler', StandardScaler())
    ])

    logging.info("Applying scaling and transformation to RFM features...")
    rfm_scaled = preprocessing_pipeline.fit_transform(rfm_features)

    # --- Step 2: Cluster Customers using K-Means ---
    logging.info("Performing K-Means clustering...")
    kmeans = KMeans(
        n_clusters=3,
        init='k-means++',
        random_state=42 # Set for reproducibility
    )
    kmeans.fit(rfm_scaled)
    
    # Assign cluster labels back to the main dataframe
    df['RFM_Cluster'] = kmeans.labels_

    # --- Step 3: Define and Assign the "High-Risk" Label ---
    logging.info("Analyzing RFM clusters to identify the high-risk group...")
    
    # Calculate the mean RFM values for each cluster to understand their characteristics
    cluster_analysis = df.groupby('RFM_Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    logging.info("Cluster Centers (Mean RFM Values):\n" + str(cluster_analysis))
    
    # Identify the high-risk cluster:
    # High Recency (long since last purchase)
    # Low Frequency (few purchases)
    # Low Monetary (low total spend)
    high_risk_cluster = cluster_analysis.idxmax()['Recency']
    
    logging.info(f"Identified Cluster {high_risk_cluster} as the high-risk segment.")

    # Create the binary target column
    df['is_high_risk'] = np.where(df['RFM_Cluster'] == high_risk_cluster, 1, 0)

    # --- Step 4: Integrate and Save the Final Dataset ---
    logging.info("Dropping intermediate columns and saving the final dataset.")
    final_df = df.drop(columns=['RFM_Cluster'])
    
    try:
        final_df.to_csv(final_data_path, index=False)
        logging.info(f"Final training dataset with target variable saved to {final_data_path}")
    except Exception as e:
        logging.error(f"Failed to save final dataset: {e}")

    # Save a plot for visual analysis of the clusters
    try:
        sns.pairplot(df, hue='RFM_Cluster', vars=['Recency', 'Frequency', 'Monetary'], palette='viridis')
        plt.suptitle("RFM Cluster Profiles", y=1.02)
        plt.savefig('../reports/figures/rfm_cluster_profiles.png')
        logging.info("Saved cluster profile plot to reports/figures.")
    except Exception as e:
        logging.warning(f"Could not generate cluster plot: {e}")

    return final_df

# --- Main execution block ---
if __name__ == '__main__':
    import os
    BASE_FEATURES_PATH = os.getcwd()+'/../data/processed/customer_features_base.csv'
    FINAL_DATA_PATH = os.getcwd()+'/../data/processed/final_training_data.csv'
    
    create_target_variable(BASE_FEATURES_PATH, FINAL_DATA_PATH)