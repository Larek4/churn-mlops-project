# scripts/train.py - FINAL VERSION

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn # NEW IMPORT
import sys
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import pandas.api.types

import utils 

# --- Configuration ---
MLFLOW_TRACKING_URI = "http://16.171.65.31:5000"


def get_data(data_path):
    """Loads and performs initial cleaning on the raw data."""
    df = pd.read_csv(data_path)
    
    # 1. Fix TotalCharges: convert to numeric and fill missing (robust fix)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    median_charges = df['TotalCharges'].median()
    df['TotalCharges'] = df['TotalCharges'].fillna(median_charges)
    
    # 2. Drop the customerID column
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    
    # 3. Map the target variable 'Churn' to 0 and 1 (robust fix)
    if pd.api.types.is_string_dtype(df[utils.TARGET_COLUMN]):
        df[utils.TARGET_COLUMN] = df[utils.TARGET_COLUMN].map({'Yes': 1, 'No': 0})
    
    # 4. Handle missing targets (the defensive programming fix)
    df.dropna(subset=[utils.TARGET_COLUMN], inplace=True)
    df[utils.TARGET_COLUMN] = df[utils.TARGET_COLUMN].astype(int)
    
    return df, median_charges


def create_pipeline(classifier_params):
    """Creates the full preprocessor + SMOTE + Classifier pipeline."""
    
    # Define the preprocessor using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), utils.NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), utils.CATEGORICAL_FEATURES)
        ],
        remainder='passthrough'
    )
    
    # Create the full imblearn pipeline (preprocessor, SMOTE, and model)
    model_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42, **classifier_params))
    ])
    
    return model_pipeline


def run_training(data_path):
    """Orchestrates the entire training, logging, and model saving process."""
    
    # --- MLFLOW SETUP ---
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(utils.MODEL_NAME)
    
    # Use a context manager for a clean run
    with mlflow.start_run() as run:
        
        # --- 1. DATA PREPARATION ---
        print("Running data preparation...")
        df, median_charges = get_data(data_path)
        X = df.drop(utils.TARGET_COLUMN, axis=1)
        y = df[utils.TARGET_COLUMN]
        
        # Split data (using the same stratify as the notebook)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # --- 2. MODEL TRAINING ---
        print("Training model...")
        
        # Define simple parameters to log
        params = {'n_estimators': 100, 'max_depth': 10}
        mlflow.log_params(params)
        
        model_pipeline = create_pipeline(params)
        model_pipeline.fit(X_train, y_train)
        
        # --- 3. MODEL SAVING (Local Artifact) ---
        # 1. Save locally for the local evaluation script to use
        local_model_path = "model_new.pkl"
        joblib.dump(model_pipeline, local_model_path)
        
        # 2. Log model properly for Registry registration (THIS IS THE CRITICAL FIX)
        mlflow.sklearn.log_model(
            sk_model=model_pipeline,
            artifact_path="model", # Standard MLflow folder name
        )

        # --- FIX: Write the Run ID to a file for the evaluator ---
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)
        
        print(f"Model trained and saved to local disk and MLflow artifact store.")
        
        # Return test data for the evaluation script
        return X_test, y_test, local_model_path


if __name__ == '__main__':
    try:
        X_test, y_test, local_model_path = run_training("churn_data_v1.csv")
        print(f"\n✅ SUCCESS: Model trained and saved locally to {local_model_path}.")
        print("Ready for evaluation.")
    except Exception as e:
        print(f"FATAL ERROR during training run: {e}")
        exit(1)