# scripts/train.py (FINAL VERSION)

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import sys, os, pandas.api.types
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import utils 

# This will now use the IP from the GitHub Action's environment variable
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://13.51.241.216:5000") # <-- NEW IP

def get_data(data_path):
    """Loads and performs initial cleaning on the raw data."""
    df = pd.read_csv(data_path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    if 'customerID' in df.columns: df.drop('customerID', axis=1, inplace=True)
    if pd.api.types.is_string_dtype(df[utils.TARGET_COLUMN]):
        df[utils.TARGET_COLUMN] = df[utils.TARGET_COLUMN].map({'Yes': 1, 'No': 0})
    df.dropna(subset=[utils.TARGET_COLUMN], inplace=True)
    df[utils.TARGET_COLUMN] = df[utils.TARGET_COLUMN].astype(int)
    return df, df['TotalCharges'].median()

def create_pipeline(classifier_params):
    """Creates the full preprocessor + SMOTE + Classifier pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), utils.NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), utils.CATEGORICAL_FEATURES)
        ], remainder='passthrough')
    return ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42, **classifier_params))
    ])

def run_training(data_path):
    """Orchestrates the entire training, logging, and model saving process."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(utils.MODEL_NAME)
    with mlflow.start_run() as run:
        print("Running data preparation...")
        df, median_charges = get_data(data_path)
        X = df.drop(utils.TARGET_COLUMN, axis=1)
        y = df[utils.TARGET_COLUMN]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print("Training model...")
        params = {'n_estimators': 100, 'max_depth': 10}
        mlflow.log_params(params)
        model_pipeline = create_pipeline(params)
        model_pipeline.fit(X_train, y_train)
        
        local_model_path = "model_new.pkl"
        joblib.dump(model_pipeline, local_model_path)
        
        # Log model properly for Registry registration
        mlflow.sklearn.log_model(
            sk_model=model_pipeline,
            artifact_path="model", # Standard MLflow folder name
        )

        # Write the Run ID to a file for the evaluator
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)
        print(f"Model trained and saved.")
        return X_test, y_test, local_model_path

if __name__ == '__main__':
    try:
        # This script is now run by GitHub Actions, which syncs S3 data
        # It will find 'churn_data_v1.csv' in the root
        X_test, y_test, local_model_path = run_training("churn_data_v1.csv")
        print(f"\n✅ SUCCESS: Model trained and saved locally to {local_model_path}.")
    except Exception as e:
        print(f"FATAL ERROR during training run: {e}")
        exit(1)