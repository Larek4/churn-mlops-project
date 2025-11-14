# scripts/evaluate.py - FINAL VERSION

import pandas as pd
import joblib
import mlflow
import utils # Our local utility file
import sys
import os

from sklearn.metrics import f1_score, classification_report
from mlflow.exceptions import RestException 

# --- Configuration ---
MLFLOW_TRACKING_URI = "http://51.21.150.132:5000"

# FIX: Add the project root to sys.path so it can find test_set.csv 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def log_and_register_model(X_test, y_test, model_path, run_id):
    """
    Runs the model quality checks (Imbalance/Bias) and promotes the model
    to the MLflow Model Registry if performance is better than the current Production model.
    """
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(utils.MODEL_NAME)
    
    # Load the new model
    model_pipeline = joblib.load(model_path)
    y_pred = model_pipeline.predict(X_test)
    
    # --- 1. HANDLE IMBALANCE: Calculate Primary Metric ---
    new_f1 = f1_score(y_test, y_pred, pos_label=1)
    mlflow.log_metric("validation_f1_score", new_f1)
    print(f"New Model F1-Score: {new_f1:.4f}")
    
    # --- 2. HANDLE BIAS: Log Subgroup Metrics (as requested) ---
    X_test_seniors = X_test[X_test['SeniorCitizen'] == 1]
    y_test_seniors = y_test.loc[X_test_seniors.index]
    y_pred_seniors = model_pipeline.predict(X_test_seniors)
    
    X_test_non_seniors = X_test[X_test['SeniorCitizen'] == 0]
    y_test_non_seniors = y_test.loc[X_test_non_seniors.index]
    y_pred_non_seniors = model_pipeline.predict(X_test_non_seniors)
    
    f1_seniors = f1_score(y_test_seniors, y_pred_seniors, pos_label=1)
    f1_non_seniors = f1_score(y_test_non_seniors, y_pred_non_seniors, pos_label=1)
    
    mlflow.log_metric("f1_seniors", f1_seniors)
    mlflow.log_metric("f1_non_seniors", f1_non_seniors)
    print(f"F1-Score for Seniors: {f1_seniors:.4f}")
    print(f"F1-Score for Non-Seniors: {f1_non_seniors:.4f}")

    # --- 3. THE MLOPS QUALITY GATE (Continuous Training Logic) ---
    
    current_prod_f1 = 0.0 
    client = mlflow.tracking.MlflowClient(MLFLOW_TRACKING_URI)
    
    try:
        latest_prod_version = client.get_latest_versions(utils.MODEL_NAME, stages=['Production'])
        
        if latest_prod_version:
            prod_version = latest_prod_version[0]
            prod_run = client.get_run(prod_version.run_id)
            current_prod_f1 = prod_run.data.metrics.get("validation_f1_score", 0.0) 
    
    except RestException as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e) or "RestModelRegistryService" in str(e):
            print("INFO: No existing Production model found. Proceeding with registration.")
            current_prod_f1 = 0.0 
        else:
            print(f"ERROR during registry check: {e}")
            print("Promotion skipped.")
            return

    print(f"\n--- QUALITY CHECK ---")
    print(f"Current Production F1: {current_prod_f1:.4f}")
    print(f"New Model F1: {new_f1:.4f}")
    
    # --- PROMOTION LOGIC ---
    if new_f1 > current_prod_f1:
        
        # Register the model using the standard artifact path
        model_uri = f"runs:/{run_id}/model" # THIS IS THE NEW, CORRECT ARTIFACT PATH
        mv = mlflow.register_model(model_uri, utils.MODEL_NAME)
        
        # Transition the model to Production stage
        client.transition_model_version_stage(
            name=utils.MODEL_NAME,
            version=mv.version,
            stage="Production"
        )
        print(f"✅ PROMOTION SUCCESS: New model (v{mv.version}) is better and deployed to Production!")
        
    else:
        print(f"❌ REGRESSION DETECTED: New model F1 is not better than Production. Model WILL NOT be deployed.")


if __name__ == '__main__':
    # --- Local Testing Setup ---
    
    # Load the golden test set
    test_df = pd.read_csv("test_set.csv")
    X_test = test_df.drop(utils.TARGET_COLUMN, axis=1)
    y_test = test_df[utils.TARGET_COLUMN]

    # Load the Run ID from the file created by train.py
    try:
        with open("run_id.txt", "r") as f:
            actual_run_id = f.read().strip()
    except FileNotFoundError:
        print("\nFATAL ERROR: 'run_id.txt' not found.")
        print("Please run 'python scripts/train.py' first to generate the Run ID.")
        exit(1)

    print("Assuming 'model_new.pkl' exists from previous training run...")
    log_and_register_model(X_test, y_test, "model_new.pkl", run_id=actual_run_id)