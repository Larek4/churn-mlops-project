# scripts/serve.py (FINAL, CORRECTED VERSION)

import mlflow
import uvicorn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import os # <-- Import os

# --- Configuration ---
# This is the address of your MLflow server
MLFLOW_TRACKING_URI = "http://16.171.65.31:5000"
MODEL_NAME = "CustomerChurnModel_V2"
MODEL_STAGE = "Production"

# --- Pydantic Model for Input Validation ---
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# --- FastAPI App ---
app = FastAPI()

# --- *** THE CRITICAL FIX *** ---
# Set the MLflow Tracking URI so the client knows where the server is
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# ---------------------------------

# --- Load the Production Model ---
print(f"Loading model '{MODEL_NAME}' from stage '{MODEL_STAGE}'...")
model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}"
)
print("Model loaded successfully!")

# --- API Endpoints ---
@app.get("/")
def read_root():
    """Root endpoint to check if the API is alive."""
    return {"status": "ok", "model_name": MODEL_NAME, "model_stage": MODEL_STAGE}

@app.post("/predict")
def predict_churn(data: CustomerData):
    """
    Prediction endpoint.
    Takes a single customer's data and returns a churn prediction.
    """
    try:
        # Convert the Pydantic model to a DataFrame, which the model expects
        df = pd.DataFrame([data.model_dump()])
        
        # Get the prediction (will be 0 or 1)
        prediction = model.predict(df)[0]
        
        # Convert prediction to a human-readable label
        label = "No Churn" if prediction == 0 else "Churn"
        
        return {
            "prediction": int(prediction),
            "label": label,
            "input_data": data.model_dump()
        }
    except Exception as e:
        return {"error": str(e)}

# --- Run the API ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)