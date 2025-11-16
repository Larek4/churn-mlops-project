# scripts/utils.py

# --- Feature Definitions ---
CATEGORICAL_FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]

NUMERICAL_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']

TARGET_COLUMN = 'Churn'

# --- Model Constants ---
# NOTE: This name must be unique. We are using V2 due to previous failures.
MODEL_NAME = "CustomerChurnModel_V3"