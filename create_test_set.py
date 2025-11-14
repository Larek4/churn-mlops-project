import sys
# FIX: Add the project root to sys.path so it can find scripts/utils.py
if '.' not in sys.path:
    sys.path.append('.')

import pandas as pd
from sklearn.model_selection import train_test_split
import scripts.utils as utils  # To get the TARGET_COLUMN name
import pandas.api.types # Required for the is_string_dtype check

# --- RE-LOAD AND RE-SPLIT DATA ---
# NOTE: This MUST use the same random_state=42 and stratify=y as before 
# to generate the exact same test set.

# Assuming your original v1 data file is still accessible.
df = pd.read_csv("churn_data_v1.csv")

# Perform the same preprocessing to clean TotalCharges before splitting
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
median_charges = df['TotalCharges'].median()
df['TotalCharges'] = df['TotalCharges'].fillna(median_charges)
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# Map the target variable 'Churn' to 0 and 1 (robust fix)
if pd.api.types.is_string_dtype(df[utils.TARGET_COLUMN]):
    df[utils.TARGET_COLUMN] = df[utils.TARGET_COLUMN].map({'Yes': 1, 'No': 0})
    
df.dropna(subset=[utils.TARGET_COLUMN], inplace=True)
df[utils.TARGET_COLUMN] = df[utils.TARGET_COLUMN].astype(int)

# --- Define X and y (Crucial for stratification) ---
X = df.drop(utils.TARGET_COLUMN, axis=1)
y = df[utils.TARGET_COLUMN]


# --- Re-split to grab the X_test and y_test sets ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Create the Golden Test Set File ---
# Combine X_test and y_test back into one DataFrame
test_set_df = pd.concat([X_test, y_test], axis=1)

# Save the combined DataFrame to a CSV file in your project root
test_set_df.to_csv("test_set.csv", index=False)

print(f"Golden Test Set saved successfully! Shape: {test_set_df.shape}")