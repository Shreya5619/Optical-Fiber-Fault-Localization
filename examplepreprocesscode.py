import pandas as pd
import numpy as np
import joblib

# Load the trained scaler
scaler = joblib.load('scaler.joblib')

# Define preprocessing function
def preprocess_input(sample_df):
    # Must have 'SNR', 'P1' to 'P30' columns
    X = sample_df[['SNR'] + [f'P{i}' for i in range(1, 31)]].values
    X_scaled = scaler.transform(X)
    return {
        'OTDR_trace': X_scaled[:, 1:31],
        'SNR': X_scaled[:, 0].reshape(-1, 1).astype(np.float32)
    }
