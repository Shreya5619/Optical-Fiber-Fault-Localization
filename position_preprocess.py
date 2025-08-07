import numpy as np
import pandas as pd

# Define the expected feature columns
EXPECTED_COLUMNS = ['SNR'] + [f'P{i}' for i in range(1, 31)]

def preprocess_input(raw_input):
    """
    Takes a dictionary, list, or pandas Series of raw input data 
    and returns a NumPy array formatted for the XGBoost model.
    
    Args:
        raw_input (dict | list | pd.Series): Raw input row of 31 features
    
    Returns:
        np.ndarray: Preprocessed array with shape (1, 31)
    """
    if isinstance(raw_input, dict):
        # Ensure correct order
        row = [raw_input.get(col, 0.0) for col in EXPECTED_COLUMNS]
    elif isinstance(raw_input, pd.Series):
        row = raw_input[EXPECTED_COLUMNS].values
    elif isinstance(raw_input, list) or isinstance(raw_input, np.ndarray):
        if len(raw_input) != 31:
            raise ValueError("Input list must have 31 values: [SNR, P1, ..., P30]")
        row = raw_input
    else:
        raise TypeError("Input must be a dict, list, Series, or ndarray")

    # Convert to 2D numpy array
    return np.array(row, dtype=np.float32).reshape(1, -1)
