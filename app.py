from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load model and scaler at startup
MODEL_PATH = 'otdr_fault_detectorv1.h5'
SCALER_PATH = 'scaler.joblib'
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction='No selected file')
        if file:
            df = pd.read_csv(file)
            # Ensure correct columns
            expected_cols = ['SNR'] + [f'P{i}' for i in range(1, 31)]
            if not all(col in df.columns for col in expected_cols):
                return render_template('index.html', prediction='CSV missing required columns')
            X = df[expected_cols].values
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)
            # If classification, get class (assume argmax for multi-class)
            if pred.shape[1] > 1:
                pred_class = np.argmax(pred, axis=1)
            else:
                pred_class = (pred > 0.5).astype(int).flatten()
            prediction = ', '.join(map(str, pred_class))
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)