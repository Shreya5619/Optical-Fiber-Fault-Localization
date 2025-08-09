from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)

# Fault class mapping
FAULT_CLASSES = {
    0: "Normal",
    1: "Fiber Tapping",
    2: "Bad Splice",
    3: "Bending Event",
    4: "Dirty Connector",
    5: "Fiber Cut",
    6: "PC Connector",
    7: "Reflector"
}

# Directory where models are stored
MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Helper functions for model loading and prediction
def load_model_safe(model_path):
    try:
        if model_path.endswith('.h5'):
            return tf.keras.models.load_model(model_path)
        else:
            try:
                return joblib.load(model_path)
            except:
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

def preprocess_for_binary(data_series):
    """Preprocess data for binary classification"""
    features = [data_series['SNR']]
    for i in range(1, 31):
        features.append(data_series[f'P{i}'])
    return np.array(features).reshape(1, -1)

def simulate_model_prediction(model_path, input_data, prediction_type):
    """Simulate or use actual model prediction"""
    if model_path:
        model = load_model_safe(model_path)
        if model:
            try:
                if 'keras' in str(type(model)).lower():
                    # Handle Keras multi-input if necessary
                    if prediction_type == 'class':
                        snr_input = input_data[:, 0].reshape(-1, 1)
                        otdr_trace_input = input_data[:, 1:]
                        pred = model.predict([otdr_trace_input, snr_input])
                    else:
                        pred = model.predict(input_data)

                    if prediction_type == 'binary':
                        if pred.shape[-1] == 1:
                            pred_label = int(pred[0][0] > 0.5)
                            confidence = float(pred[0][0]) if pred_label == 1 else 1 - float(pred[0][0])
                            return {'prediction': bool(pred_label), 'confidence': confidence}
                        else:
                            pred_label = int(np.argmax(pred[0]))
                            confidence = float(np.max(pred[0]))
                            return {'prediction': bool(pred_label), 'confidence': confidence}

                    elif prediction_type == 'class':
                        pred_label = int(np.argmax(pred[0]))
                        return {'prediction': pred_label}

                    elif prediction_type in ['position', 'reflectance', 'loss']:
                        prediction_value = float(pred[0][0])
                        if prediction_type == 'position':
                            position = np.clip(np.round(prediction_value / 0.01) * 0.01, 0, 0.30)
                            return {'prediction': position}
                        return {'prediction': prediction_value}
                else:
                    # scikit-learn models
                    if prediction_type == 'binary':
                        pred = model.predict(input_data)[0]
                        prob = model.predict_proba(input_data)[0]
                        confidence = max(prob)
                        return {'prediction': bool(pred), 'confidence': confidence}
                    elif prediction_type == 'class':
                        return {'prediction': int(model.predict(input_data)[0])}
                    elif prediction_type in ['position', 'reflectance', 'loss']:
                        pred = model.predict(input_data)[0]
                        if prediction_type == 'position':
                            position = np.clip(np.round(float(pred) / 0.01) * 0.01, 0, 0.30)
                            return {'prediction': position}
                        return {'prediction': float(pred)}
            except Exception as e:
                print(f"Prediction error with {model_path}: {e}")

    # Fallback to simulated output
    if prediction_type == 'binary':
        has_fault = np.random.choice([True, False], p=[0.7, 0.3])
        confidence = np.random.uniform(0.85, 0.99)
        return {'prediction': has_fault, 'confidence': confidence}
    elif prediction_type == 'class':
        return {'prediction': np.random.choice(list(range(8)))}
    elif prediction_type == 'position':
        return {'prediction': np.random.uniform(0.05, 0.95)}
    elif prediction_type == 'reflectance':
        return {'prediction': np.random.uniform(-0.5, 0.5)}
    elif prediction_type == 'loss':
        return {'prediction': np.random.uniform(-0.5, 0.5)}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        otdr_trace_points = []
        snr = float(request.form.get('snr', 10.0))
        for i in range(1, 31):
            otdr_trace_points.append(float(request.form.get(f'p{i}', 0.5)))

        sample_data = {'SNR': snr}
        for i, val in enumerate(otdr_trace_points):
            sample_data[f'P{i+1}'] = val
        sample_series = pd.Series(sample_data)
        otdr_trace = np.array(otdr_trace_points)

        # Load models
        binary_model_path = os.path.join(MODEL_DIR, "binary_model.h5") if os.path.exists(os.path.join(MODEL_DIR, "binary_model.h5")) else None
        class_model_path = os.path.join(MODEL_DIR, "class_model.h5") if os.path.exists(os.path.join(MODEL_DIR, "class_model.h5")) else None
        position_model_path = os.path.join(MODEL_DIR, "position_model.joblib") if os.path.exists(os.path.join(MODEL_DIR, "position_model.joblib")) else None
        reflectance_model_path = os.path.join(MODEL_DIR, "reflectance_model.joblib") if os.path.exists(os.path.join(MODEL_DIR, "reflectance_model.joblib")) else None
        loss_model_path = os.path.join(MODEL_DIR, "loss_model.joblib") if os.path.exists(os.path.join(MODEL_DIR, "loss_model.joblib")) else None
        scaler_path = os.path.join(MODEL_DIR, "scaler.joblib") if os.path.exists(os.path.join(MODEL_DIR, "scaler.joblib")) else None

        # Preprocessing for all models
        input_features = preprocess_for_binary(sample_series)

        # Step 1: Binary Prediction
        binary_prediction = simulate_model_prediction(binary_model_path, input_features, 'binary')

        # Step 2: Detailed Analysis (only if fault detected or forced)
        if binary_prediction['prediction']:
            # Apply scaler for classification models
            if scaler_path:
                scaler = load_model_safe(scaler_path)
                if scaler:
                    normalized_input = scaler.transform(input_features)
                else:
                    normalized_input = input_features
            else:
                normalized_input = input_features

            class_prediction = simulate_model_prediction(class_model_path, normalized_input, 'class')
            position_prediction = simulate_model_prediction(position_model_path, input_features, 'position')
            reflectance_prediction = simulate_model_prediction(reflectance_model_path, input_features, 'reflectance')
            loss_prediction = simulate_model_prediction(loss_model_path, input_features, 'loss')

            detailed_predictions = {
                'class': {'value': class_prediction['prediction'], 'name': FAULT_CLASSES.get(class_prediction['prediction'])},
                'position': {'value': position_prediction['prediction'], 'distance_km': position_prediction['prediction'] * 100},
                'reflectance': {'value': reflectance_prediction['prediction']},
                'loss': {'value': loss_prediction['prediction']},
            }
        else:
            detailed_predictions = None

        return render_template("index.html",
                               snr=snr,
                               otdr_trace=list(otdr_trace),
                               binary_prediction=binary_prediction,
                               detailed_predictions=detailed_predictions,
                               fault_classes=FAULT_CLASSES)

    # For GET requests
    return render_template("index.html",
                           snr=10.0,
                           otdr_trace=list(np.linspace(1.0, 0.1, 30) + np.random.normal(0, 0.05, 30)),
                           binary_prediction=None,
                           detailed_predictions=None,
                           fault_classes=FAULT_CLASSES)

if __name__ == "__main__":
    app.run(debug=True)