<<<<<<< HEAD
# OTDR Fault Detector Web App

This project is a web application that loads a pre-trained Keras `.h5` model and a `scaler.joblib` for preprocessing. The app allows users to upload a CSV file with columns `SNR`, `P1` to `P30`, preprocesses the data, and displays the predicted class.

---

## Phase 1: Environment Setup

1. **Clone the repository** (if applicable) or download the project files.
2. **Install Python 3.8+** (if not already installed).
3. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```
4. **Activate the virtual environment:**
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
5. **Install required packages:**
   Create a `requirements.txt` with the following content:
   ```
   flask
   pandas
   numpy
   scikit-learn
   tensorflow
   joblib
   ```
   Then run:
   ```bash
   pip install -r requirements.txt
   ```

---

## Phase 2: Backend Development

1. **Create a Flask app** (`app.py`) that:
   - Loads `otdr_fault_detectorv1.h5` and `scaler.joblib` at startup.
   - Provides an endpoint to upload a CSV file.
   - Reads and preprocesses the CSV using the scaler.
   - Passes the preprocessed data to the model for prediction.
   - Returns the predicted class(es) as a response.

2. **Sample Flask code structure:**
   - Load model and scaler at the top.
   - Use `pandas` to read the uploaded CSV.
   - Use `joblib` to load the scaler.
   - Use `tensorflow.keras.models.load_model` to load the `.h5` model.

---

## Phase 3: Frontend Development

1. **Create a simple HTML page** (`templates/index.html`) with:
   - A file upload form for the CSV.
   - A submit button.
   - A section to display the prediction result.

2. **Use Flask to render the HTML page** and handle form submissions.

---

## Phase 4: Running the App

1. **Start the Flask server:**
   ```bash
   python app.py
   ```
2. **Open your browser** and go to `http://127.0.0.1:5000/`.
3. **Upload your CSV file** and view the predicted class.

---

## Phase 5: (Optional) Deployment

- Deploy the app to a cloud platform (Heroku, Azure, AWS, etc.) if needed.
- Ensure all model and scaler files are included in the deployment.

---

## Example File Structure

```
.
├── app.py
├── scaler.joblib
├── otdr_fault_detectorv1.h5
├── requirements.txt
├── templates/
│   └── index.html
└── static/
```

---

## Notes

- The CSV file must have columns: `SNR`, `P1`, `P2`, ..., `P30`.
- The app will preprocess the data using the provided scaler before making predictions.
- The predicted class will be displayed on the web page after upload. 
=======
# Optical-Fiber-Fault-Localization
>>>>>>> 72ee6d72bc6f863950e93fff5914e4df5091ed66
