import sys
import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

# Add parent directory to the Python path to import custom_logistic.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logistic_regression import LogisticRegression  # Import the custom class

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
try:
    with open('project/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('project/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("Model and scaler loaded successfully!")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model, scaler = None, None  # Fallback if loading fails

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return render_template('index.html', prediction_text="Model or scaler not loaded. Please check your setup.")

    # Check if file is uploaded
    if 'file' not in request.files or request.files['file'].filename == '':
        return render_template('index.html', prediction_text="No file uploaded. Please upload a CSV file.")

    # Read the uploaded file
    file = request.files['file']
    try:
        data = pd.read_csv(file)
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error reading file: {e}")

    # Ensure the data matches the model's expected input
    try:
        input_data = data.drop(columns=['Id'], errors='ignore').values  # Ignore 'Id' column if present
        input_data = np.nan_to_num(input_data)  # Handle NaN values
        input_data_scaled = scaler.transform(input_data)  # Scale the data
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error processing data: {e}")

    # Make predictions
    predictions = model.predict(input_data_scaled)
    probabilities = model.predict_proba(input_data_scaled)

    # Create results for display
    results = []
    for i, (prediction, probability) in enumerate(zip(predictions, probabilities)):
        results.append({
            "id": data.iloc[i, 0] if 'Id' in data.columns else i + 1,  # Use 'Id' if available, else row index
            "diagnosis": "Schizophrenia" if prediction == 1 else "Control",
            "confidence": f"{probability:.2f}"
        })

    return render_template('index.html', results=results)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
