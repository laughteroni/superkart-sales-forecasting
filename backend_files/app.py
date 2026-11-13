# ---------------------------------------------------------------
# SuperKart Sales Forecasting - Flask backend API
# ---------------------------------------------------------------
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Initialise the Flask application
superkart_api = Flask("SuperKart Sales Forecaster")

# Load the serialized pipeline (preprocessing + model) once at start-up
model = joblib.load("superkart_model.joblib")

# Features expected by the model, in training order
FEATURES = [
    "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_MRP",
    "Store_Size", "Store_Location_City_Type", "Store_Type", "Product_Id_char",
    "Store_Age_Years", "Product_Type_Category",
]
NUMERIC_FEATURES = ["Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Age_Years"]


def prepare_input(df):
    """Select the model features, fix data types and apply the same cleaning as in training."""
    df = df[FEATURES].copy()
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].astype(float)
    df["Product_Sugar_Content"] = df["Product_Sugar_Content"].replace("reg", "Regular")
    return df


@superkart_api.get("/")
def home():
    """Health-check endpoint."""
    return "Welcome to the SuperKart Sales Forecasting API!"


@superkart_api.post("/v1/predict")
def predict_sales():
    """Online inference: one product-store record sent as JSON."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be a JSON object."}), 400

    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing features: {missing}"}), 400

    try:
        sample = prepare_input(pd.DataFrame([data]))
        prediction = float(model.predict(sample)[0])
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 400

    return jsonify({"Predicted_Sales": round(prediction, 2)})


@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    """Batch inference: a CSV file uploaded under the form key 'file'."""
    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "Upload a CSV file using the form key 'file'."}), 400

    try:
        input_data = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Could not read CSV: {e}"}), 400

    missing = [f for f in FEATURES if f not in input_data.columns]
    if missing:
        return jsonify({"error": f"Missing columns: {missing}"}), 400

    # cast to float64 so the JSON shows clean 2-decimal values
    predictions = np.round(model.predict(prepare_input(input_data)).astype(float), 2).tolist()

    # Use Product_Id as the key if the file has it, otherwise the row index
    if "Product_Id" in input_data.columns:
        keys = input_data["Product_Id"].astype(str).tolist()
    else:
        keys = [str(i) for i in input_data.index]

    return jsonify(dict(zip(keys, predictions)))


# Run locally with: python app.py  (in Docker the app is served by gunicorn)
if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860, debug=False)
