import os
import numpy as np
import tensorflow as tf
import torch
import joblib

from tomorrow.script.ml.preprocessing import load_data, split_data, scale_data
from tomorrow.script.ml.train import train_prophet
from tomorrow.script.ml.evaluate import evaluate_model
from tomorrow.script.ml.predict import load_model, load_scalers, predict, predict_without_scaling

from tomorrow.script.ml import MODEL_DIR,SCALER_DIR
from tomorrow import DATA_PATH,FEATURES,TARGETS

def main():
    # Step 1: Preprocessing
    data = load_data(DATA_PATH)

    # Train Prophet models for each target
    target_columns = ['NO', 'NO2', 'O3', 'RM10']
    prophet_models = {}
    for target in target_columns:
        prophet_models[target] = train_prophet(data, target)

    # Step 2: Evaluate models
    for target in target_columns:
        model_path = os.path.join(MODEL_DIR, f"prophet_model_{target}.joblib")
        model = joblib.load(model_path)
        future = model.make_future_dataframe(periods=365)
        forecast = model.predict(future)
        print(f"{target} Prophet Model Forecast:")
        print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    # Step 3: Prediction example
    # Actual data: NO, NO2, AirTemp, Press, UMR, O3, RM10
    # Input features: AirTemp, Press, UMR
    input_data = np.array([[8.9, 952.0, 59.4]])  # AirTemp, Press, UMR
    actual_values = np.array([95.16, 248.8, 17.86, 109.87])  # Actual NO, NO2, O3, RM10 values

    # Load the scalers
    scaler_x, scaler_y = load_scalers(
        os.path.join(SCALER_DIR, "scaler_x.joblib"),
        os.path.join(SCALER_DIR, "scaler_y.joblib")
    )

    # Display the results
    print("\nPredictions Comparison:")
    print("======================================")
    print("Actual Values:")
    print(f"NO: {actual_values[0]:.2f}, NO2: {actual_values[1]:.2f}, O3: {actual_values[2]:.2f}, RM10: {actual_values[3]:.2f}")
    print("======================================")

    for target in target_columns:
        model_path = os.path.join(MODEL_DIR, f"prophet_model_{target}.joblib")
        model = joblib.load(model_path)
        future = model.make_future_dataframe(periods=1)
        forecast = model.predict(future)
        prediction = forecast[['yhat']].iloc[-1].values[0]
        print(f"{target} Prophet Prediction: {prediction:.2f}")

if __name__ == "__main__":
    main()