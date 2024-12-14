"""Handles predictions using trained models."""

import numpy as np
import pandas as pd
import joblib

from tomorrow import FEATURES

def load_scaler_and_model(model_path, scaler_x_path, scaler_y_path):
    """Load a trained model and scalers."""
    model = joblib.load(model_path)
    scaler_x = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    return model, scaler_x, scaler_y

def predict(model, scaler_x, scaler_y, input_data):
    """Make predictions using the trained model."""
    input_data_df = pd.DataFrame(input_data, columns=FEATURES)
    input_scaled = scaler_x.transform(input_data_df)
    prediction_scaled = model.predict(input_scaled)
    return scaler_y.inverse_transform(prediction_scaled)