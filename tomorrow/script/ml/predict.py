"""Handles predictions using trained models."""

import numpy as np
import pandas as pd
import joblib

from tomorrow import FEATURES

def load_model(model_path):
    """Load a trained model."""
    model = joblib.load(model_path)
    return model

def load_scalers(scaler_x_path, scaler_y_path):
    """Load scalers."""
    scaler_x = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    return scaler_x, scaler_y

def predict(model, scaler_x, scaler_y, input_data):
    """Make predictions using the trained model."""
    input_data_df = pd.DataFrame(input_data, columns=FEATURES)
    input_scaled = scaler_x.transform(input_data_df)
    prediction_scaled = model.predict(input_scaled)
    return scaler_y.inverse_transform(prediction_scaled)

def predict_without_scaling(model, scaler_x, scaler_y, input_data):
    """
    Predict without scaling the input for tree-based models like Random Forest and Decision Tree.
    Still applies output scaling (if applicable) for consistency in output comparison.
    """
    # Ensure input data is properly reshaped (if necessary)
    input_data = np.array(input_data).reshape(1, -1)

    # Apply feature scaling to the input for models that require it
    # For tree-based models, we skip scaling and use the raw input
    predictions = model.predict(input_data)

    # If the model produces scaled outputs, inverse_transform them to get real-world values
    # This is useful if we scaled the target during training
    if scaler_y:
        predictions = scaler_y.inverse_transform(predictions)

    return predictions