"""Handles all data preparation tasks like reading the dataset, splitting the data, and scaling"""

import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from tomorrow.script.ml import SCALER_DIR

def load_data(data_path):
    """Load the dataset."""
    data = pd.read_csv(data_path, parse_dates=['Date'], dayfirst=True)
    return data

def split_data(data, features, targets, test_size=0.3, random_state=0):
    """Split data into train and test sets."""
    x = data[features]
    y = data[targets]
    return train_test_split(x, y, test_size=test_size, random_state=random_state)

def scale_data(x_train, x_test, y_train, y_test):
    """Scale the data using RobustScaler and save the scalers."""
    scaler_x = RobustScaler()
    scaler_x.fit(x_train)
    x_train_scaled = scaler_x.transform(x_train)
    x_test_scaled = scaler_x.transform(x_test)

    scaler_y = RobustScaler()
    scaler_y.fit(y_train)
    y_train_scaled = scaler_y.transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    # Save scalers
    joblib.dump(scaler_x, os.path.join(SCALER_DIR, "scaler_x.joblib"))
    joblib.dump(scaler_y, os.path.join(SCALER_DIR, "scaler_y.joblib"))

    return x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled