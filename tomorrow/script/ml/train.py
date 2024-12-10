"""Handles model training and saving."""

import os
import joblib

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from tomorrow.script.ml import MODEL_DIR

def train_mlp(x_train, y_train):
    """Train an MLPRegressor and save the model."""
    mlp = MLPRegressor(max_iter=1000, verbose=0, n_iter_no_change=17)
    mlp.fit(x_train, y_train)
    joblib.dump(mlp, os.path.join(MODEL_DIR, "mlp_model.joblib"))
    return mlp

def train_rf(x_train, y_train):
    """Train a RandomForestRegressor and save the model."""
    rf = RandomForestRegressor()
    rf.fit(x_train, y_train)
    joblib.dump(rf, os.path.join(MODEL_DIR, "rf_model.joblib"))
    return rf

def train_dt(x_train, y_train):
    """Train a DecisionTreeRegressor and save the model."""
    dt = DecisionTreeRegressor()
    dt.fit(x_train, y_train)
    joblib.dump(dt, os.path.join(MODEL_DIR, "dt_model.joblib"))
    return dt