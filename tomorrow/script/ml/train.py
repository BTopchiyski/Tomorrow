"""Handles model training and saving."""

import os
import joblib

from sklearn.linear_model import Lasso, ElasticNet, Ridge, SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor

from tomorrow.script.ml import MODEL_DIR

def train_lasso(x_train, y_train):
    """Train a Lasso regression model."""
    print("Training Lasso Regression...")
    lasso = Lasso(max_iter=10000)
    lasso.fit(x_train, y_train)
    model_path = os.path.join(MODEL_DIR, "lasso_model.joblib")
    joblib.dump(lasso, model_path)
    print(f"Lasso model saved to {model_path}")
    return lasso

def train_elasticnet(x_train, y_train):
    """Train an ElasticNet regression model."""
    print("Training ElasticNet Regression...")
    elnet = ElasticNet(max_iter=10000)
    elnet.fit(x_train, y_train)
    model_path = os.path.join(MODEL_DIR, "elasticnet_model.joblib")
    joblib.dump(elnet, model_path)
    print(f"ElasticNet model saved to {model_path}")
    return elnet

def train_ridge(x_train, y_train):
    """Train a Ridge regression model."""
    print("Training Ridge Regression...")
    ridge = Ridge()
    ridge.fit(x_train, y_train)
    model_path = os.path.join(MODEL_DIR, "ridge_model.joblib")
    joblib.dump(ridge, model_path)
    print(f"Ridge model saved to {model_path}")
    return ridge

def train_sgd(x_train, y_train):
    """Train a SGD regression model."""
    print("Training SGD Regression...")
    sgd = SGDRegressor(penalty='l2', max_iter=50000, early_stopping=True, n_iter_no_change=25)
    multi_output_sgd = MultiOutputRegressor(sgd)
    multi_output_sgd.fit(x_train, y_train)
    model_path = os.path.join(MODEL_DIR, "sgd_model.joblib")
    joblib.dump(multi_output_sgd, model_path)
    print(f"SGD model saved to {model_path}")
    return multi_output_sgd

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