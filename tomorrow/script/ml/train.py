"""Handles model training and saving."""

import os
import torch
import joblib
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
import tensorflow as tf

from sklearn.svm import SVR
from sklearn.linear_model import Lasso, ElasticNet, Ridge, SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from torch.utils.data import DataLoader, TensorDataset

from prophet import Prophet

from tomorrow.script.ml import MODEL_DIR

def train_prophet(data, target_column):
    """Train a time series model using Facebook Prophet."""
    print("Training Facebook Prophet model...")

    # Prepare the data
    df = data[['Date', target_column]].rename(columns={'Date': 'ds', target_column: 'y'})

    # Initialize and train the model
    model = Prophet()
    model.fit(df)

    # Save the model
    model_path = os.path.join(MODEL_DIR, f"prophet_model_{target_column}.joblib")
    joblib.dump(model, model_path)
    print(f"Facebook Prophet model saved to {model_path}")
    return model

def train_keras_nn(x_train, y_train):
    """Train a Neural Network using Keras."""
    print("Training Keras Neural Network...")
    model = Sequential()
    model.add(Dense(64, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(y_train.shape[1]))

    model.compile(optimizer='adam', loss='mse')
    model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1)

    model_path = os.path.join(MODEL_DIR, "keras_nn_model.keras")
    model.save(model_path)
    print(f"Keras Neural Network model saved to {model_path}")
    return model

class PyTorchNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PyTorchNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_pytorch_nn(x_train, y_train):
    """Train a Neural Network using PyTorch."""
    print("Training PyTorch Neural Network...")
    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]

    model = PyTorchNN(input_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    dataset = TensorDataset(x_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(100):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model_path = os.path.join(MODEL_DIR, "pytorch_nn_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"PyTorch Neural Network model saved to {model_path}")
    return model

def train_svr(x_train, y_train):
    """Train a Support Vector Regressor model."""
    print("Training Support Vector Regressor...")
    svr = SVR()
    multi_output_svr = MultiOutputRegressor(svr)
    multi_output_svr.fit(x_train, y_train)
    model_path = os.path.join(MODEL_DIR, "svr_model.joblib")
    joblib.dump(multi_output_svr, model_path)
    print(f"Support Vector Regressor model saved to {model_path}")
    return multi_output_svr

def train_gbr(x_train, y_train):
    """Train a Gradient Boosting Regressor model."""
    print("Training Gradient Boosting Regressor...")
    gbr = GradientBoostingRegressor()
    multi_output_gbr = MultiOutputRegressor(gbr)
    multi_output_gbr.fit(x_train, y_train)
    model_path = os.path.join(MODEL_DIR, "gbr_model.joblib")
    joblib.dump(multi_output_gbr, model_path)
    print(f"Gradient Boosting Regressor model saved to {model_path}")
    return multi_output_gbr

def train_xgboost(x_train, y_train):
    """Train an XGBoost Regressor model."""
    print("Training XGBoost Regressor...")
    xgboost = xgb.XGBRegressor()
    multi_output_xgboost = MultiOutputRegressor(xgboost)
    multi_output_xgboost.fit(x_train, y_train)
    model_path = os.path.join(MODEL_DIR, "xgboost_model.joblib")
    joblib.dump(multi_output_xgboost, model_path)
    print(f"XGBoost Regressor model saved to {model_path}")
    return multi_output_xgboost

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