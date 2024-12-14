import os
import numpy as np

from tomorrow.script.ml.preprocessing import load_data, split_data, scale_data
from tomorrow.script.ml.train import train_gbr, train_xgboost
from tomorrow.script.ml.evaluate import evaluate_model
from tomorrow.script.ml.predict import load_model, load_scalers, predict, predict_without_scaling

from tomorrow.script.ml import MODEL_DIR,SCALER_DIR
from tomorrow import DATA_PATH,FEATURES,TARGETS

def main():
    # Step 1: Preprocessing
    data = load_data(DATA_PATH)
    x_train, x_test, y_train, y_test = split_data(data, FEATURES, TARGETS)
    x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled = scale_data(
        x_train, x_test, y_train, y_test
    )

    # Step 2: Train models
    gbr_model = train_gbr(x_train_scaled, y_train_scaled)
    xgboost_model = train_xgboost(x_train_scaled, y_train_scaled)

    # Step 3: Evaluate models
    print("Gradient Boosting Regressor Score:", evaluate_model(gbr_model, x_test_scaled, y_test_scaled))
    print("XGBoost Regressor Score:", evaluate_model(xgboost_model, x_test_scaled, y_test_scaled))

    # Step 4: Prediction example
    # Actual data: NO, NO2, AirTemp, Press, UMR, O3, RM10
    # Input features: AirTemp, Press, UMR
    input_data = np.array([[8.9, 952.0, 59.4]])  # AirTemp, Press, UMR
    actual_values = np.array([95.16, 248.8, 17.86, 109.87])  # Actual NO, NO2, O3, RM10 values

    # Load the scalers
    scaler_x, scaler_y = load_scalers(
        os.path.join(SCALER_DIR, "scaler_x.joblib"),
        os.path.join(SCALER_DIR, "scaler_y.joblib")
    )

    # Load Gradient Boosting Regressor model
    gbr_model_loaded = load_model(os.path.join(MODEL_DIR, "gbr_model.joblib"))
    gbr_prediction = predict(gbr_model_loaded, scaler_x, scaler_y, input_data)

    # Load XGBoost model
    xgboost_model_loaded = load_model(os.path.join(MODEL_DIR, "xgboost_model.joblib"))
    xgboost_prediction = predict(xgboost_model_loaded, scaler_x, scaler_y, input_data)

    # Gradient Boosting Regressor Predictions
    print("Gradient Boosting Regressor Predictions:")
    print(f"Predicted NO: {gbr_prediction[0][0]:.2f}, Error: {abs(gbr_prediction[0][0] - actual_values[0]):.2f}")
    print(f"Predicted NO2: {gbr_prediction[0][1]:.2f}, Error: {abs(gbr_prediction[0][1] - actual_values[1]):.2f}")
    print(f"Predicted O3: {gbr_prediction[0][2]:.2f}, Error: {abs(gbr_prediction[0][2] - actual_values[2]):.2f}")
    print(f"Predicted RM10: {gbr_prediction[0][3]:.2f}, Error: {abs(gbr_prediction[0][3] - actual_values[3]):.2f}")
    print("======================================")

    # XGBoost Predictions
    print("XGBoost Predictions:")
    print(f"Predicted NO: {xgboost_prediction[0][0]:.2f}, Error: {abs(xgboost_prediction[0][0] - actual_values[0]):.2f}")
    print(f"Predicted NO2: {xgboost_prediction[0][1]:.2f}, Error: {abs(xgboost_prediction[0][1] - actual_values[1]):.2f}")
    print(f"Predicted O3: {xgboost_prediction[0][2]:.2f}, Error: {abs(xgboost_prediction[0][2] - actual_values[2]):.2f}")
    print(f"Predicted RM10: {xgboost_prediction[0][3]:.2f}, Error: {abs(xgboost_prediction[0][3] - actual_values[3]):.2f}")
    print("======================================")

if __name__ == "__main__":
    main()