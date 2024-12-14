import os
import numpy as np

from tomorrow.script.ml.preprocessing import load_data, split_data, scale_data
from tomorrow.script.ml.train import train_mlp, train_rf, train_dt
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
    mlp_model = train_mlp(x_train_scaled, y_train_scaled)
    rf_model = train_rf(x_train_scaled, y_train_scaled)
    dt_model = train_dt(x_train_scaled, y_train_scaled)

    # Step 3: Evaluate models
    print("MLP Regressor Score:", evaluate_model(mlp_model, x_test_scaled, y_test_scaled))
    print("Random Forest Score:", evaluate_model(rf_model, x_test_scaled, y_test_scaled))
    print("Decision Tree Score:", evaluate_model(dt_model, x_test_scaled, y_test_scaled))

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

    # Load MLP model
    mlp_model = load_model(os.path.join(MODEL_DIR, "mlp_model.joblib"))
    mlp_prediction = predict(mlp_model, scaler_x, scaler_y, input_data)

    # Load Random Forest model
    rf_model = load_model(os.path.join(MODEL_DIR, "rf_model.joblib"))
    rf_prediction = predict_without_scaling(rf_model, scaler_x, scaler_y, input_data)

    # Load Decision Tree model
    dt_model = load_model(os.path.join(MODEL_DIR, "dt_model.joblib"))
    dt_prediction = predict_without_scaling(dt_model, scaler_x, scaler_y, input_data)

    # Display the results
    print("\nPredictions Comparison:")
    print("======================================")
    print("Actual Values:")
    print(f"NO: {actual_values[0]:.2f}, NO2: {actual_values[1]:.2f}, O3: {actual_values[2]:.2f}, RM10: {actual_values[3]:.2f}")
    print("======================================")

    # MLP Predictions
    print("MLP Predictions:")
    print(f"Predicted NO: {mlp_prediction[0][0]:.2f}, Error: {abs(mlp_prediction[0][0] - actual_values[0]):.2f}")
    print(f"Predicted NO2: {mlp_prediction[0][1]:.2f}, Error: {abs(mlp_prediction[0][1] - actual_values[1]):.2f}")
    print(f"Predicted O3: {mlp_prediction[0][2]:.2f}, Error: {abs(mlp_prediction[0][2] - actual_values[2]):.2f}")
    print(f"Predicted RM10: {mlp_prediction[0][3]:.2f}, Error: {abs(mlp_prediction[0][3] - actual_values[3]):.2f}")
    print("======================================")

    # Random Forest Predictions
    print("Random Forest Predictions:")
    print(f"Predicted NO: {rf_prediction[0][0]:.2f}, Error: {abs(rf_prediction[0][0] - actual_values[0]):.2f}")
    print(f"Predicted NO2: {rf_prediction[0][1]:.2f}, Error: {abs(rf_prediction[0][1] - actual_values[1]):.2f}")
    print(f"Predicted O3: {rf_prediction[0][2]:.2f}, Error: {abs(rf_prediction[0][2] - actual_values[2]):.2f}")
    print(f"Predicted RM10: {rf_prediction[0][3]:.2f}, Error: {abs(rf_prediction[0][3] - actual_values[3]):.2f}")
    print("======================================")

    # Decision Tree Predictions
    print("Decision Tree Predictions:")
    print(f"Predicted NO: {dt_prediction[0][0]:.2f}, Error: {abs(dt_prediction[0][0] - actual_values[0]):.2f}")
    print(f"Predicted NO2: {dt_prediction[0][1]:.2f}, Error: {abs(dt_prediction[0][1] - actual_values[1]):.2f}")
    print(f"Predicted O3: {dt_prediction[0][2]:.2f}, Error: {abs(dt_prediction[0][2] - actual_values[2]):.2f}")
    print(f"Predicted RM10: {dt_prediction[0][3]:.2f}, Error: {abs(dt_prediction[0][3] - actual_values[3]):.2f}")


if __name__ == "__main__":
    main()