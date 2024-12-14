import os
import numpy as np

from tomorrow.script.ml.preprocessing import load_data, split_data, scale_data
from tomorrow.script.ml.train import train_lasso, train_elasticnet, train_ridge, train_sgd
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
    lasso_model = train_lasso(x_train_scaled, y_train_scaled)
    elasticnet_model = train_elasticnet(x_train_scaled, y_train_scaled)
    ridge_model = train_ridge(x_train_scaled, y_train_scaled)
    sgd_model = train_sgd(x_train_scaled, y_train_scaled)

    # Step 3: Evaluate models
    print("Lasso Score:", evaluate_model(lasso_model, x_test_scaled, y_test_scaled))
    print("ElasticNet Score:", evaluate_model(elasticnet_model, x_test_scaled, y_test_scaled))
    print("Ridge Score:", evaluate_model(ridge_model, x_test_scaled, y_test_scaled))
    print("SGD Score:", evaluate_model(sgd_model, x_test_scaled, y_test_scaled))

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

    # Load Lasso model
    lasso_model_loaded = load_model(os.path.join(MODEL_DIR, "lasso_model.joblib"))
    lasso_prediction = predict(lasso_model_loaded, scaler_x, scaler_y, input_data)

    # Load ElasticNet model
    elasticnet_model_loaded = load_model(os.path.join(MODEL_DIR, "elasticnet_model.joblib"))
    elasticnet_prediction = predict(elasticnet_model_loaded, scaler_x, scaler_y, input_data)

    # Load Ridge model
    ridge_model_loaded = load_model(os.path.join(MODEL_DIR, "ridge_model.joblib"))
    ridge_prediction = predict(ridge_model_loaded, scaler_x, scaler_y, input_data)

    # Load SGD model
    sgd_model_loaded = load_model(os.path.join(MODEL_DIR, "sgd_model.joblib"))
    sgd_prediction = predict(sgd_model_loaded, scaler_x, scaler_y, input_data)

    # Display the results
    print("\nPredictions Comparison:")
    print("======================================")
    print("Actual Values:")
    print(f"NO: {actual_values[0]:.2f}, NO2: {actual_values[1]:.2f}, O3: {actual_values[2]:.2f}, RM10: {actual_values[3]:.2f}")
    print("======================================")

    # Lasso Predictions
    print("Lasso Predictions:")
    print(f"Predicted NO: {lasso_prediction[0][0]:.2f}, Error: {abs(lasso_prediction[0][0] - actual_values[0]):.2f}")
    print(f"Predicted NO2: {lasso_prediction[0][1]:.2f}, Error: {abs(lasso_prediction[0][1] - actual_values[1]):.2f}")
    print(f"Predicted O3: {lasso_prediction[0][2]:.2f}, Error: {abs(lasso_prediction[0][2] - actual_values[2]):.2f}")
    print(f"Predicted RM10: {lasso_prediction[0][3]:.2f}, Error: {abs(lasso_prediction[0][3] - actual_values[3]):.2f}")
    print("======================================")

    # ElasticNet Predictions
    print("ElasticNet Predictions:")
    print(f"Predicted NO: {elasticnet_prediction[0][0]:.2f}, Error: {abs(elasticnet_prediction[0][0] - actual_values[0]):.2f}")
    print(f"Predicted NO2: {elasticnet_prediction[0][1]:.2f}, Error: {abs(elasticnet_prediction[0][1] - actual_values[1]):.2f}")
    print(f"Predicted O3: {elasticnet_prediction[0][2]:.2f}, Error: {abs(elasticnet_prediction[0][2] - actual_values[2]):.2f}")
    print(f"Predicted RM10: {elasticnet_prediction[0][3]:.2f}, Error: {abs(elasticnet_prediction[0][3] - actual_values[3]):.2f}")
    print("======================================")

    # Ridge Predictions
    print("Ridge Predictions:")
    print(f"Predicted NO: {ridge_prediction[0][0]:.2f}, Error: {abs(ridge_prediction[0][0] - actual_values[0]):.2f}")
    print(f"Predicted NO2: {ridge_prediction[0][1]:.2f}, Error: {abs(ridge_prediction[0][1] - actual_values[1]):.2f}")
    print(f"Predicted O3: {ridge_prediction[0][2]:.2f}, Error: {abs(ridge_prediction[0][2] - actual_values[2]):.2f}")
    print(f"Predicted RM10: {ridge_prediction[0][3]:.2f}, Error: {abs(ridge_prediction[0][3] - actual_values[3]):.2f}")
    print("======================================")

    # SGD Predictions
    print("SGD Predictions:")
    print(f"Predicted NO: {sgd_prediction[0][0]:.2f}, Error: {abs(sgd_prediction[0][0] - actual_values[0]):.2f}")
    print(f"Predicted NO2: {sgd_prediction[0][1]:.2f}, Error: {abs(sgd_prediction[0][1] - actual_values[1]):.2f}")
    print(f"Predicted O3: {sgd_prediction[0][2]:.2f}, Error: {abs(sgd_prediction[0][2] - actual_values[2]):.2f}")
    print(f"Predicted RM10: {sgd_prediction[0][3]:.2f}, Error: {abs(sgd_prediction[0][3] - actual_values[3]):.2f}")
    print("======================================")

if __name__ == "__main__":
    main()