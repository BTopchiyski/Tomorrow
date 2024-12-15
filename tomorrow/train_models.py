import os
import numpy as np
import tensorflow as tf
import torch

from tomorrow.script.ml.preprocessing import load_data, split_data, scale_data
from tomorrow.script.ml.train import train_pytorch_nn, train_keras_nn
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

    # # Step 2: Train models
    # keras_nn_model = train_keras_nn(x_train_scaled, y_train_scaled)
    # pytorch_nn_model = train_pytorch_nn(x_train_scaled, y_train_scaled)


    # Step 3: Evaluate models
    print("Keras Neural Network Score:", evaluate_model(keras_nn_model, x_test_scaled, y_test_scaled))
    print("PyTorch Neural Network Score:", evaluate_model(pytorch_nn_model, x_test_scaled, y_test_scaled))

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

    # Load Keras Neural Network model
    keras_nn_model_loaded = tf.keras.models.load_model(os.path.join(MODEL_DIR, "keras_nn_model.keras"))
    keras_nn_prediction = keras_nn_model_loaded.predict(scaler_x.transform(input_data))

    # Load PyTorch Neural Network model
    pytorch_nn_model_loaded = PyTorchNN(input_data.shape[1], actual_values.shape[0])
    pytorch_nn_model_loaded.load_state_dict(torch.load(os.path.join(MODEL_DIR, "pytorch_nn_model.pth")))
    pytorch_nn_model_loaded.eval()
    with torch.no_grad():
        pytorch_nn_prediction = pytorch_nn_model_loaded(torch.tensor(scaler_x.transform(input_data), dtype=torch.float32)).numpy()

    # Display the results
    print("\nPredictions Comparison:")
    print("======================================")
    print("Actual Values:")
    print(f"NO: {actual_values[0]:.2f}, NO2: {actual_values[1]:.2f}, O3: {actual_values[2]:.2f}, RM10: {actual_values[3]:.2f}")
    print("======================================")

    # Keras Neural Network Predictions
    print("Keras Neural Network Predictions:")
    print(f"Predicted NO: {keras_nn_prediction[0][0]:.2f}, Error: {abs(keras_nn_prediction[0][0] - actual_values[0]):.2f}")
    print(f"Predicted NO2: {keras_nn_prediction[0][1]:.2f}, Error: {abs(keras_nn_prediction[0][1] - actual_values[1]):.2f}")
    print(f"Predicted O3: {keras_nn_prediction[0][2]:.2f}, Error: {abs(keras_nn_prediction[0][2] - actual_values[2]):.2f}")
    print(f"Predicted RM10: {keras_nn_prediction[0][3]:.2f}, Error: {abs(keras_nn_prediction[0][3] - actual_values[3]):.2f}")
    print("======================================")

    # PyTorch Neural Network Predictions
    print("PyTorch Neural Network Predictions:")
    print(f"Predicted NO: {pytorch_nn_prediction[0][0]:.2f}, Error: {abs(pytorch_nn_prediction[0][0] - actual_values[0]):.2f}")
    print(f"Predicted NO2: {pytorch_nn_prediction[0][1]:.2f}, Error: {abs(pytorch_nn_prediction[0][1] - actual_values[1]):.2f}")
    print(f"Predicted O3: {pytorch_nn_prediction[0][2]:.2f}, Error: {abs(pytorch_nn_prediction[0][2] - actual_values[2]):.2f}")
    print(f"Predicted RM10: {pytorch_nn_prediction[0][3]:.2f}, Error: {abs(pytorch_nn_prediction[0][3] - actual_values[3]):.2f}")
    print("======================================")

if __name__ == "__main__":
    main()