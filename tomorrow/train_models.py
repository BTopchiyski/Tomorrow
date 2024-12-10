import os
import numpy as np

from tomorrow.script.ml.preprocessing import load_data, split_data, scale_data
from tomorrow.script.ml.train import train_mlp, train_rf, train_dt
from tomorrow.script.ml.evaluate import evaluate_model
from tomorrow.script.ml.predict import load_scaler_and_model, predict

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
    input_data = np.array([[25, 955, 80]])
    mlp_model, scaler_x, scaler_y = load_scaler_and_model(
        os.path.join(MODEL_DIR, "mlp_model.joblib"),
        os.path.join(SCALER_DIR, "scaler_x.joblib"),
        os.path.join(SCALER_DIR, "scaler_y.joblib")
    )
    prediction = predict(mlp_model, scaler_x, scaler_y, input_data)
    print("MLP Prediction:", prediction)

if __name__ == "__main__":
    main()