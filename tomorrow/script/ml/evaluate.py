"""Handles model evaluation."""

import torch
import torch.nn as nn

def evaluate_model(model, x_test, y_test):
    if hasattr(model, 'score'):
        # For scikit-learn models
        score = model.score(x_test, y_test)
    elif hasattr(model, 'evaluate'):
        # For Keras models
        loss = model.evaluate(x_test, y_test, verbose=0)
        score = -loss  # Assuming lower loss is better, convert to a score
    elif isinstance(model, nn.Module):
        # For PyTorch models
        model.eval()
        criterion = nn.MSELoss()
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        with torch.no_grad():
            outputs = model(x_test_tensor)
            loss = criterion(outputs, y_test_tensor).item()
        score = -loss  # Assuming lower loss is better, convert to a score
    else:
        raise ValueError("Model does not have a score or evaluate method.")
    return score