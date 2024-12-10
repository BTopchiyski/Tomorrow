"""Handles model evaluation."""

def evaluate_model(model, x_test, y_test):
    score = model.score(x_test, y_test)
    return score