""" Module to test the model """
import pickle as pk
import pytest
import numpy as np
from mates.config import PROCESSED_DATA_DIR
from mates.features.features import load_model, load_processed_data

@pytest.fixture
def dbc_model():
    """
    Load model to test
    """
    return load_model("mobilenet_exp_batch_62")



def test_dbc_model(dbc_model):
    """"
    Test for dog breed classification model
    """
    # Load test data
    with open(PROCESSED_DATA_DIR / 'y_valid.pkl', 'rb') as f:
        y_valid = pk.load(f)
    _, valid_data, _ = load_processed_data(32)

    # Make predictions
    y_pred = dbc_model.predict(valid_data)

    assert all(len(pred) == 120 for pred in y_pred)
    assert all(round(sum(pred)) == 1 for pred in y_pred)

    # Convert predictions and true labels to class indices
    predicted_classes = np.argmax(y_pred, axis=1)
    true_classes = np.argmax(y_valid, axis=1)

    # Calculate accuracy
    correct_predictions = np.sum(predicted_classes == true_classes)
    total_predictions = len(predicted_classes)
    # Calculate accuracy percentage
    accuracy = correct_predictions / total_predictions

    # Assert that the accuracy is above a certain threshold
    assert accuracy > 0.5, f"Expected accuracy > 0.5, but got {accuracy:.4f}"
