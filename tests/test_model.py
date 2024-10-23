""" Module to test the model. """

import os
import pickle as pk

import numpy as np
import pytest

from mates.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, TEST_DIR
from mates.features.features import (create_batches, load_model,
                                     load_processed_data, read_labels)

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.fixture
def dbc_model():
    """
    Load the dog breed classification model for testing.
    """
    return load_model("mobilenet_exp_batch_62")


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS,
    reason="Test doesn't work in Github Actions. \
                    Enough with test_dbc_model",
)
def test_dbc_model_validation(dbc_model):
    """
    Test for dog breed classification model.

    This test verifies that the model makes valid predictions with
    the valid dataset images and achieves an acceptable accuracy 
    on the validation dataset.
    """
    # Load validation labels
    with open(PROCESSED_DATA_DIR / "y_valid.pkl", "rb") as f:
        y_valid = pk.load(f)
    # Load validation data
    _, valid_data, _ = load_processed_data(32)

    # Make predictions
    y_pred = dbc_model.predict(valid_data)

    # Assertions on predictions
    assert all(len(pred) == 120 for pred in y_pred), "Each prediction should have 120 classes."
    assert all(
        round(sum(pred)) == 1 for pred in y_pred
    ), "Each prediction should be one-hot encoded."

    # Convert predictions and true labels to class indices
    predicted_classes = np.argmax(y_pred, axis=1)
    true_classes = np.argmax(y_valid, axis=1)
    print(predicted_classes[0:100])
    print(true_classes[0:100])

    # Calculate accuracy
    correct_predictions = np.sum(predicted_classes == true_classes)
    print(correct_predictions)
    total_predictions = len(predicted_classes)
    accuracy = correct_predictions / total_predictions

    # Assert that the accuracy is above a certain threshold
    assert accuracy > 0.8, f"Expected accuracy > 0.8, but got {accuracy:.4f}"


def test_dbc_model(dbc_model):
    """
    Test for dog breed classification model.

    This test verifies that the model makes valid predictions on 10 sample
    images and achieves an acceptable accuracy on the validation dataset.
    """
    image_paths = [
        TEST_DIR / f"test_images/{img}"
        for img in [
            "0021f9ceb3235effd7fcde7f7538ed62.jpg",
            "008887054b18ba3c7601792b6a453cc3.jpg",
            "00e9ed3fab1d2032603d1a90e557976f.jpg",
            "010d65bd29d246aea53d9849da142ccf.jpg",
            "0120343862761d052d6a7ade81625c94.jpg",
            "0161847d5bfae154af084993fa26a1a7.jpg",
            "0311a6a51a414ba91f3ad8055170baa1.jpg",
            "0365fe4e3e13a885c6b02ebbbf2d9173.jpg",
            "03ae71ff9e4c5ac97afdfa956ea5e191.jpg",
            "03f6435dc8a5e760d19e67831c6034fb.jpg",
        ]
    ]
    true_classes = [
        "golden_retriever",
        "boxer",
        "weimaraner",
        "boston_bull",
        "dingo",
        "giant_schnauzer",
        "lhasa",
        "cairn",
        "borzoi",
        "chow",
    ]
    _, el = read_labels(RAW_DATA_DIR)

    x_test = create_batches(15, image_paths, test_data=True)
    y_pred = dbc_model.predict(x_test)
    # Assertions on predictions
    assert all(len(pred) == 120 for pred in y_pred), "Each prediction should have 120 classes."
    assert all(
        round(sum(pred)) == 1 for pred in y_pred
    ), "Each prediction should be one-hot encoded."

    # Calculate accuracy
    predicted_classes = np.array([el[np.argmax(pred)] for pred in y_pred])
    correct_predictions = np.sum(predicted_classes == true_classes)
    accuracy = correct_predictions / 10

    # Assert that the accuracy is above a certain threshold
    assert accuracy > 0.8, f"Expected accuracy > 0.8, but got {accuracy:.4f}"
