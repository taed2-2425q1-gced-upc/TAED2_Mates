""" Module to test function predict_single from mates.modeling.predict"""

import pickle as pk
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from PIL import Image

from mates.config import IMG_SIZE, PROCESSED_DATA_DIR, RAW_DATA_DIR
from mates.features.features import load_model, read_labels
from mates.modeling.predict import predict_single


@pytest.fixture
def dbc_model():
    """Load model to test."""
    return load_model("mobilenet_exp_batch_62")


def test_predict_single(dbc_model):
    """
    Test the `predict_single` function of the dog breed classification model.

    This test validates the model's predictions against a set of known
    dog breeds using validation images. It checks that:
    - Predictions are made for each image.
    - Predicted breeds are valid and in the expected list.
    - The model achieves an accuracy above a specified threshold.

    The test uses a maximum of 500 validation images to ensure efficiency.
    It computes the accuracy by comparing predicted classes with true labels.

    Parameters:
    -----------
    dbc_model : keras.Model
        The pre-trained model for dog breed classification.

    Assertions:
    -----------
    - Each predicted breed is a valid breed.
    - The accuracy of the model is greater than or equal to 0.8.
    """
    _, breeds = read_labels(RAW_DATA_DIR)

    # Load validation data
    with open(PROCESSED_DATA_DIR / "y_valid.pkl", "rb") as f:
        y_valid = pk.load(f)
    with open(PROCESSED_DATA_DIR / "x_valid.pkl", "rb") as f:
        x_valid = pk.load(f)

    expected = [breeds[np.argmax(y)] for y in y_valid]

    # Initialize counters
    correct_predictions = 0
    num_tests = min(len(x_valid), 500)  # Limit tests to 500 samples

    for i in range(num_tests):
        img_path = Path(x_valid[i])
        with Image.open(img_path) as image:
            # Predict the breed
            predicted_breed = predict_single(dbc_model, breeds, image)
            assert (
                predicted_breed in breeds
            ), f"Predicted breed '{predicted_breed}' is not in dog breeds list."
            if predicted_breed == expected[i]:
                correct_predictions += 1

    accuracy = correct_predictions / num_tests
    assert accuracy >= 0.8, f"Expected accuracy > 0.8, but got {accuracy:.4f}"


@mock.patch("mates.modeling.predict.Image")
def test_predict_single_image_conversion(mock_image, dbc_model):
    """
    Test the `predict_single` function to ensure it converts images to RGB
    format when they are not already in that format.

    Parameters:
    -----------
    dbc_model : keras.Model
        The pre-trained model for dog breed classification.

    Assertions:
    -----------
    - The predicted breed is valid and in the expected list.
    """
    _, breeds = read_labels(RAW_DATA_DIR)
    # Create a mock grayscale image
    mock_image_instance = mock_image.new("L", (IMG_SIZE, IMG_SIZE))  # Mock grayscale image
    rgb_image = Image.open("tests/pod.jpg")
    mock_image_instance.convert.return_value = rgb_image

    # Call predict_single with the mocked image
    predicted_breed = predict_single(dbc_model, breeds, mock_image_instance)

    # Assert that the convert method was called once with 'RGB'
    mock_image_instance.convert.assert_called_once_with("RGB")

    # Validate the prediction
    assert (
        predicted_breed in breeds
    ), f"Predicted breed '{predicted_breed}' is not in dog breeds list."
