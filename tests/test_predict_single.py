""" Module to test function predict_single from mates.modeling.predict"""

from pathlib import Path
from unittest import mock

import pytest
from PIL import Image

from mates.config import IMG_SIZE, TEST_DIR
from mates.features.features import load_model
from mates.modeling.predict import predict_single


@pytest.fixture
def dbc_model():
    """Load model to test."""
    return load_model("mobilenet_exp_batch_62")


@pytest.fixture
def image_data():
    """
    Fixture providing image paths and true classes for testing.
    """
    image_paths = [
        TEST_DIR / f"test_images/{img}"
        for img in [
            "img1.jpg",
            "img2.jpg",
            "img3.jpg",
            "img4.jpg",
            "img5.jpg",
            "img6.jpg",
            "img7.jpg",
            "img8.jpg",
            "img9.jpg",
            "img10.jpg",
        ]
    ]
    true_classes = [
        "pug",
        "rottweiler",
        "german_shepherd",
        "newfoundland",
        "blenheim_spaniel",
        "saint_bernard",
        "mexican_hairless",
        "border_collie",
        "dingo",
        "beagle",
    ]

    dog_breeds = ["dog" for _ in range(120)]
    dog_breeds[88] = "pug"
    dog_breeds[91] = "rottweiler"
    dog_breeds[46] = "german_shepherd"
    dog_breeds[78] = "newfoundland"
    dog_breeds[13] = "blenheim_spaniel"
    dog_breeds[92] = "saint_bernard"
    dog_breeds[74] = "mexican_hairless"
    dog_breeds[16] = "border_collie"
    dog_breeds[37] = "dingo"
    dog_breeds[9] = "beagle"

    return image_paths, true_classes, dog_breeds


def test_predict_single(dbc_model, image_data):
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
    image_paths, true_classes, dog_breeds = image_data
    # Initialize counters
    correct_predictions = 0

    for i in range(10):
        img_path = Path(image_paths[i])
        with Image.open(img_path) as image:
            # Predict the breed
            predicted_breed = predict_single(dbc_model, dog_breeds, image)
            assert (
                predicted_breed in dog_breeds
            ), f"Predicted breed '{predicted_breed}' is not in dog breeds list."
            if predicted_breed == true_classes[i]:
                correct_predictions += 1

    accuracy = correct_predictions / 10
    assert accuracy >= 0.8, f"Expected accuracy > 0.8, but got {accuracy:.4f}"


@mock.patch("mates.modeling.predict.Image")
def test_predict_single_image_conversion(mock_image, dbc_model, image_data):
    """
    Test the `predict_single` function to ensure it converts images to RGB
    format when they are not already in that format.

    Parameters:
    -----------
    dbc_model : keras.Model
        The pre-trained model for dog breed classification.

    Assertions:
    -----------
    - The Image.conver function is called to convert the greyscale image to RGB
    - The predicted breed is valid and in the expected list.
    """
    _, _, dog_breeds = image_data

    # Create a mock grayscale image
    mock_image_instance = mock_image.new("L", (IMG_SIZE, IMG_SIZE))  # Mock grayscale image
    rgb_image = Image.open("tests/test_images/pod.jpg")
    mock_image_instance.convert.return_value = rgb_image

    # Call predict_single with the mocked image
    predicted_breed = predict_single(dbc_model, dog_breeds, mock_image_instance)

    # Assert that the convert method was called once with 'RGB'
    mock_image_instance.convert.assert_called_once_with("RGB")

    # Validate the prediction
    assert (
        predicted_breed in dog_breeds
    ), f"Predicted breed '{predicted_breed}' is not in dog breeds list."
