""" Module to test the get_label_image function from the features module. """

from pathlib import Path

import pytest
import tensorflow as tf

from mates.features.features import get_label_image

# Mock data
mock_processed_image = tf.zeros((128, 128, 3), dtype=tf.float32)  # Mock processed image tensor


@pytest.fixture
def patch_tf_functions_get_label(mocker):
    """
    Fixture to mock the process_image function used in get_label_image.

    This fixture ensures that the tests do not rely on the actual implementation
    of process_image, allowing for isolated testing of the get_label_image function.

    Parameters
    ----------
    mocker : pytest_mock.plugin.MockerFixture
        The mocker fixture used to create mocks for TensorFlow components.
    """
    mocker.patch("mates.features.features.process_image", return_value=mock_processed_image)


def test_get_label_image(patch_tf_functions_get_label):
    """
    Test the get_label_image function.

    This test verifies that the get_label_image function correctly processes
    an image and returns the expected label.

    Parameters
    ----------
    patch_tf_functions_get_label : pytest.fixture
        The fixture that mocks the process_image function.
    """
    # Given
    img_path = Path("path/to/mock_image.jpg")  # Mock path
    label = "bulldog"  # Example label

    # When
    processed_image, returned_label = get_label_image(img_path, label)

    # Then
    assert isinstance(
        processed_image, tf.Tensor
    ), "The processed image should be a TensorFlow tensor"
    assert processed_image.shape == (
        128,
        128,
        3,
    ), "The processed image should have shape (128, 128, 3)"
    assert returned_label == label, "The returned label should match the input label"
