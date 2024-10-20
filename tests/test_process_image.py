""" Module to test the `process_image` function from the features module. """

from pathlib import Path
import tensorflow as tf
import pytest
from mates.features.features import process_image

# Mock data
mock_image_data = tf.constant([[[0, 0, 0]]], dtype=tf.uint8)  # Mock image data for testing
mock_resized_image = tf.zeros((128, 128, 3), dtype=tf.float32)  # Mock resized image tensor

@pytest.fixture
def patch_tf_functions(mocker):
    """
    Mock TensorFlow functions used in the `process_image` function.

    This fixture patches the necessary TensorFlow functions to prevent
    actual file I/O and image processing during testing, allowing for
    controlled behavior and predictable outputs.
    """
    # Mock TensorFlow functions
    mocker.patch('tensorflow.io.read_file', return_value=mock_image_data)
    mocker.patch('tensorflow.image.decode_jpeg', return_value=mock_image_data)
    mocker.patch('tensorflow.image.convert_image_dtype', return_value=mock_image_data)
    mocker.patch('tensorflow.image.resize', return_value=mock_resized_image)

def test_process_image(patch_tf_functions):
    """
    Test the `process_image` function.

    This test verifies that the `process_image` function correctly
    processes an image path, returning a resized TensorFlow tensor
    as expected.
    """
    # Given
    img_path = Path('path/to/image.jpg')  # Example image path
    img_size = 128  # Example target image size

    # When
    processed_image = process_image(img_path, img_size)

    # Then
    assert isinstance(processed_image, tf.Tensor), \
        "The processed image should be a TensorFlow tensor."
    assert processed_image.shape == mock_resized_image.shape, \
        "The processed image should match the expected shape."
    assert tf.reduce_all(processed_image == mock_resized_image), \
        "The processed image should match the mock resized image."
