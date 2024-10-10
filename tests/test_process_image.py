""" Module to test function process_image from features """
from pathlib import Path
import tensorflow as tf
import pytest
from mates.features import process_image  # Adjust the import based on your actual module structure

# Mock data
mock_image_data = tf.constant([[[0, 0, 0]]], dtype=tf.uint8)  # Mock image data for testing
mock_resized_image = tf.zeros((128, 128, 3), dtype=tf.float32)  # Mock resized image tensor

@pytest.fixture
def patch_tf_functions(mocker):
    """"
    Mock tensorflow functions
    """
    # Correctly patch TensorFlow functions used in process_image
    mocker.patch('tensorflow.io.read_file', return_value=mock_image_data)
    mocker.patch('tensorflow.image.decode_jpeg', return_value=mock_image_data)
    mocker.patch('tensorflow.image.convert_image_dtype', return_value=mock_image_data)
    mocker.patch('tensorflow.image.resize', return_value=mock_resized_image)

def test_process_image(patch_tf_functions):
    """
    Test function process_image
    """
    # pylint: disable=unused-argument
    # Given
    img_path = Path('path/to/image.jpg')  # Example image path
    img_size = 128  # Example image size

    # When
    processed_image = process_image(img_path, img_size)

    # Then
    assert isinstance(processed_image, tf.Tensor),\
          "The processed image should be a TensorFlow tensor"
    assert processed_image.shape == mock_resized_image.shape,\
          "The processed image should match the expected shape"
    assert tf.reduce_all(processed_image == mock_resized_image),\
          "The processed image should match the mock resized image"
