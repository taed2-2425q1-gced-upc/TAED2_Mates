""" Module to test function get_label_image from features """
from pathlib import Path
import tensorflow as tf
import pytest
from mates.features.features import get_label_image

# Mock data
mock_processed_image = tf.zeros((128, 128, 3), dtype=tf.float32)  # Mock processed image tensor

@pytest.fixture
def patch_tf_functions_get_label(mocker):
    """
    Mock function process_image
    """
    # Patch the process_image function used in get_label_image
    mocker.patch('mates.features.process_image', return_value=mock_processed_image)

def test_get_label_image(patch_tf_functions_get_label):
    """"
    Test for get_label_image
    """
    # pylint: disable=unused-argument
    # Given
    img_path = Path('path/to/mock_image.jpg')  # Mock path
    label = 'bulldog'  # Example label

    # When
    processed_image, returned_label = get_label_image(img_path, label)

    # Then
    assert isinstance(processed_image, tf.Tensor), \
        "The processed image should be a TensorFlow tensor"
    assert processed_image.shape == (128, 128, 3), \
        "The processed image should have shape (128, 128, 3)"
    assert returned_label == label, "The returned label should match the input label"
