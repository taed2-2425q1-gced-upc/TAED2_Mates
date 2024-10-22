""" Module to test the create_batches function from the features module. """

import pytest
import tensorflow as tf

from mates.features.features import create_batches


# Mock functions for testing
def mock_process_image(image):
    """
    Mock function to simulate image processing.

    Parameters
    ----------
    image : str
        The path to the image to be processed.

    Returns
    -------
    str
        The processed image path for testing purposes.
    """
    return image


def mock_get_label_image(image, label):
    """
    Mock function to simulate retrieving image and label.

    Parameters
    ----------
    image : str
        The path to the image.
    label : int
        The corresponding label for the image.

    Returns
    -------
    tuple
        A tuple containing the image and its label for testing purposes.
    """
    return image, label


# Use the pytest fixture for patching
@pytest.fixture
def patch_tf_functions_create_batches(mocker):
    """
    Fixture to mock the process_image and get_label_image functions.

    This fixture allows the tests to run without relying on the actual
    implementations of image processing or label retrieval.

    Parameters
    ----------
    mocker : pytest_mock.plugin.MockerFixture
        The mocker fixture used to create mocks.
    """
    # Adjust the patch path to reflect the correct import path
    mocker.patch("mates.features.features.process_image", side_effect=mock_process_image)
    mocker.patch("mates.features.features.get_label_image", side_effect=mock_get_label_image)


def test_create_batches_with_validation_data(patch_tf_functions_create_batches):
    """
    Test the create_batches function for validation data.

    This test checks whether the function correctly creates batches
    for validation data and returns a TensorFlow Dataset.

    Parameters
    ----------
    patch_tf_functions_create_batches : mock.Mock
        The fixture that mocks TensorFlow functions.
    """
    # Given
    batch_size = 10
    x = ["image1.jpg", "image2.jpg", "image3.jpg"]
    y = [0, 1, 1]  # Dummy labels for testing
    valid_data = True

    # When
    data_batch = create_batches(batch_size=batch_size, x=x, y=y, valid_data=valid_data)

    # Then
    assert isinstance(data_batch, tf.data.Dataset)
    # Since we have 3 items and batch size is 10
    assert data_batch.cardinality().numpy() == 1
    for batch in data_batch:
        assert batch[0].numpy().tolist() == [b"image1.jpg", b"image2.jpg", b"image3.jpg"]
        assert batch[1].numpy().tolist() == [0, 1, 1]


def test_create_batches_with_test_data(patch_tf_functions_create_batches):
    """
    Test the create_batches function for test data.

    This test verifies that the function properly creates batches for
    test data and returns a TensorFlow Dataset.

    Parameters
    ----------
    patch_tf_functions_create_batches : mock.Mock
        The fixture that mocks TensorFlow functions.
    """
    # Given
    batch_size = 10
    x = ["image1.jpg", "image2.jpg", "image3.jpg"]
    valid_data = False
    test_data = True

    # When
    data_batch = create_batches(
        batch_size=batch_size, x=x, valid_data=valid_data, test_data=test_data
    )

    # Then
    assert isinstance(data_batch, tf.data.Dataset)
    assert data_batch.cardinality().numpy() == 1  # All images in one batch since batch size is 10
    for batch in data_batch:
        assert batch.numpy().tolist() == [b"image1.jpg", b"image2.jpg", b"image3.jpg"]


def test_create_batches_with_training_data(patch_tf_functions_create_batches):
    """
    Test the create_batches function for training data.

    This test checks that the function creates appropriate batches
    for training data, validating both batch sizes and contents.

    Parameters
    ----------
    patch_tf_functions_create_batches : mock.Mock
        The fixture that mocks TensorFlow functions.
    """
    # Given
    batch_size = 2
    x = ["image1.jpg", "image2.jpg", "image3.jpg"]
    y = [0, 1, 1]  # Dummy labels for testing
    valid_data = False
    test_data = False

    # When
    data_batch = create_batches(
        batch_size=batch_size, x=x, y=y, valid_data=valid_data, test_data=test_data
    )

    # Then
    assert isinstance(data_batch, tf.data.Dataset)
    # batch size is 2, so we expect 2 batches (because there are 3 elements)
    assert data_batch.cardinality().numpy() == 2
    batches = list(data_batch.as_numpy_iterator())
    assert len(batches) == 2  # Check we have 2 batches
    assert all(len(batch) == batch_size for batch in batches)  # Check batch size
    assert len(batches[-1][0]) == 1  # Last batch should contain remaining item
