""" Module to test function create_batches from features """
import tensorflow as tf
import pytest
from mates.features import create_batches  # Adjust the import based on your actual module structure

# Mock functions
def mock_process_image(image):
    """
    Return image for testing
    """
    return image  # Return the image directly for testing

def mock_get_label_image(image, label):
    """
    Return iamge and image label
    """
    return image, label  # Return the image and label directly for testing

# Use the pytest fixture for patching
@pytest.fixture
def patch_tf_functions_create_batches(mocker):
    """
    Fixture to mock functinos process_image and get_label_image
    """
    # Adjust the patch path to reflect the correct import path
    mocker.patch('mates.features.process_image', side_effect=mock_process_image)
    mocker.patch('mates.features.get_label_image', side_effect=mock_get_label_image)

def test_create_batches_with_validation_data(patch_tf_functions_create_batches):
    """"
    Test for create_batches function for validation data
    """
    # pylint: disable=unused-argument
    # Given
    batch_size = 10
    x = ['image1.jpg', 'image2.jpg', 'image3.jpg']
    y = [0, 1, 1]  # Dummy labels for testing
    valid_data = True

    # Cal fcuntion to create batches
    data_batch = create_batches(batch_size=batch_size, x=x, y=y, valid_data=valid_data)

    # Assert
    assert isinstance(data_batch, tf.data.Dataset)
    # Since we have 3 items and batch size is 10
    assert data_batch.cardinality().numpy() == 1
    for batch in data_batch:
        assert batch[0].numpy().tolist() == [b'image1.jpg', b'image2.jpg', b'image3.jpg']
        assert batch[1].numpy().tolist() == [0, 1, 1]

def test_create_batches_with_test_data(patch_tf_functions_create_batches):
    """"
    Test for create_batches function for test data
    """
    # pylint: disable=unused-argument
    # Given
    batch_size = 10
    x = ['image1.jpg', 'image2.jpg', 'image3.jpg']
    valid_data = False
    test_data = True

    # When
    data_batch = create_batches(batch_size=batch_size, x=x,
                                valid_data=valid_data, test_data=test_data)

    # Then
    assert isinstance(data_batch, tf.data.Dataset)
    assert data_batch.cardinality().numpy() == 1  # All images in one batch since batch size is 10
    for batch in data_batch:
        assert batch.numpy().tolist() == [b'image1.jpg', b'image2.jpg', b'image3.jpg']

def test_create_batches_with_training_data(patch_tf_functions_create_batches):
    """
    Test for create_batches function for training data
    """
    # pylint: disable=unused-argument
    # Given
    batch_size = 2
    x = ['image1.jpg', 'image2.jpg', 'image3.jpg']
    y = [0, 1, 1]  # Dummy labels for testing
    valid_data = False
    test_data = False

    # When
    data_batch = create_batches(batch_size=batch_size, x=x, y=y,
                                valid_data=valid_data, test_data=test_data)

    # Then
    assert isinstance(data_batch, tf.data.Dataset)
    # batch size is 2, so we expect 2 batches (because there are 3 elements)
    assert data_batch.cardinality().numpy() == 2
    batches = list(data_batch.as_numpy_iterator())
    assert len(batches) == 2  # Check we have 2 batches
    assert all(len(batch) == batch_size for batch in batches)  # Check batch size
    assert len(batches[-1][0]) == 1  # Last batch should contain remaining item
