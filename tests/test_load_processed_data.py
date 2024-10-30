""" Module to test the load_processed_data function from features. """

from unittest import mock

import tensorflow as tf

from mates.features.features import PROCESSED_DATA_DIR, load_processed_data


@mock.patch("mates.features.features.create_batches")
@mock.patch("mates.features.features.pk.load")
@mock.patch("builtins.open", new_callable=mock.mock_open)
def test_load_processed_data(mock_open, mock_pk_load, mock_create_batches):
    """
    Test for load_processed_data.

    This test verifies that the load_processed_data function correctly loads
    processed data and prepares it for training and validation.
    """
    # Sample data to be returned by pk.load
    mock_output_shape = 2  # Example output shape
    mock_x_train = ["image1.jpg", "image2.jpg"]  # Example training data
    mock_y_train = [[1, 0], [0, 1]]  # Example training labels
    mock_x_valid = ["image3.jpg"]  # Example validation data
    mock_y_valid = [[0, 1]]  # Example validation labels

    # Configure mock for pk.load to return values in sequence
    mock_pk_load.side_effect = [
        mock_output_shape,  # First call returns output_shape
        mock_x_train,  # Second call returns x_train
        mock_y_train,  # Third call returns y_train
        mock_x_valid,  # Fourth call returns x_valid
        mock_y_valid,  # Fifth call returns y_valid
    ]

    # Mock create_batches to return dummy TensorFlow Datasets
    mock_train_data = tf.data.Dataset.from_tensor_slices(
        (tf.constant(mock_x_train), tf.constant(mock_y_train))
    )
    mock_valid_data = tf.data.Dataset.from_tensor_slices(
        (tf.constant(mock_x_valid), tf.constant(mock_y_valid))
    )
    mock_create_batches.side_effect = [mock_train_data, mock_valid_data]

    # Call the function under test
    batch_size = 1
    _, _, output_shape = load_processed_data(batch_size)

    # Assertions
    assert mock_pk_load.call_count == 5
    mock_open.assert_any_call(PROCESSED_DATA_DIR / "output_shape.pkl", "rb")
    mock_open.assert_any_call(PROCESSED_DATA_DIR / "X_train.pkl", "rb")
    mock_open.assert_any_call(PROCESSED_DATA_DIR / "y_train.pkl", "rb")
    mock_open.assert_any_call(PROCESSED_DATA_DIR / "X_valid.pkl", "rb")
    mock_open.assert_any_call(PROCESSED_DATA_DIR / "y_valid.pkl", "rb")

    assert output_shape == mock_output_shape
    # We don't test train_data and valid_data because create_batches has been tested.

    # Verify create_batches was called with the correct arguments
    mock_create_batches.assert_any_call(batch_size, mock_x_train, mock_y_train)
    mock_create_batches.assert_any_call(batch_size, mock_x_valid, mock_y_valid, valid_data=True)
