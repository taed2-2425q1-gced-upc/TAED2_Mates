""" Module to test the load_model function from the features module. """

from unittest import mock

import tf_keras

from mates.features.features import MODELS_DIR, load_model


@mock.patch(
    "mates.features.features.tf_keras.models.load_model",
    return_value=mock.Mock(spec=tf_keras.Model),
)
def test_load_model(mock_load_model):
    """
    Test the load_model function.

    This test verifies that the load_model function correctly loads a
    specified model from the filesystem and returns it.

    Parameters
    ----------
    mock_load_model : unittest.mock.Mock
        The mock object replacing the tf.keras.models.load_model function
        to isolate the test from the actual model loading process.
    """
    model_name = "mobilenet_exp_batch_32"

    # Call the function to test
    model = load_model(model_name)

    # Assert that the returned model is the mocked model
    assert (
        model is mock_load_model.return_value
    ), "The returned model should be the mocked model instance"

    # Assert that the load_model function was called with the correct arguments
    mock_load_model.assert_called_once_with(
        MODELS_DIR / f"{model_name}.h5", custom_objects={"KerasLayer": mock.ANY}
    )
