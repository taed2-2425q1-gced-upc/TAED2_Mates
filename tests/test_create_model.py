""" Module to test the create_model function from the features module. """

from unittest import mock

import pytest

from mates.features.features import IMG_SIZE, create_model


@pytest.fixture
def mock_tf_components(mocker):
    """
    Fixture to mock TensorFlow components used in the model creation process.

    This fixture ensures that the tests do not depend on the actual TensorFlow
    implementation, allowing for isolated testing of the create_model function.

    Parameters
    ----------
    mocker : pytest_mock.plugin.MockerFixture
        The mocker fixture used to create mocks for TensorFlow components.

    Returns
    -------
    dict
        A dictionary containing mocked TensorFlow components.
    """
    mock_sequential = mocker.patch(
        "mates.features.features.tf_keras.Sequential", new_callable=mock.Mock
    )
    mock_hub_keras_layer = mocker.patch(
        "mates.features.features.hub.KerasLayer", new_callable=mock.Mock
    )
    mock_add = mocker.patch(
        "mates.features.features.tf_keras.Sequential.add", new_callable=mock.Mock
    )
    mock_dense = mocker.patch(
        "mates.features.features.tf_keras.layers.Dense", new_callable=mock.Mock
    )
    mock_adam = mocker.patch(
        "mates.features.features.tf_keras.optimizers.Adam", new_callable=mock.Mock
    )
    mock_sgd = mocker.patch(
        "mates.features.features.tf_keras.optimizers.SGD", new_callable=mock.Mock
    )
    mock_rmsprop = mocker.patch(
        "mates.features.features.tf_keras.optimizers.RMSprop", new_callable=mock.Mock
    )
    mock_adamw = mocker.patch(
        "mates.features.features.tf_keras.optimizers.AdamW", new_callable=mock.Mock
    )
    mock_categorical_crossentropy = mocker.patch(
        "mates.features.features.tf_keras.losses.CategoricalCrossentropy", new_callable=mock.Mock
    )

    # Return all mocks
    return {
        "mock_sequential": mock_sequential,
        "mock_hub_keras_layer": mock_hub_keras_layer,
        "mock_add": mock_add,
        "mock_dense": mock_dense,
        "mock_adam": mock_adam,
        "mock_sgd": mock_sgd,
        "mock_rmsprop": mock_rmsprop,
        "mock_adamw": mock_adamw,
        "mock_categorical_crossentropy": mock_categorical_crossentropy,
    }


def test_create_model_rmsprop(mock_tf_components):
    """
    Test the create_model function using the RMSprop optimizer.

    This test verifies that the model is created correctly with the specified
    input shape, output shape, model URL, optimizer, and metrics.

    Parameters
    ----------
    mock_tf_components : dict
        The fixture containing mocked TensorFlow components for the test.
    """
    # Unpack the mocked components from the fixture
    components = mock_tf_components
    mock_sequential = components["mock_sequential"]
    mock_hub_keras_layer = components["mock_hub_keras_layer"]
    mock_dense = components["mock_dense"]
    mock_rmsprop = components["mock_rmsprop"]
    mock_categorical_crossentropy = components["mock_categorical_crossentropy"]

    # Define input parameters for the model
    input_shape = [IMG_SIZE, IMG_SIZE, 3]
    output_shape = 10  # Example output shape
    model_url = (
        "https://kaggle.com/models/google/mobilenet-v2/TensorFlow2/035-128-classification/2"
    )
    optimizer = "rmsprop"  # Example optimizer
    metrics = ["accuracy"]  # Example metrics

    # Call the function to test
    model = create_model(input_shape, output_shape, model_url, optimizer, metrics)

    # Check if the model is a Sequential instance
    assert model is mock_sequential.return_value

    # Assert that KerasLayer was called with the correct model URL
    mock_hub_keras_layer.assert_called_once_with(model_url, input_shape=(IMG_SIZE, IMG_SIZE, 3))

    # Assert that Dense layer was added with the correct output shape and activation
    mock_dense.assert_called_once_with(output_shape, activation="softmax")
    model.add.assert_called_with(mock_dense.return_value)

    # Assert that the optimizer was created and used in the compile method
    mock_sequential.return_value.compile.assert_called_once_with(
        loss=mock_categorical_crossentropy(), optimizer=mock_rmsprop(), metrics=metrics
    )

    # Assert that the model was built with the correct input shape
    model.build.assert_called_once_with(input_shape)


def test_create_model_adam(mock_tf_components):
    """
    Test the create_model function using the Adam optimizer.

    This test checks that the model is created correctly with the specified
    parameters, including the Adam optimizer.

    Parameters
    ----------
    mock_tf_components : dict
        The fixture containing mocked TensorFlow components for the test.
    """
    # Unpack the mocked components from the fixture
    components = mock_tf_components
    mock_sequential = components["mock_sequential"]
    mock_hub_keras_layer = components["mock_hub_keras_layer"]
    mock_dense = components["mock_dense"]
    mock_adam = components["mock_adam"]
    mock_categorical_crossentropy = components["mock_categorical_crossentropy"]

    # Define input parameters for the model
    input_shape = [IMG_SIZE, IMG_SIZE, 3]
    output_shape = 10  # Example output shape
    model_url = (
        "https://kaggle.com/models/google/mobilenet-v2/TensorFlow2/035-128-classification/2"
    )
    optimizer = "adam"  # Example optimizer
    metrics = ["accuracy"]  # Example metrics

    # Call the function to test
    model = create_model(input_shape, output_shape, model_url, optimizer, metrics)

    # Check if the model is a Sequential instance
    assert model is mock_sequential.return_value

    # Assert that KerasLayer was called with the correct model URL
    mock_hub_keras_layer.assert_called_once_with(model_url, input_shape=(IMG_SIZE, IMG_SIZE, 3))

    # Assert that Dense layer was added with the correct output shape and activation
    mock_dense.assert_called_once_with(output_shape, activation="softmax")
    model.add.assert_called_with(mock_dense.return_value)

    # Assert that the optimizer was created and used in the compile method
    mock_sequential.return_value.compile.assert_called_once_with(
        loss=mock_categorical_crossentropy(), optimizer=mock_adam(), metrics=metrics
    )

    # Assert that the model was built with the correct input shape
    model.build.assert_called_once_with(input_shape)


def test_create_model_adamw(mock_tf_components):
    """
    Test the create_model function using the AdamW optimizer.

    This test verifies that the model is created correctly with the specified
    parameters, including the AdamW optimizer.

    Parameters
    ----------
    mock_tf_components : dict
        The fixture containing mocked TensorFlow components for the test.
    """
    # Unpack the mocked components from the fixture
    components = mock_tf_components
    mock_sequential = components["mock_sequential"]
    mock_hub_keras_layer = components["mock_hub_keras_layer"]
    mock_dense = components["mock_dense"]
    mock_adamw = components["mock_adamw"]
    mock_categorical_crossentropy = components["mock_categorical_crossentropy"]

    # Define input parameters for the model
    input_shape = [IMG_SIZE, IMG_SIZE, 3]
    output_shape = 10  # Example output shape
    model_url = (
        "https://kaggle.com/models/google/mobilenet-v2/TensorFlow2/035-128-classification/2"
    )
    optimizer = "adamw"  # Example optimizer
    metrics = ["accuracy"]  # Example metrics

    # Call the function to test
    model = create_model(input_shape, output_shape, model_url, optimizer, metrics)

    # Check if the model is a Sequential instance
    assert model is mock_sequential.return_value

    # Assert that KerasLayer was called with the correct model URL
    mock_hub_keras_layer.assert_called_once_with(model_url, input_shape=(IMG_SIZE, IMG_SIZE, 3))

    # Assert that Dense layer was added with the correct output shape and activation
    mock_dense.assert_called_once_with(output_shape, activation="softmax")
    model.add.assert_called_with(mock_dense.return_value)

    # Assert that the optimizer was created and used in the compile method
    mock_sequential.return_value.compile.assert_called_once_with(
        loss=mock_categorical_crossentropy(), optimizer=mock_adamw(), metrics=metrics
    )

    # Assert that the model was built with the correct input shape
    model.build.assert_called_once_with(input_shape)


def test_create_model_sgd(mock_tf_components):
    """
    Test the create_model function using the SGD optimizer.

    This test checks that the model is created correctly with the specified
    parameters, including the SGD optimizer.

    Parameters
    ----------
    mock_tf_components : dict
        The fixture containing mocked TensorFlow components for the test.
    """
    # Unpack the mocked components from the fixture
    components = mock_tf_components
    mock_sequential = components["mock_sequential"]
    mock_hub_keras_layer = components["mock_hub_keras_layer"]
    mock_dense = components["mock_dense"]
    mock_sgd = components["mock_sgd"]
    mock_categorical_crossentropy = components["mock_categorical_crossentropy"]

    # Define input parameters for the model
    input_shape = [IMG_SIZE, IMG_SIZE, 3]
    output_shape = 10  # Example output shape
    model_url = (
        "https://kaggle.com/models/google/mobilenet-v2/TensorFlow2/035-128-classification/2"
    )
    optimizer = "sgd"  # Example optimizer
    metrics = ["accuracy"]  # Example metrics

    # Call the function to test
    model = create_model(input_shape, output_shape, model_url, optimizer, metrics)

    # Check if the model is a Sequential instance
    assert model is mock_sequential.return_value

    # Assert that KerasLayer was called with the correct model URL
    mock_hub_keras_layer.assert_called_once_with(model_url, input_shape=(IMG_SIZE, IMG_SIZE, 3))

    # Assert that Dense layer was added with the correct output shape and activation
    mock_dense.assert_called_once_with(output_shape, activation="softmax")
    model.add.assert_called_with(mock_dense.return_value)

    # Assert that the optimizer was created and used in the compile method
    mock_sequential.return_value.compile.assert_called_once_with(
        loss=mock_categorical_crossentropy(), optimizer=mock_sgd(), metrics=metrics
    )

    # Assert that the model was built with the correct input shape
    model.build.assert_called_once_with(input_shape)
