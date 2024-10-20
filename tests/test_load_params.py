""" Module to test the load_params function from the features module. """

from unittest import mock
import pytest
import yaml
from mates.features.utils import load_params  # Adjust the import based on your actual module structure

@pytest.fixture
def patch_file_system_load_params(mocker):
    """
    Fixture to mock file system interactions for the load_params function.

    This fixture sets up the necessary mocks for file reading and YAML loading,
    allowing tests to simulate different parameter configurations without
    relying on the actual file system.

    Returns
    -------
    mock_open : unittest.mock.mock_open
        A mock object representing the open function to simulate file reading.
    """
    # Patch yaml.safe_load to return predefined parameters for different stages
    mocker.patch('mates.features.utils.yaml.safe_load',
                 return_value={
                     'prepare': {'is_train': True},
                     'train': {'save_model': True, 'epochs': 15},
                     'predict': {'model_name': "mobilenet_exp_batch_32", 'batch_size': 32}
                 })
    mock_open = mocker.patch('builtins.open', new_callable=mock.mock_open)
    mocker.patch('mates.features.utils.Path', return_value="parameters.yaml")
    return mock_open

def test_load_params_train(patch_file_system_load_params):
    """
    Test the load_params function for the 'train' stage.

    This test verifies that the correct parameters are loaded from the
    configuration file for the training stage.

    Parameters
    ----------
    patch_file_system_load_params : mock.Mock
        The fixture that mocks file system interactions.
    """
    params = load_params("train")

    patch_file_system_load_params.assert_called_once_with("parameters.yaml", "r", encoding='utf-8')
    assert params == {'save_model': True, 'epochs': 15}

def test_load_params_prepare(patch_file_system_load_params):
    """
    Test the load_params function for the 'prepare' stage.

    This test checks that the parameters for the preparation stage are
    correctly retrieved from the configuration.

    Parameters
    ----------
    patch_file_system_load_params : mock.Mock
        The fixture that mocks file system interactions.
    """
    params = load_params("prepare")

    patch_file_system_load_params.assert_called_once_with("parameters.yaml", "r", encoding='utf-8')
    assert params == {'is_train': True}

def test_load_params_predict(patch_file_system_load_params):
    """
    Test the load_params function for the 'predict' stage.

    This test ensures that the parameters related to prediction are
    accurately loaded from the configuration file.

    Parameters
    ----------
    patch_file_system_load_params : mock.Mock
        The fixture that mocks file system interactions.
    """
    params = load_params("predict")

    patch_file_system_load_params.assert_called_once_with("parameters.yaml", "r", encoding='utf-8')
    assert params == {'model_name': "mobilenet_exp_batch_32", 'batch_size': 32}

def test_load_params_yaml_error(mocker):
    """
    Test the load_params function for handling YAML loading errors.

    This test simulates a scenario where the YAML configuration file
    cannot be loaded due to a parsing error, ensuring that the function
    handles this gracefully.

    Parameters
    ----------
    mocker : pytest_mock.plugin.MockerFixture
        The mocker fixture for creating mocks in tests.
    """
    # Patch the open function to simulate file reading and raise a YAMLError
    mocker.patch('mates.features.utils.Path', return_value="parameters.yaml")
    mock_open = mocker.patch('builtins.open', new_callable=mock.mock_open)
    mocker.patch('mates.features.utils.yaml.safe_load', side_effect=yaml.YAMLError("Error loading YAML"))

    # Call the function with an expected stage
    params = load_params("train")

    # Verify that the open function was called correctly
    mock_open.assert_called_once_with("parameters.yaml", "r", encoding='utf-8')

    # Assert that the function returns an empty dict (based on your implementation)
    assert params == {}
