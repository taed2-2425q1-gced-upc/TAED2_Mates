""" Module to test function load_params from features """
from unittest import mock
import pytest
from mates.features import load_params  # Adjust the import based on your actual module structure

@pytest.fixture
def patch_file_system_load_params(mocker):
    """
    Fixture to mock functions
    """
    # Patch os.listdir and pandas.read_csv
    mocker.patch('mates.features.yaml.safe_load',
                 return_value={'prepare':{ 'is_train':True},
                                'train':{  'save_model': True, 'epochs': 15},
                                'predict':{'model_name': "mobilenet", 'batch_size': 32}})
    mock_open = mocker.patch('builtins.open', new_callable=mock.mock_open)
    mocker.patch('mates.features.Path', return_value="parameters.yaml")
    return mock_open

def test_load_params_train(patch_file_system_load_params):
    """
    Test function load_params
    """
    params = load_params("train")

    patch_file_system_load_params.assert_called_with("parameters.yaml", "r", encoding='utf-8')
    assert params == {'save_model': True, 'epochs': 15}

def test_load_params_prepare(patch_file_system_load_params):
    """
    Test function load_params
    """
    params = load_params("prepare")

    patch_file_system_load_params.assert_called_with("parameters.yaml", "r", encoding='utf-8')
    assert params == {'is_train':True}

def test_load_params_predict(patch_file_system_load_params):
    """
    Test function load_params
    """
    params = load_params("predict")

    patch_file_system_load_params.assert_called_with("parameters.yaml", "r", encoding='utf-8')
    assert params == {'model_name': "mobilenet", 'batch_size': 32}
