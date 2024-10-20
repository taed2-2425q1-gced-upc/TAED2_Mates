""" Module to test function load_model from features """
from unittest import mock
import tf_keras
from mates.features.features import load_model, MODELS_DIR

@mock.patch('mates.features.tf_keras.models.load_model',
            return_value=mock.Mock(spec=tf_keras.Model))
def test_load_model(mock_load_model):
    """"
    Test function load_model
    """
    model_name = "mobilenet_exp_batch_32"

    model = load_model(model_name)

    assert model is mock_load_model.return_value
    mock_load_model.assert_called_once_with(MODELS_DIR / f"{model_name}.h5",
                                             custom_objects={'KerasLayer': mock.ANY})
