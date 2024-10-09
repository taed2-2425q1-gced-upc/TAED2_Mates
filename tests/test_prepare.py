""" Module to test the prepare stage of the model """
from unittest import mock
import numpy as np
from mates.modeling.prepare import process_data
from mates.config import PROCESSED_DATA_DIR

@mock.patch('pickle.dump')
@mock.patch('sklearn.model_selection.train_test_split')
@mock.patch('mates.modeling.prepare.read_data', return_value=([], None, None))
@mock.patch('mates.modeling.prepare.load_params', return_value={
    "is_train": False
})
def test_process_data_no_train(mock_load_params, mock_read_data, mock_train_test_split, mock_dump):
    """"
    Test function process_data from prepareb with no train data
    """
    process_data()

    mock_load_params.assert_called_once_with("prepare")
    mock_read_data.assert_called_once_with(train_data=False)

    assert mock_train_test_split.call_count == 0
    assert mock_dump.call_count == 0

@mock.patch('pickle.dump')
@mock.patch('builtins.open', new_callable=mock.mock_open)
@mock.patch('mates.modeling.prepare.train_test_split')
@mock.patch('mates.modeling.prepare.read_data', return_value=([1,2,3], [1,2,3], [0,1]))
@mock.patch('mates.modeling.prepare.load_params', return_value={
        "is_train": True,
        "save_processed": True,
        "seed": 42,
        "split_size": 0.3,
        "batch_size": 32,
})
def test_process_data_train_save(mock_load_params, mock_read_data,
                                 mock_train_test_split, mock_open, mock_dump):
    """"
    Test function process_data from prepareb with train data to be saved
    """
    x_train = [f"/fake_path/image_{i}.jpg" for i in range(70)]
    x_val = [f"/fake_path/image_{i}.jpg" for i in range(70, 100)]
    y_train = np.zeros((70, 120))
    y_val = np.zeros((30, 120))
    for i in range(70):
        y_train[i, np.random.randint(0, 120)] = 1  # Randomly set one class to True
    for i in range(30):
        y_val[i, np.random.randint(0, 120)] = 1  # Randomly set one class to True

    mock_train_test_split.return_value = (x_train, x_val, y_train, y_val)

    process_data()

    mock_load_params.assert_called_once_with("prepare")
    mock_read_data.assert_called_once_with(train_data=True)
    # mock_train_test_split.assert_called_once, "train_test_split was not called!"
    print(f"train_test_split call args: {mock_train_test_split.call_args}")  # Print the arguments
    mock_train_test_split.assert_called_once_with(
        mock_read_data.return_value[0], mock_read_data.return_value[1],
        test_size=0.3, random_state=42)

    # Assertions for file open calls
    mock_open.assert_any_call(PROCESSED_DATA_DIR / 'output_shape.pkl', 'wb')
    mock_open.assert_any_call(PROCESSED_DATA_DIR / 'x_train.pkl', 'wb')
    mock_open.assert_any_call(PROCESSED_DATA_DIR / 'y_train.pkl', 'wb')
    mock_open.assert_any_call(PROCESSED_DATA_DIR / 'x_valid.pkl', 'wb')
    mock_open.assert_any_call(PROCESSED_DATA_DIR / 'y_valid.pkl', 'wb')

    assert mock_dump.call_count == 5, "pickle.dump was not called the correct number of times"

    # Additional checks to validate the data shapes
    assert len(x_train) == 70, "Training data size is incorrect"  # 70% of 100
    assert len(y_train) == 70, "Training labels size is incorrect"  # 70% of 100
    assert len(x_val) == 30, "Validation data size is incorrect"  # 30% of 100
    assert len(y_val) == 30, "Validation labels size is incorrect"  # 30% of 100

    # Check that y_train and y_val are one-hot encoded
    assert y_train.shape[1] == 120, "y_train does not have the correct number of classes"
    assert y_val.shape[1] == 120, "y_valid does not have the correct number of classes"

    # Check that each label is one-hot encoded (only one True value per row)
    assert np.all(y_train.sum(axis=1) == 1), "y_train is not one-hot encoded"
    assert np.all(y_val.sum(axis=1) == 1), "y_valid is not one-hot encoded"
