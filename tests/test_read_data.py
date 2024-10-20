""" Module to test the `read_data` function from the features module. """

from pathlib import Path
import pytest
import pandas as pd
from mates.features.features import read_data

# Mock training data
mock_train_data = pd.DataFrame({
    'id': ['img1', 'img2', 'img3'],
    'breed': ['affenpinscher', 'afghan_hound', 'affenpinscher']
})

# Mock image list for testing
mock_image_list = ['img1.jpg', 'img2.jpg', 'img3.jpg']

@pytest.fixture
def mock_file_operations(mocker):
    """
    Fixture to mock file operations for the tests.

    This fixture replaces the actual file operations with mock 
    returns to allow testing the read_data function in isolation.
    """
    # Mock pd.read_csv to return the mock training data
    mocker.patch('pandas.read_csv', return_value=mock_train_data)

    # Mock os.listdir to return a list of image filenames
    mocker.patch('os.listdir', return_value=mock_image_list)

def test_read_data_with_training_data(mock_file_operations):
    """
    Test the `read_data` function when loading training data.

    This test verifies that the correct image paths and one-hot 
    encoded labels are returned when training data is specified.
    """
    mock_dir_path = Path('mock/raw_data')  # Mock directory path

    # When
    x, y, encoding_labels = read_data(dir_path=mock_dir_path, train_data=True)

    # Expected values
    expected_x = [mock_dir_path / 'train' / f'{id}.jpg' for id in mock_train_data['id']]
    expected_y = pd.get_dummies(mock_train_data['breed']).to_numpy()
    expected_encoding_labels = ['affenpinscher', 'afghan_hound']

    # Then
    assert len(x) == 3, "Should return 3 image paths."
    assert x == expected_x, "Image paths should match the mocked list."
    
    assert y is not None, "Should return labels when train_data is True."
    assert (y == expected_y).all(), "Labels should be one-hot encoded correctly."
    assert y.shape == (3, 2), "Labels should have shape (3, 2) due to two unique breeds."
    
    assert y.tolist() == [
        [1, 0],  # affenpinscher
        [0, 1],  # afghan_hound
        [1, 0],  # affenpinscher
    ], "Labels should match the expected one-hot encoding."
    
    assert encoding_labels is not None, "Encoding labels should not be None when train_data is True."
    assert encoding_labels.tolist() == expected_encoding_labels, "Encoding labels should match unique breeds."

def test_read_data_with_test_data(mock_file_operations):
    """
    Test the `read_data` function when loading test data.

    This test verifies that only image paths are returned and that 
    no labels or encoding labels are provided when loading test data.
    """
    mock_dir_path = Path('/mock/path')

    # When
    x, y, encoding_labels = read_data(dir_path=mock_dir_path, train_data=False)

    # Expected values
    expected_x = [mock_dir_path / 'test' / f for f in mock_image_list]

    # Then
    assert x == expected_x, "Image paths should match the expected test image filenames."
    assert y is None, "y should be None when loading test data."
    assert encoding_labels is None, "encoding_labels should be None when loading test data."
