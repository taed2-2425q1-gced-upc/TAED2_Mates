""" Module to test the `read_labels` function from the modeling.features module. """

import pandas as pd
import pytest
from mates.features.features import read_labels  # Adjust to your actual module name

@pytest.fixture
def mock_labels_csv(tmp_path):
    """
    Create a mock CSV file with sample labels for testing.

    This fixture generates a temporary CSV file containing breed 
    labels and returns the path for use in tests.
    
    Returns:
        Path: Path to the mock CSV file.
    """
    data = {
        'breed': ['labrador', 'poodle', 'bulldog', 'labrador', 'poodle']
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / 'labels.csv'
    df.to_csv(csv_path, index=False)
    return tmp_path

def test_read_labels(mock_labels_csv):
    """
    Test the `read_labels` function.

    This test verifies that the `read_labels` function correctly 
    reads breed labels from the provided CSV file and returns 
    both the labels DataFrame and the unique encoding labels.
    """
    labels, encoding_labels = read_labels(mock_labels_csv)

    # Check the contents of the labels DataFrame
    assert len(labels) == 5, "Expected 5 rows in the labels DataFrame."
    assert set(labels['breed']) == {'labrador', 'poodle', 'bulldog'}, \
        "The unique breeds in the DataFrame do not match the expected set."

    # Check encoding labels
    assert len(encoding_labels) == 3, "Expected 3 unique breeds for encoding."
    assert set(encoding_labels) == {'bulldog', 'labrador', 'poodle'}, \
        "The unique encoding labels do not match the expected set."

    # Check the order of encoding labels (if needed)
    expected_order = ['bulldog', 'labrador', 'poodle']
    assert list(encoding_labels) == expected_order, \
        "The encoding labels do not match the expected order."
