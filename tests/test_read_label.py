import pandas as pd
import pytest
from unittest import mock
from pathlib import Path
from mates.features import read_labels  # Replace with your actual module name

@pytest.fixture
def mock_labels_csv(tmp_path):
    """Create a mock CSV file with sample labels for testing."""
    data = {
        'breed': ['labrador', 'poodle', 'bulldog', 'labrador', 'poodle']
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / 'labels.csv'
    df.to_csv(csv_path, index=False)
    return tmp_path

def test_read_labels(mock_labels_csv):
    """Test the read_labels function."""
    labels, encoding_labels = read_labels(mock_labels_csv)

    # Check the contents of the labels DataFrame
    assert len(labels) == 5  # Check the number of rows
    assert set(labels['breed']) == {'labrador', 'poodle', 'bulldog'}  # Check unique breeds

    # Check encoding labels
    assert len(encoding_labels) == 3  # Should be 3 unique breeds
    assert set(encoding_labels) == {'bulldog', 'labrador', 'poodle'}  # Check the unique encodings

    # Check the order of encoding labels (if needed)
    expected_order = ['bulldog', 'labrador', 'poodle']
    assert list(encoding_labels) == expected_order
