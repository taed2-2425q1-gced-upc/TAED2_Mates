""" Module to test the `predict_test` function from the `mates.modeling.predict` module. """

import os

import pandas as pd
import pytest

from mates.config import OUTPUT_DATA_DIR, RAW_DATA_DIR
from mates.features.features import read_labels
from mates.modeling.predict import predict_test

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

@pytest.mark.skipif(
    IN_GITHUB_ACTIONS,
    reason="Test doesn't work in Github Actions. \
                    Enough with test_dbc_model",
)
def test_predict():
    """
    Test for the `predict_test` function.

    This test verifies that the `predict_test` function correctly predicts the dog breed
    for all images located in the `raw/tests` directory. It checks that the output CSV
    file is generated with the expected number of rows and valid breed names.

    Assertions:
    - The number of predictions matches the number of images processed.
    - The output CSV contains exactly two columns.
    - All predicted breeds are from the predefined list of dog breeds.
    """

    # Call the predict_test function to generate predictions
    predict_test()

    # Path to the generated CSV file containing predictions
    output_csv_path = os.path.join(OUTPUT_DATA_DIR, "predictions_test.csv")

    # Read the generated predictions into a DataFrame
    generated_df = pd.read_csv(output_csv_path)

    num_images = 10357

    # Assertions to validate predictions
    assert (
        len(generated_df) == num_images
    ), f"Expected {num_images} predictions, but got {len(generated_df)}."

    assert generated_df.shape[1] == 2, f"Expected 2 columns, but got {generated_df.shape[1]}."

    _, dog_breeds = read_labels(RAW_DATA_DIR)
    # Define the list of valid dog breeds

    # Assert that all breeds in the predictions are in the predefined list
    assert all(
        generated_df["breed"].isin(dog_breeds)
    ), "One or more breeds in the predictions are not in the predefined dog breeds list."
