"""
Data Preparation Module

This module provides functionality to prepare data for machine learning tasks such as 
training and validation. It processes raw image data, performs train-validation splits, 
and saves the processed datasets for further use. 

Key Features:
- Configurable data preparation: The parameters, such as train/test mode, validation split size, 
  and random seed, are loaded from a YAML configuration file.
- Train-validation split: For training datasets, the module splits the data into training and 
  validation sets, ensuring proper data separation for model training.
- Data serialization: Processed datasets, including training, validation, and output shape, 
  are saved as pickle files for easy reuse without having to reprocess raw data each time.

Workflow:
1. Load preparation parameters from a configuration file.
2. Read the raw image data and corresponding labels from the dataset.
3. If preparing training data:
   - Split the dataset into training and validation sets based on the configuration.
   - Save the split datasets and encoding labels.
4. Log the progress and results of the data preparation.

The module uses the following dependencies:
- `pickle`: For saving and loading processed datasets.
- `typer`: For providing a command-line interface (CLI) to execute the data preparation process.
- `loguru`: For logging information, warnings, and errors during data processing.
- `sklearn.model_selection.train_test_split`: For splitting data into training and validation sets.

External module imports:
- `PROCESSED_DATA_DIR` from mates.config: The directory where processed data is saved.
- `load_params`, `read_data` from mates.features: Functions for loading configuration parameters
    and reading raw data.
"""

import pickle as pk

import typer
from loguru import logger
from sklearn.model_selection import train_test_split

from mates.config import PROCESSED_DATA_DIR
from mates.features.features import read_data
from mates.features.utils import load_params

app = typer.Typer()


@app.command()
def process_data():
    """
    Process and prepare data for training or validation. This function reads raw data,
    performs a train-validation split if required, and saves the processed data.

    The function reads parameters from a configuration file, such as whether the data
    is for training or testing, the size of the validation split, and whether to save
    the processed data.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    # Load preparation parameters from the config
    params = load_params("prepare")

    logger.info("Processing data...")

    # Read raw data (for training or testing based on params)
    x, y, encoding_labels = read_data(train_data=params["is_train"])

    # If training data is being processed
    if params["is_train"]:
        output_shape = len(encoding_labels)

        # Perform train-validation split
        x_train, x_val, y_train, y_val = train_test_split(
            x, y, test_size=params["split_size"], random_state=params["seed"]
        )

        # Save processed data if specified in the params
        if params["save_processed"]:
            with open(PROCESSED_DATA_DIR / "output_shape.pkl", "wb") as f:
                pk.dump(output_shape, f)
            with open(PROCESSED_DATA_DIR / "X_train.pkl", "wb") as f:
                pk.dump(x_train, f)
            with open(PROCESSED_DATA_DIR / "y_train.pkl", "wb") as f:
                pk.dump(y_train, f)
            with open(PROCESSED_DATA_DIR / "X_valid.pkl", "wb") as f:
                pk.dump(x_val, f)
            with open(PROCESSED_DATA_DIR / "y_valid.pkl", "wb") as f:
                pk.dump(y_val, f)

    logger.success("Data processing complete.")


if __name__ == "__main__":
    process_data()
