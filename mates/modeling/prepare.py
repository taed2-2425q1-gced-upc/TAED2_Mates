"""Module providing a function to prepare the data."""

import pickle as pk

import typer
from loguru import logger
from sklearn.model_selection import train_test_split

from mates.config import PROCESSED_DATA_DIR
from mates.features import load_params, read_data

app = typer.Typer()

@app.command()
def process_data(
):
    """
    Function to process data. Performs data processing and saves processed data.
    """
    params = load_params("prepare")

    logger.info("Processing data...")
    x, y, encoding_labels = read_data(train_data=params["is_train"])

    if params["is_train"]:
        output_shape = len(encoding_labels)
        x_train, x_val, y_train, y_val = train_test_split(
            x, y, test_size=params["split_size"], random_state=params["seed"]
        )

        if params["save_processed"]:
            with open(PROCESSED_DATA_DIR / 'output_shape.pkl', 'wb') as f:
                pk.dump(output_shape, f)

            with open(PROCESSED_DATA_DIR / 'x_train.pkl', 'wb') as f:
                pk.dump(x_train, f)
            with open(PROCESSED_DATA_DIR / 'y_train.pkl', 'wb') as f:
                pk.dump(y_train, f)
            with open(PROCESSED_DATA_DIR / 'x_valid.pkl', 'wb') as f:
                pk.dump(x_val, f)
            with open(PROCESSED_DATA_DIR / 'y_valid.pkl', 'wb') as f:
                pk.dump(y_val, f)

    logger.success("Data processing complete.")

if __name__ == "__main__":
    process_data()
