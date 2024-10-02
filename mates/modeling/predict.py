
"""
Module to predict on test data. The function loads the model and predicts on the test data.
"""

import os
import pickle as pk

import typer

from mates.config import EXTERNAL_DATA_DIR
from mates.features import create_batches, load_model, load_params, read_data

app = typer.Typer()


@app.command()
def predict(
):
    """
    Function to predict on test data. Loads the model and predicts on the test data.
    """
    params = load_params("predict")
    model = load_model(params["model_name"])
    x_test, _, _ = read_data(train_data=False)
    test_data = create_batches(params["batch_size"], x_test, test_data=True)

    y_pred = model.predict(test_data)

    os.makedirs(EXTERNAL_DATA_DIR, exist_ok=True)

    with open(EXTERNAL_DATA_DIR / 'y_test.pkl', 'wb') as f:
        pk.dump(y_pred, f)


if __name__ == "__main__":
    app()
