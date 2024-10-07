"""
Module for generating predictions on test data using a pre-trained model.

This module provides a command-line interface (CLI) to load a pre-trained model, 
process test data, and generate predictions. The predictions are saved in a 
pickle file for further analysis or evaluation.

Commands available:
- predict: Loads the pre-trained model and test data, generates predictions, 
  and saves the predictions to an external directory.

Workflow:
1. Load prediction parameters from a YAML configuration file.
2. Load the pre-trained model from the models directory.
3. Read and preprocess the test dataset.
4. Create batches of test data for prediction.
5. Generate predictions using the model on the test data.
6. Save the predicted results as a pickle file in the external data directory.

Dependencies:
- os: For creating directories to save the predictions.
- pickle: For saving the predictions as a pickle file.
- typer: For building the command-line interface.

Additional module imports:
- EXTERNAL_DATA_DIR from mates.config: Path to the directory where predictions are saved.
- create_batches, load_model, load_params, read_data from mates.features: Functions for
    data processing, model loading, and batching.
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
