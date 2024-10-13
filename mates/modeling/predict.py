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
- OUTPUT_DATA_DIR from mates.config: Path to the directory where predictions are saved.
- create_batches, load_model, load_params, read_data from mates.features: Functions for
    data processing, model loading, and batching.
"""

import os

import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras
import typer
from PIL import Image

from mates.config import IMG_SIZE, OUTPUT_DATA_DIR, RAW_DATA_DIR
from mates.features import (
    create_batches,
    load_model,
    load_params,
    read_data,
    read_labels,
)

app = typer.Typer()


@app.command()
def predict_test(
):
    """
    Function to predict on test data. Loads the model and predicts on the test data.
    """
    params = load_params("predict")
    model = load_model(params["model_name"])
    x_test, _, _ = read_data(train_data=False)
    test_data = create_batches(params["batch_size"], x_test, test_data=True)

    y_pred = model.predict(test_data)

    _, encoding_labels = read_labels(RAW_DATA_DIR)
    y_pred = [encoding_labels[np.argmax(pred)] for pred in y_pred]

    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)

    y_pred_df = pd.DataFrame(y_pred, index=[x.stem for x in x_test], columns=["breed"])
    y_pred_df.to_csv(OUTPUT_DATA_DIR / "predictions_test.csv")


def predict_single(
    model: tf_keras.Model,
    encoding_labels: list,
    image: Image.Image,
):
    """Predict function for a single image."""

    # Preprocess image
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = tf_keras.preprocessing.image.img_to_array(image)
    image = tf.expand_dims(image, axis=0)

    # Make prediction
    prediction = model.predict(image)

    # Get predicted label
    predicted_label = encoding_labels[np.argmax(prediction[0])]
    return predicted_label


if __name__ == "__main__":
    app()
