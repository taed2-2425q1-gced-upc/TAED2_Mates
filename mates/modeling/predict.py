"""
Module for generating predictions on test data using a pre-trained model.

This module provides a command-line interface (CLI) to load a pre-trained model,
process test data, and generate predictions. The predicted results are saved as a
CSV file for further analysis or evaluation.

Commands available:
- predict_test: Loads the pre-trained model and test data, generates predictions,
  and saves the predictions to a specified directory.

Workflow:
1. Load prediction parameters from a YAML configuration file.
2. Load the pre-trained model from the models directory.
3. Read and preprocess the test dataset.
4. Create batches of test data for efficient prediction.
5. Generate predictions using the model on the test data.
6. Map the predicted output to corresponding labels (dog breeds).
7. Save the predicted results as a CSV file in the external data directory.

Dependencies:
- os: For handling directory creation to save predictions.
- pandas: For saving predictions as a DataFrame and exporting to CSV.
- numpy: For efficient handling of predictions and array manipulations.
- typer: For building the command-line interface (CLI).
- tensorflow/keras: For loading pre-trained models and making predictions.
- PIL (Pillow): For image processing when predicting single images.

Additional module imports:
- IMG_SIZE, OUTPUT_DATA_DIR, RAW_DATA_DIR from mates.config: Constants for image size
  and paths to data directories.
- create_batches, load_model, load_params, read_data, read_labels from mates.features:
  Utility functions for data processing, model loading, parameter fetching, and label encoding.
"""

import os

import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras
import typer
from PIL import Image

from mates.config import IMG_SIZE, OUTPUT_DATA_DIR, RAW_DATA_DIR
from mates.features.features import create_batches, load_model, read_data, read_labels
from mates.features.utils import load_params

app = typer.Typer()


@app.command()
def predict_test():
    """
    Function to predict on test data. Loads the model, processes the test data, and
    generates predictions that are saved as a CSV file.

    Returns
    -------
    None
    """
    # Load prediction parameters from the configuration file
    params = load_params("predict")

    # Load the pre-trained model specified in the parameters
    model = load_model(params["model_name"])

    # Read test data (without labels)
    x_test, _, _ = read_data(train_data=False)

    # Create batches for efficient processing
    test_data = create_batches(params["batch_size"], x_test, test_data=True)

    # Generate predictions
    y_pred = model.predict(test_data)

    # Load label encodings to map predictions to breed names
    _, encoding_labels = read_labels(RAW_DATA_DIR)
    y_pred = [encoding_labels[np.argmax(pred)] for pred in y_pred]

    # Ensure the output directory exists
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)

    # Save predictions as a CSV file
    y_pred_df = pd.DataFrame(y_pred, index=[x.stem for x in x_test], columns=["breed"])
    y_pred_df.to_csv(OUTPUT_DATA_DIR / "predictions_test.csv")


def predict_single(
    model: tf_keras.Model,
    encoding_labels: list,
    image: Image.Image,
) -> str:
    """
    Predict function for a single image. Loads and preprocesses the image,
    then generates a prediction for the corresponding dog breed.

    Parameters
    ----------
    model : tf_keras.Model
        Pre-trained model for prediction.
    encoding_labels : list
        List of encoding labels for mapping predictions to dog breeds.
    image : Image.Image
        Image to predict the breed for.

    Returns
    -------
    predicted_label : str
        Predicted dog breed.
    """
    # Preprocess image: resize, convert to array, normalize and add batch dimension
    if image.mode != "RGB":
        image = image.convert("RGB")

    img = image.resize((IMG_SIZE, IMG_SIZE))
    img = tf_keras.preprocessing.image.img_to_array(img)
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)
    img = tf.image.convert_image_dtype(img, tf.float32)

    prediction = model.predict(img)
    # Get the predicted label (dog breed)
    predicted_label = encoding_labels[np.argmax(prediction[0])]
    return predicted_label


if __name__ == "__main__":
    app()
