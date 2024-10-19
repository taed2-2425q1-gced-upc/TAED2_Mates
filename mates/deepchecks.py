"""
Module for performing deep learning data checks and preparing train-validation split.

This module provides functionalities to create image batches, generate vision datasets, 
and run model checks using the DeepChecks library. It uses a MobileNetV2 model to validate 
the train-validation split and outputs the results as HTML reports. The module leverages 
Typer for building a command-line interface (CLI).

Commands available:
- custom_generator: Custom data generator that creates batches of images and labels.
- create_vision_data: Creates a VisionData object for use with DeepChecks.
- run_checks: Runs the full DeepChecks suite on the train and validation datasets.
- train_val_split_check: Splits the dataset into training and validation, then performs 
  checks to validate the split.

Dependencies:
- numpy: For handling numerical operations on image data.
- typer: For building the command-line interface (CLI).
- loguru: For logging information during the execution.
- PIL: For image processing (loading and resizing images).
- sklearn: For splitting the dataset into training and validation.
- deepchecks: For running model checks and validating the train-validation split.

Additional module imports:
- RAW_DATA_DIR, REPORTS_DIR, IMG_SIZE from mates.config: Constants for raw data path, 
  reports directory, and image size.
- read_data, load_params, load_model from mates.features: Utility functions for reading data, 
  loading model parameters, and loading the pre-trained model.
"""

import numpy as np
import typer
from loguru import logger
from PIL import Image
from deepchecks.vision.suites import full_suite
from deepchecks.vision import VisionData, BatchOutputFormat
from sklearn.model_selection import train_test_split

from mates.config import RAW_DATA_DIR, REPORTS_DIR, IMG_SIZE
from mates.features import read_data, load_params, load_model

app = typer.Typer()

@app.command()
def custom_generator(X, y, batch_size: int, target_size: tuple = (IMG_SIZE, IMG_SIZE)):
    """
    Custom generator to yield batches of images and labels.

    Parameters
    ----------
    X : list
        List of image file paths.
    y : np.array
        Array of one-hot encoded labels corresponding to the images.
    batch_size : int
        Size of each batch to be generated.
    target_size : tuple, optional
        Target size for resizing images, default is (IMG_SIZE, IMG_SIZE).

    Yields
    ------
    BatchOutputFormat
        A batch of images and labels in the format required by DeepChecks.
    """
    n = len(X)
    for i in range(0, n, batch_size):
        images_batch = []
        labels_batch = []

        for j in range(i, min(i + batch_size, n)):
            img = Image.open(X[j]).resize(target_size)
            # Ensure pixel values are between 0 and 255
            img = np.array(img, dtype=np.uint8)
            label = np.where(y[j] == 1)[0][0]
            images_batch.append(img)
            labels_batch.append(label)

        images_batch = np.array(images_batch)
        labels_batch = np.array(labels_batch)

        yield BatchOutputFormat(images=images_batch, labels=labels_batch)


@app.command()
def create_vision_data(generator, task_type: str):
    """
    Create a VisionData object using the custom generator.

    Parameters
    ----------
    generator : generator
        Generator that yields batches of images and labels.
    task_type : str
        Type of task (e.g., 'classification') for DeepChecks.

    Returns
    -------
    VisionData
        A VisionData object for performing checks.
    """
    return VisionData(generator, task_type=task_type, reshuffle_data=False)


@app.command()
def run_checks(train_ds, val_ds, reports_dir: str):
    """
    Run DeepChecks full suite on training and validation datasets.

    Parameters
    ----------
    train_ds : VisionData
        The training dataset formatted for DeepChecks.
    val_ds : VisionData
        The validation dataset formatted for DeepChecks.
    reports_dir : str
        Directory where the report will be saved.

    Returns
    -------
    None
    """
    reports_dir.mkdir(parents=True, exist_ok=True)

    model = load_model("mobilenet_exp_batch_62")

    suite = full_suite()
    result = suite.run(train_ds, val_ds, model)
    result.save_as_html(str(reports_dir / "train_val_split_check.html"))


@app.command()
def train_val_split_check():
    """
    Perform a train-validation split and run checks to validate the split.

    This function loads the dataset, splits it into training and validation sets, 
    then performs DeepChecks validation on the split and saves the results.

    Returns
    -------
    None
    """
    logger.info("Running checks on train-validation split...")
    params = load_params("prepare")

    X, y, _ = read_data(dir_path=RAW_DATA_DIR, train_data=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                      test_size=params["split_size"],
                                                      random_state=params["seed"])

    train_ds = create_vision_data(
        custom_generator(X_train, y_train, params["batch_size"]),
        'classification')
    val_ds = create_vision_data(
        custom_generator(X_val, y_val, params["batch_size"]),
        'classification')
    run_checks(train_ds, val_ds, REPORTS_DIR / "deepchecks")

    logger.success("Checks completed.")


if __name__ == "__main__":
    train_val_split_check()
