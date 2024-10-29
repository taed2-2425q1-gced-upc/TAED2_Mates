"""
Module for handling data processing, model creation, and batch generation.

This module provides various utility functions for loading and preprocessing data, 
loading models, and generating batches of data for training, validation, and testing 
purposes. It also includes functionalities for creating and compiling models using 
pre-trained TensorFlow Hub modules.

Commands available:
- load_processed_data: Loads pre-processed training and validation data, then creates 
  batches for each.
- load_model: Loads a pre-trained model from the specified directory.
- process_image: Pre-processes images by reading, decoding, resizing, and converting them.
- read_labels: Reads labels and encodings from a CSV file.
- read_data: Reads image paths and labels for training or testing data.
- create_batches: Generates batches of images and labels for efficient training and evaluation.
- create_model: Creates a new model with a pre-trained backbone, adds dense layers, and compiles it.

Dependencies:
- os: For managing file paths and directories.
- pathlib: For handling paths in a platform-independent way.
- pickle: For loading serialized datasets.
- pandas: For handling label data in DataFrame format.
- tensorflow/keras: For model creation, image processing, and prediction.
- TensorFlow Hub: For loading pre-trained models.
- typer: For building the command-line interface (CLI).

Additional module imports:
- IMG_SIZE, MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR from mates.config: Constants for 
  model paths, data paths, and image size used across the functions.
"""

import os
import pickle as pk
from pathlib import Path

import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tf_keras
import typer

from mates.config import IMG_SIZE, MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def load_processed_data(batch_size: int):
    """
    Function to load processed data and create batches for training and validation.

    Parameters
    ----------
    batch_size : int
        Batch size for the data.

    Returns
    -------
    train_data : tf.data.Dataset
        Batched training data.
    valid_data : tf.data.Dataset
        Batched validation data.
    output_shape : tuple
        Output shape of the dataset for model configuration.
    """
    with open(PROCESSED_DATA_DIR / "output_shape.pkl", "rb") as f:
        output_shape = pk.load(f)
    with open(PROCESSED_DATA_DIR / "x_train.pkl", "rb") as f:
        x_train = pk.load(f)
    with open(PROCESSED_DATA_DIR / "y_train.pkl", "rb") as f:
        y_train = pk.load(f)
    with open(PROCESSED_DATA_DIR / "x_valid.pkl", "rb") as f:
        x_valid = pk.load(f)
    with open(PROCESSED_DATA_DIR / "y_valid.pkl", "rb") as f:
        y_valid = pk.load(f)

    train_data = create_batches(batch_size, x_train, y_train)
    valid_data = create_batches(batch_size, x_valid, y_valid, valid_data=True)

    return train_data, valid_data, output_shape


@app.command()
def load_model(model_name: str):
    """
    Load a pre-trained model from the MODELS_DIR directory.

    Parameters
    ----------
    model_name : str
        Name of the model to load.

    Returns
    -------
    model : tf.keras.Model
        The loaded model.
    """
    model = tf_keras.models.load_model(
        MODELS_DIR / f"{model_name}.h5", custom_objects={"KerasLayer": hub.KerasLayer}
    )
    return model


@app.command()
def process_image(
    img_path: Path,
    img_size: int = IMG_SIZE,
):
    """
    Pre-process an image by resizing and converting it to a tensor.

    Parameters
    ----------
    img_path : Path
        Path to the image file.
    img_size : int
        Size to resize the image to.

    Returns
    -------
    img : tf.Tensor
        Preprocessed image as a TensorFlow tensor.
    """
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, size=[img_size, img_size])

    return img


@app.command()
def get_label_image(img_path: Path, label: int):
    """
    Fetch the label and image for a given path.

    Parameters
    ----------
    img_path : Path
        Path to the image.
    label : int
        Label corresponding to the image.

    Returns
    -------
    img : tf.Tensor
        Processed image tensor.
    """
    return process_image(img_path), label


@app.command()
def read_labels(
    dir_path: Path,
):
    """
    Read labels from a CSV file and generate encoding labels.

    Parameters
    ----------
    dir_path : Path
        Path to the directory containing the labels CSV file.

    Returns
    -------
    labels : pd.DataFrame
        DataFrame of labels.
    encoding_labels : list
        List of unique label encodings (breeds).
    """
    labels = pd.read_csv(dir_path / "labels.csv")
    encoding_labels = pd.get_dummies(labels["breed"]).columns

    return labels, encoding_labels


@app.command()
def read_data(
    dir_path: Path = RAW_DATA_DIR,
    train_data: bool = True,
):
    """
    Load image paths and labels for either training or testing data.

    Parameters
    ----------
    dir_path : Path
        Path to the data directory.
    train_data : bool
        Whether to load training data or test data.

    Returns
    -------
    x : list
        List of image paths.
    y : list
        List of labels (if train_data is True).
    encoding_labels : list
        List of encoding labels (if train_data is True).
    """
    data_type = "train" if train_data else "test"
    imgs = os.listdir(dir_path / f"{data_type}/")

    if train_data:
        labels, encoding_labels = read_labels(dir_path)
        y = pd.get_dummies(labels["breed"]).to_numpy()
        x = [dir_path / f"{data_type}/" / f"{id}.jpg" for id in labels["id"]]
        encoding_labels = labels["breed"].unique()
    else:
        imgs = os.listdir(dir_path / f"{data_type}/")
        x = [dir_path / f"{data_type}/" / f for f in imgs]
        y = None
        encoding_labels = None

    return x, y, encoding_labels


@app.command()
def create_batches(
    batch_size: int,
    x: list,
    y: list = None,
    valid_data: bool = False,
    test_data: bool = False,
):
    """
    Create batches of data for training, validation, or testing.

    Parameters
    ----------
    batch_size : int
        Batch size.
    x : list
        List of image paths.
    y : list
        List of labels.
    valid_data : bool
        Whether the data is validation data.
    test_data : bool
        Whether the data is test data.

    Returns
    -------
    data_batch : tf.data.Dataset
        Batched data.
    """
    if test_data:
        data = tf.data.Dataset.from_tensor_slices(tf.constant([str(x) for x in x]))
        data_batch = data.map(process_image).batch(batch_size)
    elif valid_data:
        data = tf.data.Dataset.from_tensor_slices(
            (tf.constant([str(x) for x in x]), tf.constant(y))
        )
        data_batch = data.map(get_label_image).batch(batch_size)
    else:
        data = tf.data.Dataset.from_tensor_slices(
            (tf.constant([str(x) for x in x]), tf.constant(y))
        )
        data = data.shuffle(buffer_size=len(x))
        data_batch = data.map(get_label_image).batch(batch_size)

    return data_batch


@app.command()
def create_model(
    input_shape: list,
    output_shape: int,
    model_url: str,
    optimizer: str,
    metrics: list,
):
    """
    Create and compile a model using a pre-trained backbone from TensorFlow Hub.

    Parameters
    ----------
    input_shape : list
        Input shape of the model (e.g., [None, 224, 224, 3]).
    output_shape : int
        Output shape (number of classes).
    model_url : str
        URL of the pre-trained model to load from TensorFlow Hub.
    optimizer : str
        Optimizer to use for training (e.g., 'adam', 'sgd').
    metrics : list
        List of metrics to use for model evaluation.

    Returns
    -------
    model : tf.keras.Model
        Compiled Keras model.
    """
    model = tf_keras.Sequential()
    model.add(hub.KerasLayer(model_url, input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(tf_keras.layers.Dense(output_shape, activation="softmax"))

    if optimizer == "adam":
        optimizer = tf_keras.optimizers.Adam()
    elif optimizer == "sgd":
        optimizer = tf_keras.optimizers.SGD()
    elif optimizer == "rmsprop":
        optimizer = tf_keras.optimizers.RMSprop()
    elif optimizer == "adamw":
        optimizer = tf_keras.optimizers.AdamW()

    model.compile(
        loss=tf_keras.losses.CategoricalCrossentropy(), optimizer=optimizer, metrics=metrics
    )

    model.build(input_shape)

    return model
