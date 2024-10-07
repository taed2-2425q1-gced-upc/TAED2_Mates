"""
Module for loading, processing, and managing machine learning data and models.

This module provides a set of commands to load processed data, parameters,
and models, as well as to create and process images for training machine
learning models. It utilizes TensorFlow and its Keras API to build
neural networks with pre-trained models from TensorFlow Hub.

Commands available:
- load_processed_data: Load training and validation data from pickle files and create batches.
- load_params: Load parameters for a specified stage from a YAML configuration file.
- load_model: Load a pre-trained model from the models directory.
- create_model: Build a new model using a specified pre-trained model from TensorFlow Hub.
- read_data: Load image paths and labels from a specified directory.
- process_image: Preprocess an image for input into the model.
- get_label_image: Retrieve the label and processed image for a given image path.
- create_batches: Generate batches of data for training or validation.

Dependencies:
- TensorFlow
- TensorFlow Hub
- Typer
- Pandas
- YAML
"""

import os
import pickle as pk
from pathlib import Path
import yaml
import typer
import tf_keras
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub


from mates.config import IMG_SIZE, PROCESSED_DATA_DIR, RAW_DATA_DIR, MODELS_DIR


app = typer.Typer()


@app.command()
def load_processed_data(
    batch_size: int
):
    """
    Function to load processed data. Loads processed data and creates batches.

    Parameters
    ----------
    batch_size : int
        Batch size for the data.

    Returns
    -------
    train_data : tf.data.Dataset
        Training data.
    valid_data : tf.data.Dataset
        Validation data.
    """
    with open(PROCESSED_DATA_DIR / 'output_shape.pkl', 'rb') as f:
        output_shape = pk.load(f)
    with open(PROCESSED_DATA_DIR / 'x_train.pkl', 'rb') as f:
        x_train = pk.load(f)
    with open(PROCESSED_DATA_DIR / 'y_train.pkl', 'rb') as f:
        y_train = pk.load(f)
    with open(PROCESSED_DATA_DIR / 'x_valid.pkl', 'rb') as f:
        x_valid = pk.load(f)
    with open(PROCESSED_DATA_DIR / 'y_valid.pkl', 'rb') as f:
        y_valid = pk.load(f)

    train_data = create_batches(batch_size, x_train, y_train)
    valid_data = create_batches(batch_size, x_valid, y_valid, valid_data=True)

    return train_data, valid_data, output_shape


@app.command()
def load_params(
    stage: str
) -> dict:
    """
    Load parameters from the params.yaml file.

    Parameters
    ----------
    stage : str
        Stage of the pipeline.

    Returns
    -------
    params : dict
        Parameters for the specified stage.
    """
    params_path = Path("params.yaml")

    with open(params_path, "r", encoding='utf-8') as params_file:
        try:
            params = yaml.safe_load(params_file)
            params = params[stage]
        except yaml.YAMLError as exc:
            print(exc)

    return params


@app.command()
def load_model(
    model_name: str
):
    """
    Load a model from the MODELS_DIR directory.
    
    Parameters
    ----------
    model_name : str
        Name of the model to load.
        
    Returns
    -------
        model : tf.keras.Model
    """
    model = tf_keras.models.load_model(MODELS_DIR / f"{model_name}.h5",
                                       custom_objects={'KerasLayer': hub.KerasLayer})
    return model


@app.command()
def create_model(
    input_shape: list,
    output_shape: int,
    model_url: str,
    optimizer: str,
    metrics: list,
):
    """
    Create a model using a pre-trained model from TensorFlow Hub.

    Parameters
    ----------
    input_shape : list
        Input shape of the model.
    output_shape : int
        Output shape of the model.
    model_url : str
        URL of the pre-trained model.
    optimizer : str
        Optimizer to use.
    metrics : list
        List of metrics to use.
    loss_function : tf.keras.losses
        Loss function to use.
    activation_function : str
        Activation function to use.

    Returns
    -------
    model : tf.keras.Model
        Compiled model.
    """
    model = tf_keras.Sequential()
    model.add(hub.KerasLayer(model_url, input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(tf_keras.layers.Dense(output_shape, activation='softmax'))

    if optimizer == 'adam':
        optimizer = tf_keras.optimizers.Adam()
    elif optimizer == 'sgd':
        optimizer = tf_keras.optimizers.SGD()
    elif optimizer == 'rmsprop':
        optimizer = tf_keras.optimizers.RMSprop()
    elif optimizer == 'adamw':
        optimizer = tf_keras.optimizers.AdamW()

    model.compile(
        loss=tf_keras.losses.CategoricalCrossentropy(),
        optimizer=optimizer,
        metrics=metrics
    )

    model.build(input_shape)

    return model


@app.command()
def read_data(
    dir_path: Path = RAW_DATA_DIR,
    train_data: bool = True,
):
    """
    Load training or test data.
    Returns image paths, labels (if train_data), and encoding labels (if train_data).

    Parameters
    ----------
    dir_path : Path
        Path to the data directory.
    train_data : bool
        Whether to load training or test data.

    Returns
    -------
    x : list
        List of image paths.
    y : list
        List of labels. None if train_data is False.
    encoding_labels : list
        List of encoding labels. None if train_data is False.
    """

    data_type = 'train' if train_data else 'test'
    imgs = os.listdir(dir_path / f'{data_type}/')

    x = [dir_path / f'{data_type}/' / f for f in imgs]

    if train_data:
        labels = pd.read_csv(dir_path / 'labels.csv')
        y = pd.get_dummies(labels['breed']).to_numpy()
        encoding_labels = labels['breed'].unique()
    else:
        y = None
        encoding_labels = None

    return x, y, encoding_labels


@app.command()
def process_image(
    img_path: Path,
    img_size: int = IMG_SIZE,
):
    """
    Process an image. Read the image, decode it, convert it to float32, and resize it.

    Parameters
    ----------
    img_path : Path
        Path to the image.
    img_size : int
        Size of the image.

    Returns
    -------
    img : tf.Tensor
        Processed image.
    """
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, size=[img_size, img_size])

    return img


@app.command()
def get_label_image(img_path: Path, label: int):
    """
    Get the label and image for a given image path.

    Parameters
    ----------
    img_path : Path
        Path to the image.
    label : int
        Label for the image.

    Returns
    -------
    img : tf.Tensor
        Processed image.
    """
    return process_image(img_path), label


@app.command()
def create_batches(
    batch_size: int,
    x: list,
    y: list = None,
    valid_data: bool = False,
    test_data: bool = False,
):
    """
    Create batches of data.

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
        data = tf.data.Dataset.from_tensor_slices(
            tf.constant([str(x) for x in x]))
        data_batch = data.map(process_image).batch(batch_size)
    elif valid_data:
        data = tf.data.Dataset.from_tensor_slices(
            (tf.constant([str(x) for x in x]), tf.constant(y)))
        data_batch = data.map(get_label_image).batch(batch_size)
    else:
        data = tf.data.Dataset.from_tensor_slices(
            (tf.constant([str(x) for x in x]), tf.constant(y)))
        data = data.shuffle(buffer_size=len(x))
        data_batch = data.map(get_label_image).batch(batch_size)

    return data_batch
