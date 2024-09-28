import os
import typer
import pandas as pd
import tensorflow as tf
from pathlib import Path

from mates.config import IMG_SIZE, RAW_DATA_DIR, BATCH_SIZE

app = typer.Typer()


@app.command()
def read_data(
    train_data: bool = True,
):
    """
    Load training or test data.
    Returns image paths, labels (if train_data), and encoding labels (if train_data).

    Parameters
    ----------
    train_data : bool
        Whether to load training or test data.

    Returns
    -------
    X : list
        List of image paths.
    y : list
        List of labels. None if train_data is False.
    encoding_labels : list
        List of encoding labels. None if train_data is False.
    """
    
    data_type = 'train' if train_data else 'test'
    imgs = os.listdir(RAW_DATA_DIR / f'{data_type}/')
    
    X = [RAW_DATA_DIR / f'{data_type}/' / f for f in imgs]
    
    if train_data:
        labels = pd.read_csv(RAW_DATA_DIR / 'labels.csv')
        y = pd.get_dummies(labels['breed']).to_numpy()
        encoding_labels = labels['breed'].unique()
    else:
        y = None
        encoding_labels = None
    
    return X, y, encoding_labels


@app.command()
def prepare_image(
    img_path: Path,
    img_size: int = IMG_SIZE,
):
    """
    """

    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, size=[img_size, img_size])
    
    return img


@app.command()
def get_label_image(img_path, label):
    """
    """
    
    return prepare_image(img_path), label


@app.command()
def create_batches(
    X: list,
    y: list = None,
    batch_size: int = BATCH_SIZE,
    valid_data: bool = False,
    test_data: bool = False,
):
    """
    """
    
    if test_data:
        data = tf.data.Dataset.from_tensor_slices(tf.constant([str(x) for x in X]))
        data_batch = data.map(prepare_image).batch(batch_size)
    elif valid_data:
        data = tf.data.Dataset.from_tensor_slices((tf.constant([str(x) for x in X]), tf.constant(y)))
        data_batch = data.map(get_label_image).batch(batch_size)
    else:
        data = tf.data.Dataset.from_tensor_slices((tf.constant([str(x) for x in X]), tf.constant(y)))
        data = data.shuffle(buffer_size=len(X))
        data_batch = data.map(get_label_image).batch(batch_size)
    
    return data_batch