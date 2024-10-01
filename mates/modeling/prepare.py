import os
import yaml
import typer
import pickle as pk
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split

from mates.features import load_params
from mates.config import IMG_SIZE, RAW_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


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
    X : list
        List of image paths.
    y : list
        List of labels. None if train_data is False.
    encoding_labels : list
        List of encoding labels. None if train_data is False.
    """
    
    data_type = 'train' if train_data else 'test'
    imgs = os.listdir(dir_path / f'{data_type}/')
    
    X = [dir_path / f'{data_type}/' / f for f in imgs]
    
    if train_data:
        labels = pd.read_csv(dir_path / 'labels.csv')
        y = pd.get_dummies(labels['breed']).to_numpy()
        encoding_labels = labels['breed'].unique()
    else:
        y = None
        encoding_labels = None
    
    return X, y, encoding_labels


@app.command()
def process_image(
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
    
    return process_image(img_path), label


@app.command()
def create_batches(
    batch_size: int,
    X: list,
    y: list = None,
    valid_data: bool = False,
    test_data: bool = False,
):
    """
    """
    if test_data:
        data = tf.data.Dataset.from_tensor_slices(tf.constant([str(x) for x in X]))
        data_batch = data.map(process_image).batch(batch_size)
    elif valid_data:
        data = tf.data.Dataset.from_tensor_slices((tf.constant([str(x) for x in X]), tf.constant(y)))
        data_batch = data.map(get_label_image).batch(batch_size)
    else:
        data = tf.data.Dataset.from_tensor_slices((tf.constant([str(x) for x in X]), tf.constant(y)))
        data = data.shuffle(buffer_size=len(X))
        data_batch = data.map(get_label_image).batch(batch_size)
    
    return data_batch


@app.command()
def process_data(
): 
    """
    """
    params = load_params("prepare")

    X, y, encoding_labels = read_data(train_data=params["is_train"])
    
    if params["is_train"]:
        output_shape = len(encoding_labels)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=params["split_size"], random_state=params["seed"])

        if params["save_processed"]:
            with open(PROCESSED_DATA_DIR / 'output_shape.pkl', 'wb') as f:
                pk.dump(output_shape, f)

            with open(PROCESSED_DATA_DIR / 'X_train.pkl', 'wb') as f:
                pk.dump(X_train, f)
            with open(PROCESSED_DATA_DIR / 'y_train.pkl', 'wb') as f:
                pk.dump(y_train, f)
            with open(PROCESSED_DATA_DIR / 'X_valid.pkl', 'wb') as f:
                pk.dump(X_val, f)
            with open(PROCESSED_DATA_DIR / 'y_valid.pkl', 'wb') as f:
                pk.dump(y_val, f)


if __name__ == "__main__":
    process_data()