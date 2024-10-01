import os
import yaml
import typer
import tf_keras
import pickle as pk
import pandas as pd
import tensorflow as tf
from pathlib import Path
import tensorflow_hub as hub


from mates.config import IMG_SIZE, PROCESSED_DATA_DIR, RAW_DATA_DIR, MODELS_DIR


app = typer.Typer()


@app.command()
def load_processed_data(
    batch_size: int
):
    """
    """
    with open(PROCESSED_DATA_DIR / 'output_shape.pkl', 'rb') as f:
        output_shape = pk.load(f)
    with open(PROCESSED_DATA_DIR / 'X_train.pkl', 'rb') as f:
        X_train = pk.load(f)
    with open(PROCESSED_DATA_DIR / 'y_train.pkl', 'rb') as f:
        y_train = pk.load(f)
    with open(PROCESSED_DATA_DIR / 'X_valid.pkl', 'rb') as f:
        X_valid = pk.load(f)
    with open(PROCESSED_DATA_DIR / 'y_valid.pkl', 'rb') as f:
        y_valid = pk.load(f)

    train_data = create_batches(batch_size, X_train, y_train)
    valid_data = create_batches(batch_size, X_valid, y_valid, valid_data=True)

    return train_data, valid_data, output_shape


@app.command()
def load_params(
    stage: str
) -> dict:
    """
    """
    params_path = Path("params.yaml")

    with open(params_path, "r") as params_file:
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
    """"""
    with open(MODELS_DIR / f"{model_name}.pkl", "rb") as f:
        model = pk.load(f)
    return model
    


@app.command()
def create_model(
    input_shape: list,
    output_shape: int,
    model_url: str,
    loss_function = tf_keras.losses.CategoricalCrossentropy(),
    optimizer = tf_keras.optimizers.Adam(),
    activation_function: str = 'softmax',
    metrics: list = ['accuracy'],
):
    """
    """
    model = tf_keras.Sequential()
    model.add(hub.KerasLayer(model_url, input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(tf_keras.layers.Dense(output_shape, activation=activation_function))
    
    model.compile(
        loss=loss_function, 
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
