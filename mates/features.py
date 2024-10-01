import yaml
import typer
import tf_keras
import pickle as pk
from pathlib import Path
import tensorflow_hub as hub

from mates.modeling.prepare import create_batches
from mates.config import IMG_SIZE, PROCESSED_DATA_DIR

app = typer.Typer()


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


def load_processed_data(
):
    """
    """
    with open(PROCESSED_DATA_DIR / 'output_shape.pkl', 'wb') as f:
        output_shape = pk.load(f)
    with open(PROCESSED_DATA_DIR / 'X_train.pkl', 'wb') as f:
        X_train = pk.load(f)
    with open(PROCESSED_DATA_DIR / 'y_train.pkl', 'wb') as f:
        y_train = pk.load(f)
    with open(PROCESSED_DATA_DIR / 'X_valid.pkl', 'wb') as f:
        X_valid = pk.load(f)
    with open(PROCESSED_DATA_DIR / 'y_valid.pkl', 'wb') as f:
        y_valid = pk.load(f)

    return create_batches(X_train, y_train), create_batches(X_valid, y_valid), output_shape


def load_params(
    stage: str
) -> dict:
    """
    """
    params_path = Path("params.yaml")

    # Read data preparation parameters
    with open(params_path, "r") as params_file:
        try:
            params = yaml.safe_load(params_file)
            params = params[stage]
        except yaml.YAMLError as exc:
            print(exc)

    return params