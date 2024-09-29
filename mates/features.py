import typer
import tensorflow_hub as hub
import tf_keras
import pickle as pk


from mates.config import MODEL_URL, IMG_SIZE, PROCESSED_DATA_DIR
from mates.utils import read_data, create_batches

app = typer.Typer()


@app.command()
def create_model(
    input_shape: list,
    output_shape: int,
    loss_function = tf_keras.losses.CategoricalCrossentropy(),
    optimizer = tf_keras.optimizers.Adam(),
    activation_function: str = 'softmax',
    metrics: list = ['accuracy'],
    model_url: str = MODEL_URL,
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
    try:
        with open(PROCESSED_DATA_DIR / 'train_data.pkl', 'rb') as f:
            train_data = pk.load(f)
        with open(PROCESSED_DATA_DIR / 'valid_data.pkl', 'rb') as f:
            valid_data = pk.load(f)
        with open(PROCESSED_DATA_DIR / 'output_shape.pkl', 'rb') as f:
            output_shape = pk.load(f)
    except Exception as e:
        raise Exception(f"Error loading processed data: {e}")
    
    return train_data, valid_data, output_shape
