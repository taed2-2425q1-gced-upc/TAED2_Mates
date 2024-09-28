import typer
import tensorflow_hub as hub
import tensorflow.keras as tfk
import tf_keras
from sklearn.model_selection import train_test_split 


from mates.config import MODEL_URL, IMG_SIZE
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


@app.command()
def load_data(
    is_train: bool = True,
    test_size: float = 0.3,
    seed: int = 42,
    # save_processed: bool = False,
): 
    """
    """

    X, y, encoding_labels = read_data(is_train)
    
    if is_train:
        output_shape = len(encoding_labels)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=seed)

        train_data = create_batches(X_train, y_train)
        valid_data = create_batches(X_val, y_val, valid_data=True)
        return train_data, valid_data, output_shape

    test_data = create_batches(X, y)
    return test_data, None, None