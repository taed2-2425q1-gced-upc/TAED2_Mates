import typer
import mlflow
import pickle as pk
import tensorflow as tf
import tf_keras

from mates.config import EPOCHS, INPUT_SHAPE, MODELS_DIR
from mates.features import create_model, load_processed_data

app = typer.Typer()


@app.command()
def train(
    experiment_name: str,
    model_name: str,
    epochs: int = EPOCHS,
    save_model: bool = True,
):
    """
    """
    
    mlflow.set_experiment(experiment_name)
    mlflow.sklearn.autolog(log_model_signatures=True, log_datasets=True)

    with mlflow.start_run():
        train_data, valid_data, output_shape = load_processed_data()
        model = create_model(input_shape=INPUT_SHAPE, output_shape=output_shape)
    
        early_stopping = tf_keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)

        model.fit(x=train_data, 
            epochs=epochs, 
            validation_data=valid_data, 
            validation_freq=1,
            callbacks=[early_stopping]
            )
        
        if save_model:
            with open(MODELS_DIR / f"{model_name}.pkl", "wb") as f:
                pk.dump(model, f)


if __name__ == "__main__":
    app()