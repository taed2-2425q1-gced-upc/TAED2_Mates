import typer
import mlflow
import pickle as pk
import tf_keras

from mates.config import INPUT_SHAPE, MODELS_DIR
from mates.features import create_model, load_processed_data, load_params

app = typer.Typer()


@app.command()
def train(
):
    """
    """

    params = load_params("train")
    
    mlflow.set_experiment(params["experiment_name"])
    mlflow.sklearn.autolog(log_model_signatures=True, log_datasets=True)

    with mlflow.start_run():
        train_data, valid_data, output_shape = load_processed_data()
        model = create_model(input_shape=INPUT_SHAPE,
                             output_shape=output_shape,
                             model_url=params["model_url"])
    
        early_stopping = tf_keras.callbacks.EarlyStopping(monitor=params["monitor"],
                                                          patience=params["patience"])

        model.fit(x=train_data, 
            epochs=params["epochs"],
            validation_data=valid_data, 
            validation_freq=1,
            callbacks=[early_stopping]
            )
        
        if params["save_model"]:
            with open(MODELS_DIR / f"{params["model_name"]}.pkl", "wb") as f:
                pk.dump(model, f)


if __name__ == "__main__":
    app()