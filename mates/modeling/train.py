import typer
import mlflow
import tf_keras
import pickle as pk
from loguru import logger


from mates.config import INPUT_SHAPE, MODELS_DIR
from mates.features import create_model, load_processed_data, load_params

app = typer.Typer()


@app.command()
def train(
):
    """
    Function to train a model. Loads processed data, trains a model, and saves the model.
    """
    params = load_params("train")
    
    mlflow.set_experiment(params["experiment_name"])
    mlflow.sklearn.autolog(log_model_signatures=True, log_datasets=True)

    with mlflow.start_run():
        logger.info("Processing dataset...")
        train_data, valid_data, output_shape = load_processed_data(params["batch_size"])

        logger.info("Training model...")
        model = create_model(input_shape=INPUT_SHAPE,
                             output_shape=output_shape,
                             model_url=params["model_url"],
                             optimizer=params["optimizer"],
                             metrics=params["metrics"]
                            )
    
        early_stopping = tf_keras.callbacks.EarlyStopping(monitor=params["monitor"],
                                                          patience=params["patience"])

        model.fit(x=train_data, 
            epochs=params["epochs"],
            validation_data=valid_data, 
            validation_freq=1,
            callbacks=[early_stopping]
            )

        logger.success("Model training complete.")
        if params["save_model"]:
            with open(MODELS_DIR / f"{params['model_name']}.pkl", "wb") as f:
                pk.dump(model, f)


if __name__ == "__main__":
    app()