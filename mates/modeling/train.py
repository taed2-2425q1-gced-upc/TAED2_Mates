"""
Module to train a model.
"""

from pathlib import Path
import mlflow
import pandas as pd
import tf_keras
import typer
from codecarbon import EmissionsTracker
from loguru import logger

from mates.config import INPUT_SHAPE, METRICS_DIR, MODELS_DIR
from mates.features import create_model, load_params, load_processed_data

app = typer.Typer()


@app.command()
def train(
):
    """
    Function to train a model. Loads processed data, trains a model, and saves the model.
    """
    params = load_params("train")

    mlflow.set_experiment(params["experiment_name"])
    mlflow.tensorflow.autolog()

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

        out_file = f"{params['model_name']}_{params['experiment_name']}_emissions.csv"

        # Track the CO2 emissions of training the model
        tracker = EmissionsTracker(project_name="Dog_breed_classification_model",
                                   measure_power_secs=1,
                                   tracking_mode="process",
                                   output_dir=METRICS_DIR,
                                   output_file=out_file,
                                   on_csv_write="update",
                                   default_cpu_power=45,
                                   )
        try:
            tracker.start()
            history = model.fit(x=train_data,
                epochs=params["epochs"],
                validation_data=valid_data,
                validation_freq=1,
                callbacks=[early_stopping]
                )
        finally:
            tracker.stop()

        # Log CO2 with mlflow
        emissions = pd.read_csv(METRICS_DIR / out_file)
        emissions_metrics = emissions.iloc[-1, 4:13].to_dict()
        emissions_params = emissions.iloc[-1, 13:].to_dict()
        mlflow.log_params(emissions_params)
        mlflow.log_metrics(emissions_metrics)

        # Log additional metrics from the History object
        log = list(history.history.items())
        for epoch, metrics in enumerate(log):
            metric_name = metrics[0]
            metric_values = metrics[1][0]
            mlflow.log_metric(f"train_{metric_name}", metric_values, step=epoch)

        # Save the model as a pickle file
        Path("models").mkdir(exist_ok=True)

        logger.success("Model training complete.")
        if params["save_model"]:
            model.save(MODELS_DIR / f"{params['model_name']}.h5")
            mlflow.tensorflow.log_model(model, params['model_name'])


if __name__ == "__main__":
    app()
