"""
Module for training a machine learning model using processed data and tracking emissions.

This module provides a command-line interface (CLI) to load processed data, train a model,
and save the model and its metrics. It also tracks CO2 emissions during the training process
using the CodeCarbon library and logs relevant metrics to MLflow for experiment tracking.

Commands available:
- `train`: Trains a model with multiple combinations of hyperparameters. This function calls
    the `single_train` function for each combination of hyperparameters.

- `single_train`: Trains a model with a single set of hyperparameters. This function is called
    by the `train` function for each combination of hyperparameters.


Workflow:
1. Load training parameters from a YAML configuration file.
2. Set up an MLflow experiment and enable automatic logging for TensorFlow.
3. Load the processed training and validation datasets.
4. Build a machine learning model using a pre-trained model URL and specified parameters.
5. Train the model using the loaded data, with early stopping based on validation performance.
6. Track and log CO2 emissions during the training process using CodeCarbon.
7. Log the training metrics, CO2 emissions, and model parameters to MLflow.
8. Optionally, save the trained model as an HDF5 file and log it to MLflow.

Dependencies:
- mlflow
- pandas
- tf_keras
- typer
- codecarbon
- loguru
- pathlib

Additional module imports:
- INPUT_SHAPE, METRICS_DIR, MODELS_DIR from mates.config: Paths and model shape settings.
- create_model, load_params, load_processed_data from mates.features: Functions for loading
    data, model parameters, and creating models.
"""

import itertools
from pathlib import Path

import mlflow
import pandas as pd
import tf_keras
import typer
from codecarbon import EmissionsTracker
from loguru import logger

from mates.config import INPUT_SHAPE, METRICS_DIR, MODELS_DIR
from mates.features.features import create_model, load_processed_data
from mates.features.utils import load_params

app = typer.Typer()


@app.command()
def single_train(params: dict):
    """
    Function to train a model. Loads processed data, trains a model, and saves the model.

    Parameters:
    ----------
    params: dict
        Dictionary containing the training parameters.
    """
    mlflow.tensorflow.autolog()
    mlflow.set_experiment(params["experiment_name"])

    with mlflow.start_run(run_name=f"{params['optimizer']}_{params['batch_size']}"):
        mlflow.log_param("optimizer", params["optimizer"])
        mlflow.log_param("model_url", params["model_url"])
        mlflow.log_param("batch_size", params["batch_size"])
        mlflow.log_param("epochs", params["epochs"])
        logger.info("Processing dataset...")
        train_data, valid_data, output_shape = load_processed_data(params["batch_size"])

        logger.info("Training model...")
        model = create_model(
            input_shape=INPUT_SHAPE,
            output_shape=output_shape,
            model_url=params["model_url"],
            optimizer=params["optimizer"],
            metrics=params["metrics"],
        )

        early_stopping = tf_keras.callbacks.EarlyStopping(
            monitor=params["monitor"], patience=params["patience"]
        )

        out_file = f"{params['model_name']}_{params['experiment_name']}_\
            {params['batch_size']}_emissions.csv"

        # Track the CO2 emissions of training the model
        tracker = EmissionsTracker(
            project_name="Dog_breed_classification_model",
            measure_power_secs=1,
            tracking_mode="process",
            output_dir=METRICS_DIR,
            output_file=out_file,
            on_csv_write="update",
            default_cpu_power=45,
        )
        try:
            tracker.start()
            history = model.fit(
                x=train_data,
                epochs=params["epochs"],
                validation_data=valid_data,
                validation_freq=1,
                callbacks=[early_stopping],
            )
        finally:
            tracker.stop()

        # Log CO2 with mlflow
        emissions = pd.read_csv(METRICS_DIR / out_file)
        mlflow.log_params(emissions.iloc[-1, 13:].to_dict())
        mlflow.log_metrics(emissions.iloc[-1, 4:13].to_dict())

        # Log additional metrics from the History object
        log = list(history.history.items())

        for _, metrics in enumerate(log):
            metric_name = metrics[0]
            for epoch, metric_values in enumerate(metrics[1]):
                mlflow.log_metric(metric_name, metric_values, step=epoch)

        Path("models").mkdir(exist_ok=True)

        logger.success("Model training complete.")
        if params["save_model"]:
            model.save(
                MODELS_DIR
                / f"{params['model_name']}_{params['experiment_name']}_{params['batch_size']}.h5"
            )
            mlflow.tensorflow.log_model(
                model, f"{params['model_name']}_{params['experiment_name']}_{params['batch_size']}"
            )


@app.command()
def train():
    """
    Function to train a model with multiple combinations of hyperparameters.
    """
    params = load_params("train")

    hyperparams = params["hyperparams"]
    epochs_options = hyperparams["epochs"]
    batch_size_options = hyperparams["batch_size"]
    optimizer_options = hyperparams["optimizer"]

    # Generate all combinations of hyperparameters
    combs = list(itertools.product(epochs_options, batch_size_options, optimizer_options))

    # Iterate over each combination of hyperparameters
    for epochs, batch_size, optimizer in combs:
        params["epochs"] = epochs
        params["batch_size"] = batch_size
        params["optimizer"] = optimizer

        logger.info(
            f"Starting training with epochs={epochs}, batch={batch_size}, optimizer={optimizer}"
        )

        single_train(params)


if __name__ == "__main__":
    train()
