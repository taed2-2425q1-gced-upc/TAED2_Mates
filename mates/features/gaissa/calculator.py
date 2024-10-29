"""
Module for calculating and saving metrics of TensorFlow Keras models.

This module provides a function, `calculateH5`, which loads a Keras model
from a specified file path and computes important metrics such as:
- The number of parameters in the model.
- The file size of the model in megabytes.
- The floating point operations per second (FLOPS) during inference.

The results are then written to a CSV file, allowing for easy analysis
and documentation of the model's performance characteristics.

Dependencies:
-------------
- TensorFlow: For loading Keras models and performing calculations.
- TensorFlow Hub: For handling custom layers in Keras models.
- CSV: For writing the metrics to a CSV file.
- OS: For handling file paths and obtaining file sizes.

Functions:
----------
- calculate_h5(model_path, output_path): Calculate model metrics and save them to a CSV file.
"""

import csv
import os

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2_as_graph,
)
from tf_keras.models import load_model


def calculate_h5(model_path, output_path):
    """
    Calculate and save metrics for a TensorFlow model.

    This function loads a Keras model from the specified path, calculates
    the number of parameters, the model file size in megabytes, and the
    floating point operations per second (FLOPS) of the model when
    performing inference. The results are written to a CSV file.

    Parameters:
    ----------
    model_path : str
        Path to the Keras model file (H5 format).
    output_path : str
        Path where the output CSV file will be saved. The CSV will contain
        the model's size, file size, and FLOPS.

    Returns:
    -------
    None
    """
    # Load the Keras model from the specified path
    model = load_model(model_path, custom_objects={"KerasLayer": hub.KerasLayer})

    # Get number of parameters
    num_params = model.count_params()

    # Get the file size in bytes and convert to megabytes
    model_size_bytes = os.path.getsize(model_path)
    model_size_mb = model_size_bytes / (1024 * 1024)

    # Convert tf.keras model into a frozen graph to count FLOPS
    # FLOPS depends on batch size
    inputs = [tf.TensorSpec([1] + inp.shape[1:], inp.dtype) for inp in model.inputs]
    real_model = tf.function(model).get_concrete_function(inputs)
    frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

    # Calculate FLOPS with tf.profiler
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(
        graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
    )

    # Writing the metrics to a CSV file
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        # Write the header
        writer.writerow(["model_size", "file_size", "flops"])

        # Write the calculated values
        writer.writerow([num_params, model_size_mb, flops.total_float_ops])
