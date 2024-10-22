"""
Module for generating output metrics for TensorFlow Keras models.

This module defines a `PlugIn` class that implements the `PluginInterface`.
The class is responsible for generating a CSV output file containing metrics
for a specified Keras model, using the `calculateH5` function.

Dependencies:
-------------
- os: For handling file paths.
- plugin_interface: For defining the PluginInterface.
- calculator: For calculating and saving model metrics.
"""

import os

from mates.features.gaissa.calculator import calculate_h5
from mates.features.gaissa.plugin_interface import PluginInterface


class PlugIn(PluginInterface):
    """
    Class for generating output metrics for a Keras model.

    This class implements the `PluginInterface` and provides a method to
    calculate model metrics and save them to a specified output file.

    Methods:
    --------
    generate_output(model_path, output_directory, filename):
        Generates a CSV file containing metrics for the specified Keras model.
    """

    # pylint: disable=R0903
    def generate_output(self, model_path, output_directory, filename):
        """
        Generate a CSV output file with model metrics.

        This method creates a full path for the output file using the specified
        output directory and filename, then calls the `calculateH5` function
        to compute and save the model metrics to the CSV file.

        Parameters:
        -----------
        model_path : str
            Path to the Keras model file (H5 format).
        output_directory : str
            Directory where the output CSV file will be saved.
        filename : str
            Name of the output CSV file (without extension).

        Returns:
        --------
        None
        """
        # Create the full path for the output file
        output_path = os.path.join(output_directory, filename + ".csv")

        calculate_h5(model_path, output_path)

        print(f"Output file '{filename}' generated in '{output_directory}'")
