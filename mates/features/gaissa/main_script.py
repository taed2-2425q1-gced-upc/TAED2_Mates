"""
Module for executing the Plugin for generating model metrics.

This module provides a command-line interface to interact with the 
PlugIn class, allowing users to input the model's file path, 
output directory, and desired filename for the output CSV file. 
The main function orchestrates the process of generating the 
model metrics using the specified plugin.

Dependencies:
-------------
- gaissaplugin: Importing the PlugIn class that implements the plugin interface.
"""

from mates.features.gaissa.gaissaplugin import PlugIn


def main():
    """
    Main function to execute the plugin for generating model metrics.

    This function prompts the user for the model's file path, output directory,
    and filename, then instantiates the PlugIn class and calls its method to
    generate the output CSV file with the model's metrics.
    """
    # Get user input for model path and output directory and filename
    model_path = input("Enter the model's file path: ")
    output_directory = input("Enter the output directory: ")
    filename = input("Enter the filename: ")

    # Instantiate the plugin
    plugin = PlugIn()

    # Call the plugin to generate the output
    plugin.generate_output(model_path, output_directory, filename)


if __name__ == "__main__":
    main()
