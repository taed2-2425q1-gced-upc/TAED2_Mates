"""
Module for utility functions used in the mates project.

This module provides various utility functions for loading configuration parameters from YAML files.

Commands available:
- load_params: Loads parameters from a YAML configuration file based on the pipeline stage.

Dependencies:
- yaml: For loading configuration parameters from YAML files.

"""

from pathlib import Path

import typer
import yaml


app = typer.Typer()

@app.command()
def load_params(
    stage: str
) -> dict:
    """
    Load parameters from the params.yaml configuration file.

    Parameters
    ----------
    stage : str
        Stage of the pipeline (e.g., 'train', 'predict').

    Returns
    -------
    params : dict
        Dictionary of parameters for the specified stage.
    """
    params_path = Path("params.yaml")

    with open(params_path, "r", encoding='utf-8') as params_file:
        try:
            params = yaml.safe_load(params_file)
            params = params[stage]
        except yaml.YAMLError as exc:
            print(exc)

    return params
