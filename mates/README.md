# Mates

## Overview

This directory contains all the source code for the project. It is structured as a Python package:

- **`__init__.py`**: Marks this directory as a Python package. It can be used to initialize package-level imports or settings if needed.

- **config.py**: Contains configuration settings for the project (paths, hyperparameters, etc.).

- **features.py**: Handles feature engineering, including transformations and new feature creation.

- **modeling/**: Contains the core code for modeling tasks:
  - **`__init__.py`**: Marks the `modeling` directory as a subpackage.
  - **prepare.py**: Prepares the data for training, such as splitting datasets or normalizing features.
  - **train.py**: Contains the training loop and model optimization code.
  - **predict.py**: Runs inference on new data using the trained models.


## Usage

- Import modules from `mates` in your scripts or notebooks.
- Make sure to update `config.py` with your specific settings before running the code.
