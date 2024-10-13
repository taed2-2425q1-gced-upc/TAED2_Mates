"""
Module for configuring paths and model parameters for training.

This module provides the configuration settings necessary for loading data, saving models,
and defining key parameters for the machine learning model. It utilizes **pathlib** to manage
file paths and **dotenv** to load environment variables from a .env file.

Commands available:
- PROJ_ROOT: Root path of the project directory.
- DATA_DIR: Main directory for storing data.
- RAW_DATA_DIR: Directory for storing raw data.
- INTERIM_DATA_DIR: Directory for storing interim processed data.
- PROCESSED_DATA_DIR: Directory for storing fully processed data.
- EXTERNAL_DATA_DIR: Directory for storing external data.
- MODELS_DIR: Directory for saving and loading machine learning models.
- METRICS_DIR: Directory for storing model metrics.
- REPORTS_DIR: Directory for generating reports, including visualizations.
- FIGURES_DIR: Directory for storing generated figures.
  
Model parameters:
- IMG_SIZE: Defines the size of the images used in the model (128x128).
- INPUT_SHAPE: Defines the input shape for the model, with a batch size (None)
    and the specified image dimensions (128x128) in 3 color channels (RGB).

Dependencies:
- dotenv
- loguru
- pathlib
"""


from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DATA_DIR = DATA_DIR / "output"

MODELS_DIR = PROJ_ROOT / "models"
METRICS_DIR = PROJ_ROOT / "metrics"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

PORT = 5000


# Model parameters
IMG_SIZE = 128
INPUT_SHAPE = [None, IMG_SIZE, IMG_SIZE, 3]
