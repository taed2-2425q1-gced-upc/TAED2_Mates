"""
API module for the Mates application.

This module provides a REST API for dog breed classification using pre-trained models.
Users can upload images to the API and receive predictions for the dog's breed. The 
API is built with FastAPI and leverages TensorFlow/Keras models for predictions.

Key Features:
1. FastAPI Integration: 
   - Provides an efficient and scalable REST API for real-time predictions and batch 
     processing of images.
   
2. Model Loading and Management: 
   - Automatically loads all pre-trained models from a specified directory (MODELS_DIR)
     during the API startup, and stores them in a global dictionary for quick access.
   
3. Image Upload and Processing: 
   - Users can upload dog images in common formats (JPEG, PNG), and the API processes 
     the image (resizing and scaling) before feeding it into the model for prediction.
   
4. Prediction with Pre-trained Models: 
   - Leverages a fine-tuned MobileNetV3 model for dog breed classification across 
     over 120 breeds. The predicted breed is returned to the user.

5. Command-line Interface (CLI): 
   - Additionally, it includes a CLI to batch process test datasets for model 
     predictions, saving the results as a CSV file for further evaluation.

Dependencies:
- FastAPI: For building the API and managing requests.
- Pandas: For handling data outputs and saving results.
- PIL (Pillow): For image loading and processing.
- TensorFlow/Keras: For loading pre-trained models and making predictions.
- Loguru: For structured logging and tracking.
- Typer: For creating the CLI.

API Workflow:
1. Model Initialization:
   - At startup, all `.h5` model files in the MODELS_DIR are loaded into memory and 
     stored in the `models_dict` dictionary. Labels (dog breeds) associated with 
     each model are also loaded and stored globally for prediction use.
   
2. Image Prediction:
   - Upon receiving an image via the `/predict` endpoint, the image is preprocessed 
     (resized to IMG_SIZE, scaled) and passed through the selected pre-trained model 
     for classification.
   - The predicted breed label is returned to the client as a JSON response.

3. Error Handling:
   - If an invalid model name is provided or the uploaded image is corrupt/unsupported, 
     appropriate error messages are returned using HTTP status codes.

4. Model Listing:
   - The `/models` endpoint provides a list of all available pre-trained models 
     for users to select from.

CLI Features:
- The command-line interface (CLI) allows users to generate predictions on a batch 
  of test images by specifying a pre-trained model, processing test data, and saving 
  the results as a CSV file for further analysis.

Key Functions:
- `predict_test`: 
   - Loads a specified pre-trained model, processes the test dataset in batches, 
     generates predictions, and saves the predictions as a CSV file in the output 
     directory.

- `predict_single`: 
   - Takes a single image as input, processes it, and returns the predicted dog breed.

API Endpoints:
- `/`: Root endpoint that returns a welcome message and checks API health.
- `/models`: Returns a list of all available models for classification.
- `/predict`: Takes an image and the name of the selected model and returns the 
  predicted dog breed as a JSON response.
"""

import logging
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import List

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger
from PIL import Image

from mates.config import MODELS_DIR, RAW_DATA_DIR
from mates.features import load_model, read_labels
from mates.modeling.predict import predict_single

# Global variable to store models
models_dict = {}
encoding_labels = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads all model files from MODELS_DIR and adds them to models_dict."""
    global models_dict
    global encoding_labels

    # Load all models from MODELS_DIR
    model_paths = [
        filename.stem
        for filename in MODELS_DIR.iterdir()
        if filename.suffix == ".h5"
    ]

    for model_name in model_paths:
        models_dict[model_name] = {"model": load_model(model_name)}

    # Load labels
    _, encoding_labels = read_labels(RAW_DATA_DIR)

    yield

    models_dict.clear()
    encoding_labels = []


logger.info(f"Initalizing FastAPI application...")

# Define application
app = FastAPI(
    title="Mates API for Dog Breed Classification",
    description="This API classifies dog breeds using a pre-trained MobileNetV3 model.",
    version="0.1",
    lifespan=lifespan,
)


@app.get("/", tags=["General"])
async def _index():
    """Root endpoint."""

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"message": "Welcome to the Mates API!"},
    }
    return response


@app.get("/models", response_model=List[str], tags=["Models"])
async def _get_models():
    """Endpoint to list all available models"""
    return list(models_dict.keys())


@app.post("/predict", tags=["Prediction"])
async def _predict_dog_breed(model_name: str, file: UploadFile = File(...)):
    """Endpoint to upload an image and get a dog breed prediction"""

    if model_name not in models_dict:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"Model {model_name} not found. Please choose from available models."
        )

    try:
        image = Image.open(file.file)
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"Invalid image format: {e}"
        )

    # Predict breed using the selected model
    try:
        prediction = predict_single(models_dict[model_name]["model"], encoding_labels, image)
        return JSONResponse(
            status_code=HTTPStatus.OK,
            content={
                "model": model_name,
                "prediction": prediction
            }
        )
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="An error occurred during prediction."
        )