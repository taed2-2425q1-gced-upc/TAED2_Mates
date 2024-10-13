from loguru import logger
import logging
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import List
import pandas as pd


from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from mates.features import load_model
from mates.config import MODELS_DIR
from PIL import Image


from mates.modeling.predict import predict_single
from mates.features import read_labels
from mates.config import RAW_DATA_DIR

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