"""
Module to test the functions of the API.
"""

from http import HTTPStatus
from io import BytesIO
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from mates.app.api import app


@pytest.fixture(scope="module")
def api_client():
    """Fixture that provides a TestClient for testing the FastAPI application."""
    # Patching read_labels globally, so the mock is available during app startup
    dog_breeds = ["dog" for _ in range(120)]
    dog_breeds[16] = "border_collie"
    with patch("mates.app.api.read_labels", return_value=(None, dog_breeds)):
        with TestClient(app) as client:
            yield client


@pytest.fixture
def sample_image_file():
    """Fixture to provide a valid image for testing."""
    with open("tests/test_images/bit.jpg", "rb") as f:  # Adjust path as necessary
        image_data = BytesIO(f.read())
    return image_data


def test_root_endpoint(api_client):
    """Test the root endpoint of the API."""
    response = api_client.get("/")
    json = response.json()
    assert response.status_code == 200
    assert json["data"]["message"] == "Welcome to the Mates API!"
    assert json["message"] == "OK"
    assert json["status-code"] == 200


def test_get_models(api_client):
    """Test the endpoint that retrieves available models."""
    response = api_client.get("/models")
    json = response.json()
    assert json == ["mobilenet_exp_batch_62", "mobilenet_exp_batch_32"]


def test_model_prediction(api_client, sample_image_file):
    """Test the model prediction endpoint with a valid image."""
    response = api_client.post(
        "/predict?model_name=mobilenet_exp_batch_62",
        files={"file": ("test.jpg", sample_image_file, "image/jpeg")},
    )
    json = response.json()
    print(json)
    assert response.status_code == 200
    assert json == {"model": "mobilenet_exp_batch_62", "prediction": "border_collie"}


def test_model_prediction_not_found(api_client, sample_image_file):
    """Test the model prediction endpoint with an invalid model name."""
    response = api_client.post(
        "/predict?model_name=invalid_model",
        files={"file": ("test.jpg", sample_image_file, "image/png")},
    )
    json = response.json()
    assert response.status_code == 400
    assert json["detail"] == "Model invalid_model not found. Please choose from available models."


def test_model_prediction_bad_image(api_client):
    """Test the model prediction endpoint with an invalid image format."""
    response = api_client.post(
        "/predict?model_name=mobilenet_exp_batch_62",
        files={"file": ("test.txt", "invalid content", "text")},
    )
    assert response.status_code == 400
    assert "Invalid image format" in response.json()["detail"]


def test_predict_dog_breed_internal_server_error(api_client, sample_image_file):
    """Test the model prediction endpoint for handling internal server errors."""
    with patch("mates.app.api.predict_single", side_effect=Exception("error")):
        response = api_client.post(
            "/predict?model_name=mobilenet_exp_batch_62",
            files={"file": ("test.jpg", sample_image_file, "image/jpeg")},
        )

        assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
        assert response.json() == {"detail": "An error occurred during prediction."}