from http import HTTPStatus
from PIL import Image
import pytest
from fastapi.testclient import TestClient
from io import BytesIO

from mates.app.api import app


@pytest.fixture(scope="module", autouse=True)
def client():
    # Use the TestClient with a `with` statement to trigger the startup and shutdown events.
    with TestClient(app) as client:
        yield client

@pytest.fixture
def valid_image():
    """Fixture to provide a valid image for testing."""
    with open('tests/bit.jpg', 'rb') as f:  # Adjust path as necessary
        image_data = BytesIO(f.read())
    return image_data


def test_root_endpoint(client):
    response = client.get("/")
    json = response.json()
    assert response.status_code == 200
    assert (
        json["data"]["message"]
        == "Welcome to the Mates API!"
    )
    assert json["message"] == "OK"
    assert json["status-code"] == 200

def test_get_models(client):
    response = client.get("/models")
    json = response.json()
    print(json)
    assert json == ['mobilenet_exp_batch_62', 'mobilenet_exp_batch_32']

def test_model_prediction(client, valid_image):
    response = client.post('/predict?model_name=mobilenet_exp_batch_62',
                           files={"file": ("test.jpg", valid_image, "image/jpeg")})
    json = response.json()
    assert response.status_code == 200
    assert json=={
            "model": "mobilenet_exp_batch_62",
            "prediction": "border_collie"
        }

def test_model_prediction_not_found(client, valid_image):
    response = client.post('/predict?model_name=invalid_model',
                           files={"file": ("test.jpg", valid_image, "image/png")})
    json = response.json()
    assert response.status_code == 400
    assert json["detail"] == "Model invalid_model not found. Please choose from available models."

def test_model_prediction_bad_image(client):
    response = client.post('/predict?model_name=mobilenet_exp_batch_62', 
                           files = {"file": ("test.txt", "invalid content", "text")})
    assert response.status_code == 400
    assert "Invalid image format" in response.json()["detail"]