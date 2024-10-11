import streamlit as st
import requests
from PIL import Image
import io

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from mates.config import PORT


API_URL = f"http://localhost:{PORT}"


def get_available_models():
    """Fetch the list of available models from the API."""
    response = requests.get(f"{API_URL}/models")
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Failed to retrieve models.")
        return []

def predict_breed(model_name, image_file):
    """Send the image and model name to the API for prediction."""
    files = {"file": image_file}
    data = {"model_name": model_name}
    response = requests.post(f"{API_URL}/predict", files=files, data=data)

    if response.status_code == 200:
        return response.json()["prediction"]
    else:
        st.error(f"Prediction failed: {response.json()['detail']}")
        return None

# Streamlit app layout
st.title("Dog Breed Classifier")
st.write("Select a model, upload a dog image, and get a breed prediction.")

# Step 1: Model selection
st.header("Step 1: Select a model")
models = get_available_models()
if models:
    selected_model = st.selectbox("Choose a model", models)

# Step 2: Image upload
st.header("Step 2: Upload an image")
uploaded_file = st.file_uploader("Upload an image of a dog", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Step 3: Prediction
    if st.button("Predict"):
        with st.spinner("Predicting..."):
            prediction = predict_breed(selected_model, uploaded_file)
            if prediction:
                st.success(f"Predicted Dog Breed: {prediction}")
