import streamlit as st
import requests
from PIL import Image
import io

# Define the API URL (modify this to your FastAPI server URL)
API_URL = "http://localhost:5000/"  # Replace with your actual FastAPI URL

# Streamlit app title
st.title("Dog Breed Classification App")

# Sidebar for selecting a model
st.sidebar.title("Model Selection")

# Function to fetch available models from the FastAPI
def get_available_models():
    response = requests.get(f"{API_URL}models")
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Failed to load models. Please try again.")
        return []

# Get the available models
available_models = get_available_models()

# If models are available, allow user to select one
if available_models:
    selected_model = st.sidebar.selectbox("Select Model", available_models)

# File uploader for image input
uploaded_file = st.file_uploader("Upload an image of a dog", type=["jpg", "jpeg", "png"])

# Display uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

# Predict button
if st.button("Predict") and uploaded_file is not None and selected_model:
    # Prepare the file for the request
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)  # Reset file pointer to the start

    # Make the prediction request to the FastAPI
    try:
        files = {'file': ('dog_image.jpg', img_bytes, 'image/jpeg')}
        response = requests.post(f"{API_URL}/predict", params={'model_name': selected_model}, files=files)
        
        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.success(f"Predicted Breed: {prediction}")
        else:
            st.error(f"Prediction failed: {response.json()['detail']}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload an image and select a model.")
