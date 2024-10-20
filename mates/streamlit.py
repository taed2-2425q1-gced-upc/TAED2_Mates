"""
AAA
"""

import io
import json
import requests
from PIL import Image
import streamlit as st
from loguru import logger

from config import DATA_DIR, FIGURES_APP_DIR


# Define the API URL (modify this to your FastAPI server URL)
API_URL = "http://localhost:5000/"
logger.info(f"Initalizing Streamlit app with API URL: {API_URL}")

# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state.page = "Welcome Page"  # Start on the Welcome Page by default

# Function to handle page navigation
def set_page(page_name: str):
    """
    Set the page to navigate to based on the button clicked.

    Parameters
    ----------
    page_name : str
        Name of the page to navigate

    Returns
    -------
    None
    """
    st.session_state.page = page_name

# Sidebar with clickable text links styled as buttons
st.sidebar.title("Navigation")

if st.sidebar.button("Main Page"):
    set_page("Main Page")
if st.sidebar.button("Assistance"):
    set_page("Assistance")

############################################
### Welcome Page
############################################

if st.session_state.page == "Welcome Page":
    st.title("Welcome to the Dog Breed Classification App!")
    st.write("""
    Welcome! This app helps you classify dog breeds using machine learning models.     
    """)
    st.write("Upload a photo, and let's unveil the unique characteristics of our furry friends!")
    st.write("")

    if st.button("Go to Prediction Page"):
        set_page("Main Page")  # Navigate to the Main Prediction Page

    st.write("##")
    st.image(str(FIGURES_APP_DIR / "footer.jpg"))

############################################
### Main Page
############################################

elif st.session_state.page == "Main Page":
    st.title("Dog Breed Classification App")

    st.write("""
        Welcome to the Dog Breed Classification App!
    """)
    st.write("""Upload an image of a dog and find its breed using machine learning models.""")
    st.write("")
    st.write("")

    # Function to fetch available models from the FastAPI
    def get_available_models():
        """
        Fetch the available models from the FastAPI server.

        Returns:
        --------
        list
            List of available models
        """
        response = requests.get(f"{API_URL}models", timeout=10)
        if response.status_code == 200:
            return response.json()
        st.error("Failed to load models. Please try again.")
        return []

    # Get the available models
    available_models = get_available_models()

    # Display the model selection dropdown on the main page (not the sidebar)
    selected_model = []
    if available_models:
        selected_model = st.selectbox("Select a Model", available_models)

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
        files = {'file': ('dog_image.jpg', img_bytes, 'image/jpeg')}
        res = requests.post(f"{API_URL}/predict",
                            params={'model_name': selected_model},
                            files=files,
                            timeout=10)

        if res.status_code == 200:
            prediction = res.json()["prediction"]
            st.success(f"Predicted Breed: {prediction}")
        else:
            st.error(f"Prediction failed: {res.json()['detail']}")
    else:
        st.info("Please upload an image and select a model.")

    st.write("##")
    st.image(str(FIGURES_APP_DIR / "footer.jpg"))

############################################
### Assistance Page
############################################

elif st.session_state.page == "Assistance":
    st.title("Assistance")
    st.write("Find out more about our little furry friends!")

    # List of dog breeds, images, and descriptions for the search
    def get_dog_data():
        """
        Get the dog breed data from customized catalogue dictionary.
        
        Returns:
        --------
        list[dict]:
            Dictionary of dog breeds and their corresponding group.
        """
        with open(DATA_DIR / "dog_catalogue_data.json", 'r', encoding='utf-8') as f:
            return json.load(f)

    dog_data = get_dog_data()
    # Breed names for the search dropdown
    breed_names = [dog['name'] for dog in dog_data]

    st.header("Dog Image Gallery")

    # Search bar for selecting a dog breed (empty on load with a placeholder)
    search_term = st.selectbox("Search for a dog breed here!",
                               [""] + breed_names,
                               format_func=lambda x: "" if x == "" else x)

    # Display breed information if the user searches for a breed
    if search_term:
        # Find the selected dog breed from the data
        selected_dog = next(dog for dog in dog_data if dog["name"] == search_term)

        with st.expander(f"{selected_dog['name']} Information", expanded=True):
            st.image(selected_dog["image_path"],
                     caption=selected_dog['name'],
                     use_column_width=True)
            st.subheader(selected_dog['name'])
            st.write(selected_dog["description"])

    st.write("")
    st.write("")
    st.write("")

    # Create a grid of images with buttons (same as before)
    columns = st.columns(4)
    for idx, dog in enumerate(dog_data):
        col_idx = idx % 4  # Ensure 4 columns per row
        with columns[col_idx]:
            # Display image of the dog
            st.image(str(FIGURES_APP_DIR / dog["image_path"]))

            # Show an expander with dog details when button is clicked
            with st.expander(f"{dog['name']}", expanded=False):
                # st.image(dog["image_path"], caption=dog["name"], use_column_width=True)
                st.subheader(dog["name"])
                st.write(dog["description"])

    st.write("##")
    st.image(str(FIGURES_APP_DIR / "footer.jpg"))
