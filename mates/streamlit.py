"""
Dog Breed Classification Application

This module implements a Streamlit web application for classifying dog breeds
using machine learning models. The application interacts with a FastAPI backend
to fetch available models and make predictions based on uploaded dog images.

Features:
---------
- Users can navigate between a welcome page, a main prediction page, and an assistance page.
- The main page allows users to upload an image of a dog, select a machine learning model,
  and receive breed predictions.
- The assistance page provides information about various dog breeds from a custom 
  catalogue, including images and descriptions.

Dependencies:
-------------
- streamlit: For building the web interface.
- requests: For making API calls to the FastAPI backend.
- PIL (Pillow): For image handling.
- loguru: For logging information.
- json: For loading breed data from a JSON file.

Usage:
------
To run the application, use the command:
    streamlit run <this_script_name.py>

Make sure the FastAPI backend is running at the specified API_URL before using the app.
"""

import base64
import io
import json
import sys
from pathlib import Path

import altair as alt
import mlflow
import pandas as pd
import requests
import streamlit as st
from loguru import logger
from PIL import Image

from mates.config import DATA_DIR, FIGURES_APP_DIR, METRICS_DIR

# Setting path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

# Define the API URL (modify this to your FastAPI server URL)
API_URL = "http://localhost:5000/"  # Local FastAPI server
# API_URL = "http://172.16.4.39:8080/"  # VM FastAPI server
logger.info(f"Initalizing Streamlit app with API URL: {API_URL}")

st.set_page_config(page_title=None, layout="wide")

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

if st.sidebar.button("Prediction Page"):
    set_page("Prediction Page")
if st.sidebar.button("Assistance Page"):
    set_page("Assistance Page")
if st.sidebar.button("Tracking Page"):
    set_page("Tracking Page")

############################################
### Welcome Page
############################################

if st.session_state.page == "Welcome Page":
    st.title("Welcome to the Dog Breed Classification App!")
    st.subheader(
        """
        Welcome! This app is designed to help you classify dog breeds \
        with the power of machine learning. 
        """
    )

    st.write(
        """
        Using a fine-tuned MobileNetV2 model, our app can analyze an uploaded \
        dog image and predict its breed. This project also serves as a hands-on \
        demonstration of best practices in software engineering for machine learning. \
        The Dog Breed Classification App showcases our commitment to building reliable, \
        scalable ML solutions through strong software engineering principles. \
        By using modular code, version control, and automated tracking of training metrics,\
        we ensure a robust and maintainable pipeline.
        """
    )

    st.info(
        "Head over to the Prediction Page to upload an image and reveal the unique \
        characteristics of our furry friends!"
    )

    # Navigate to the Prediction Prediction Page
    if st.button("Go to Prediction Page"):
        set_page("Prediction Page")

    st.write("##")
    st.image(str(FIGURES_APP_DIR / "footer.jpg"), use_column_width="always")

############################################
### Prediction Page
############################################

elif st.session_state.page == "Prediction Page":
    st.title("Dog Breed Classification App")

    st.subheader(
        """
        Welcome to the Prediction Page!
        """
    )

    st.divider()

    st.write(
        """
        **Instructions**:
        1. Select a model from the dropdown list below.
        2. Upload a clear image of a dog (supported formats: JPG, JPEG, PNG).
        3. Click "Predict" to receive the breed classification.
        """
    )

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

    st.write("")
    st.write("")

    # File uploader for image input
    uploaded_file = st.file_uploader("Upload an image of a dog", type=["jpg", "jpeg", "png"])

    # Display uploaded image
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image")

    # Predict button
    if st.button("Predict") and uploaded_file is not None and selected_model:
        # Prepare the file for the request
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG")
        img_bytes.seek(0)  # Reset file pointer to the start

        # Make the prediction request to the FastAPI
        files = {"file": ("dog_image.jpg", img_bytes, "image/jpeg")}
        res = requests.post(
            f"{API_URL}predict", params={"model_name": selected_model}, files=files, timeout=10
        )

        if res.status_code == 200:
            prediction = res.json()["prediction"]
            st.success(f"Predicted Breed: {prediction}")
            st.write("Enjoy discovering the breed details and unique traits!")
        else:
            st.error(f"Prediction failed: {res.json()['detail']}")

    st.write("##")
    st.image(str(FIGURES_APP_DIR / "footer.jpg"), use_column_width="always")

############################################
### Assistance Page Page
############################################

elif st.session_state.page == "Assistance Page":
    st.title("Assistance Page")
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
        with open(DATA_DIR / "dog_catalogue_data.json", "r", encoding="utf-8") as f:
            return json.load(f)

    dog_data = get_dog_data()
    # Breed names for the search dropdown
    breed_names = [dog["name"] for dog in dog_data]

    st.header("Dog Image Gallery")

    # Search bar for selecting a dog breed (empty on load with a placeholder)
    search_term = st.selectbox(
        "Search for a dog breed here!",
        [""] + breed_names,
        format_func=lambda x: "" if x == "" else x,
    )

    # Display breed information if the user searches for a breed
    if search_term:
        # Find the selected dog breed from the data
        selected_dog = next(dog for dog in dog_data if dog["name"] == search_term)

        with st.expander(f"{selected_dog['name']} Information", expanded=True):
            st.image(
                str(FIGURES_APP_DIR / selected_dog["image_path"]),
                caption=selected_dog["name"],
                use_column_width=True,
            )
            st.subheader(selected_dog["name"])
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
    st.image(str(FIGURES_APP_DIR / "footer.jpg"), use_column_width="always")


############################################
### Tracking Page Page
############################################

elif st.session_state.page == "Tracking Page":

    st.title("Tracking Page and monitoring dashboard")
    st.write("Make AI development easy!")

    # Load the CSV files into DataFrames
    emissions_32 = pd.read_csv(METRICS_DIR / "mobilenet_exp_batch_32_emissions.csv")
    emissions_62 = pd.read_csv(METRICS_DIR / "mobilenet_exp_batch_62_emissions.csv")
    emissions_32 = emissions_32[
        [
            "timestamp",
            "run_id",
            "duration",
            "emissions",
            "cpu_power",
            "gpu_power",
            "ram_power",
            "energy_consumed",
            "country_name",
        ]
    ]
    emissions_62 = emissions_32[
        [
            "timestamp",
            "run_id",
            "duration",
            "emissions",
            "cpu_power",
            "gpu_power",
            "ram_power",
            "energy_consumed",
            "country_name",
        ]
    ]
    # Create a combined dataframe for comparison
    emissions_32["batch_size"] = "32"
    emissions_62["batch_size"] = "62"
    combined_emissions = pd.concat([emissions_32, emissions_62])

    # MLFLOW DATA
    mlflow.set_tracking_uri("https://dagshub.com/0J0P0/TAED2_Mates.mlflow/")

    exp_32 = mlflow.get_experiment_by_name("exp_batch_32")
    exp_32_id = exp_32.experiment_id

    exp_62 = mlflow.get_experiment_by_name("exp_batch_62")
    exp_62_id = exp_62.experiment_id

    runs_32 = mlflow.search_runs(experiment_ids=[exp_32_id])
    runs_62 = mlflow.search_runs(experiment_ids=[exp_62_id])

    df_runs_32 = pd.DataFrame(runs_32)
    df_runs_62 = pd.DataFrame(runs_62)

    combined_metrics = pd.concat([df_runs_32, df_runs_62])
    combined_metrics = combined_metrics[
        [
            "run_id",
            "status",
            "metrics.val_accuracy",
            "metrics.accuracy",
            "metrics.loss",
            "metrics.val_loss",
            "params.epochs",
            "params.optimizer",
            "params.batch_size",
        ]
    ]

    # Function to display PDF as base64
    def display_pdf(pdf_file):
        """Display a PDF file using a base64 string."""
        with open(pdf_file, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" \
            width="100%" height="500"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

    # Prediction layout with two columns: one for the PDF, the other for KPIs and plots
    col1, col2 = st.columns([1, 2])

    # Column 1: PDF Viewer
    with col1:
        st.header("Gaissa Label")
        display_pdf(
            METRICS_DIR / "gaissa/gaissa_label.pdf"
        )  # Call the function to display the PDF

    with col2:
        st.header("Key Performance Indicators")

        # Compute KPIs based on your dataset
        avg_duration = combined_emissions["duration"].mean()
        total_emissions = combined_emissions["emissions"].sum()
        avg_energy_consumed = combined_emissions["energy_consumed"].mean()

        # Create KPIs
        kpi1, kpi2, kpi3 = st.columns([1, 1, 1])
        kpi1.metric(label="Avg Duration (s)", value=f"{avg_duration:.4f} s")
        kpi2.metric(label="Total Emissions (kgCO2)", value=f"{total_emissions:.4f} kgCO2")
        kpi3.metric(label="Avg Consumption (kWh)", value=f"{avg_energy_consumed:.4f} kWh")

        st.header("Emission Comparison")

        bar_chart = (
            alt.Chart(combined_emissions)
            .transform_calculate(short_id="substring(datum.run_id, 0, 5)")
            .mark_bar()
            .encode(
                x=alt.X("short_id:N", title="Model", axis=alt.Axis(labelAngle=-45)),
                y=alt.Y("sum(emissions):Q", title="Total Emissions"),
                color="batch_size:N",
                column="batch_size:N",
            )
            .properties(
                title="Emissions Comparison Across Models and Batch Sizes", width=245, height=150
            )
        )

        st.altair_chart(bar_chart)

    col3, col4 = st.columns([2, 1])

    with col3:
        st.header("Model Metrics")
        st.dataframe(combined_metrics, height=200)

        st.header("Model Emissions")
        st.dataframe(combined_emissions, height=200)

    with col4:
        # Data for optimizers and batch sizes
        optim_counts = combined_metrics["params.optimizer"].value_counts().reset_index()
        optim_counts.columns = ["optimizer", "count"]

        batch_size_counts = combined_metrics["params.batch_size"].value_counts().reset_index()
        batch_size_counts.columns = ["batch_size", "count"]

        st.header("Optimizer Stats")

        # Pie chart for optimizers using Altair with smaller dimensions
        optimizer_chart = (
            alt.Chart(optim_counts)
            .mark_arc()
            .encode(
                theta=alt.Theta(field="count", type="quantitative"),
                color=alt.Color(field="optimizer", type="nominal"),
                tooltip=["optimizer", "count"],
            )
            .properties(
                title="Optimizers Used", width=300, height=225  # Smaller width  # Smaller height
            )
        )
        st.altair_chart(optimizer_chart)

        st.header("Batch Sizes Stats")

        # Pie chart for batch sizes using Altair with smaller dimensions
        batch_chart = (
            alt.Chart(batch_size_counts)
            .mark_arc()
            .encode(
                theta=alt.Theta(field="count", type="quantitative"),
                color=alt.Color(field="batch_size", type="nominal"),
                tooltip=["batch_size", "count"],
            )
            .properties(
                title="Batch Sizes", width=300, height=225  # Smaller width  # Smaller height
            )
        )
        st.altair_chart(batch_chart)

    st.image(str(FIGURES_APP_DIR / "footer.jpg"), use_column_width="always")
