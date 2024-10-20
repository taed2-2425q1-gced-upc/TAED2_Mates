### README.md

---

# MATES Project Overview

The `mates` directory contains the core components and scripts for the dog breed classification project. This includes API code, configuration settings, feature extraction, model training, and utilities. The project is organized into several subfolders to maintain modularity and facilitate ease of use, development, and testing.

The following sections describe the folder structure and provide an overview of the functionality of each file.

---

## Folder Structure

```
mates/
├── README.md                           # This README file providing an overview of the mates directory.
├── __init__.py                         # Initialization file for the mates package.
├── app/
│   └── api.py                          # API implementation for model serving and interaction.
├── config.py                           # Configuration file containing global settings for the project.
├── features/
│   ├── __init__.py                     # Initialization file for the features module.
│   ├── deepchecks.py                   # Script for performing Deepchecks validations on the dataset.
│   ├── features.py                     # Feature extraction and processing methods.
│   ├── gaissa/
│   │   ├── __init__.py                 # Initialization file for the GAiSSA feature module.
│   │   ├── calculator.py               # Script for calculating GAiSSA metrics (sustainability metrics).
│   │   ├── gaissaplugin.py             # GAiSSA plugin for integrating sustainability tracking.
│   │   ├── main_script.py              # Main script to run the GAiSSA sustainability checks.
│   │   └── plugin_interface.py         # Interface for GAiSSA plugin integration.
│   └── utils.py                        # Utility functions for feature extraction and general processing.
├── modeling/
│   ├── __init__.py                     # Initialization file for the modeling module.
│   ├── predict.py                      # Script for making predictions using the trained model.
│   ├── prepare.py                      # Script for preparing data for training and validation.
│   └── train.py                        # Script for training the model on the dataset.
└── streamlit.py                        # Streamlit application file for interactive model demos and interfaces.
```

---

## File Descriptions

### 1. `app/`
- **`api.py`**: Implements the API that serves the trained model for inference. It handles requests to predict dog breeds from input images and returns results in a suitable format. This API can be integrated into web applications or other external systems.

### 2. `config.py`
This file contains the global configuration settings for the project, such as file paths, hyperparameters, and environment variables. It helps in managing constants and settings across different modules, making the project easier to configure and scale.

### 3. `features/`
This folder contains code related to feature extraction and data validation.
- **`deepchecks.py`**: A script to run various Deepchecks tests on the dataset to validate the training and validation data quality.
- **`features.py`**: Core feature extraction methods, responsible for processing input data into a format suitable for model training.
- **`gaissa/`**: This subfolder contains scripts related to GAiSSA (Green AI Sustainable Software Assessment), which tracks the environmental impact of the project.
  - **`calculator.py`**: Script that calculates sustainability metrics (e.g., energy consumption, carbon emissions) for the training process.
  - **`gaissaplugin.py`**: Plugin for integrating GAiSSA with the project, enabling real-time sustainability tracking.
  - **`main_script.py`**: Main script to execute GAiSSA checks.
  - **`plugin_interface.py`**: Interface for communication between GAiSSA and other project components.
- **`utils.py`**: A collection of utility functions used for feature extraction, data manipulation, and other support functions.

### 4. `modeling/`
This folder contains scripts related to model training, prediction, and data preparation.
- **`predict.py`**: Script that loads the trained model and performs predictions on new data.
- **`prepare.py`**: Prepares and processes the dataset for training and validation. This includes splitting, cleaning, and formatting the data.
- **`train.py`**: The script responsible for training the machine learning model using the processed dataset.

### 5. `streamlit.py`
This file contains the code for a Streamlit application, allowing for interactive demonstrations of the dog breed classification model. The app provides a user-friendly interface for uploading images and receiving breed predictions.

---

## How to Use

1. **API Usage**: The `app/api.py` file contains the necessary code to run an API for inference. Use this to integrate the trained model into external systems or web applications.
   
2. **Training and Inference**: Use the scripts in the `modeling/` folder to train the model (`train.py`) and make predictions on new images (`predict.py`). The data must be preprocessed using `prepare.py` before training.

3. **Feature Extraction**: The `features/` folder provides code for extracting features from the dataset and validating the quality of the data through Deepchecks and GAiSSA sustainability checks.

4. **Streamlit Application**: Run `streamlit.py` to launch an interactive web app where users can upload dog images and receive breed predictions. This is useful for demonstrating model performance in real-time.
