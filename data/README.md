### README.md

---

# Data Overview

This directory contains the necessary data for the dog breed classification project. The data includes raw images, processed features, labels, and model predictions.

The following sections describe the purpose of each document included in this directory.

---

## Folder Structure

```
./data/
├── README.md                   # This README file with details about the dataset structure and contents.
├── dog_catalogue_data.json      # A JSON file containing additional metadata for each dog breed.
├── output/
│   └── predictions_test.csv     # Final model predictions on the test set.
├── processed/
│   ├── X_train.pkl              # Preprocessed training set features from split data (ready for modeling).
│   ├── X_valid.pkl              # Preprocessed validation set features from split data (ready for modeling).
│   ├── output_shape.pkl         # Shape of the output layer used for modeling.
│   ├── y_train.pkl              # Labels for the training set from split data (encoded).
│   └── y_valid.pkl              # Labels for the validation set from split data (encoded).
└── raw/
    ├── labels.csv               # CSV file mapping image filenames to their respective dog breed labels.
    ├── test/
    │   └── <...>.jpg            # Test set images (in .jpg format).
    ├── test.dvc                 # DVC file for managing the test data versioning.
    ├── train/
    │   └── <...>.jpg            # Training set images (in .jpg format).
    └── train.dvc                # DVC file for managing the train data versioning.
```

---

## Data Overview

### 1. `dog_catalogue_data.json`
This file contains metadata for each dog breed. It includes breed names, image paths, descriptions, and other relevant information for the user interface.

### 2. `output/`
- **`predictions_test.csv`**: This file contains the predictions made by the trained model on the test set.

### 3. `processed/`
- **`X_train.pkl`**: Preprocessed feature data for training.
- **`X_valid.pkl`**: Preprocessed feature data for validation.
- **`output_shape.pkl`**: Information about the output shape for model training and prediction.
- **`y_train.pkl`**: Labels (dog breeds) for the training set.
- **`y_valid.pkl`**: Labels (dog breeds) for the validation set.

### 4. `raw/`
- **`labels.csv`**: A CSV file containing image filenames and their corresponding dog breed labels, used to train the model.
- **`train/`**: A directory containing the raw images for training, in `.jpg` format.
- **`test/`**: A directory containing the raw images for testing, in `.jpg` format.
- **`train.dvc` & `test.dvc`**: DVC files for managing dataset version control for the raw training and testing data.

---

## How to Use This Data

### Training and Validation Data
- Use `X_train.pkl` and `y_train.pkl` for model training.
- Use `X_valid.pkl` and `y_valid.pkl` for model validation.

### Predictions
- Load the `predictions_test.csv` file to view the model's predictions on the test set.

### Image Access
- The raw images for both training and testing are located in the `train/` and `test/` folders. Each image filename from the train set corresponds to an entry in `labels.csv`.

---

## Version Control with DVC
- The raw data is tracked using [DVC (Data Version Control)](https://dvc.org/). To pull the dataset, ensure you have DVC installed and initialized in your project.

