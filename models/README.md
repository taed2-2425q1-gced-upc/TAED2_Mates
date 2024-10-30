
### README.md

---

# Models Overview

This directory contains the trained models for the dog breed classification project. The models are based on the MobileNet architecture and have been fine-tuned on the dataset for different batch sizes. These models are saved in `.h5` format and can be loaded for further inference or evaluation tasks.

The following sections describe the folder structure and contents of each file.

---

## Folder Structure

```
models/
├── README.md                           # This README file providing an overview of the models directory.
├── mobilenet_exp_batch_32.h5           # Trained MobileNet model with batch size 32.
└── mobilenet_exp_batch_62.h5           # Trained MobileNet model with batch size 62.
```

---

## File Descriptions

Since the model is being tracked in DVC, when cloning for the first time, the model files won't be available. To download the model files, run the following command in the terminal:

```bash
dvc pull models/mobilenet_exp_batch_32.h5

dvc pull models/mobilenet_exp_batch_62.h5
```

### 1. `mobilenet_exp_batch_32.h5`
This file contains the MobileNet model trained with a batch size of 32. The model was fine-tuned on the dog breed classification dataset and is stored in the HDF5 format (`.h5`). It can be used for inference or further fine-tuning by loading it into a Keras or TensorFlow environment.

**Usage**: 
To load the model:
```python
from tensorflow.keras.models import load_model
model = load_model('models/mobilenet_exp_batch_32.h5')
```

### 2. `mobilenet_exp_batch_62.h5`
This file contains the MobileNet model trained with a batch size of 62. Like the batch 32 model, it has been fine-tuned on the dog breed classification dataset and saved in the HDF5 format (`.h5`). It is ready for inference or continued training.

**Usage**: 
To load the model:
```python
from tensorflow.keras.models import load_model
model = load_model('models/mobilenet_exp_batch_62.h5')
```

---

## Model Usage

1. **Loading the Models**: The models can be loaded directly into any environment that supports TensorFlow or Keras. They can be used for both inference (classifying dog breeds from new images) and further training if needed.
   
2. **Evaluation**: These models have been fine-tuned on the dataset provided in the project and can be evaluated on additional datasets using standard model evaluation methods such as accuracy, precision, recall, etc.

3. **Inference**: Use these models to predict the breed of new dog images by passing the image data through the loaded model.
