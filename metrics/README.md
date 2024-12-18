### README.md

---

# Metrics Overview

This directory contains the performance metrics and emission tracking logs for different model experiments in the dog breed classification project. The metrics help track model performance and CO2 emissions during training, which is part of the project's effort to monitor environmental impact.

The following sections describe the folder structure and the content of each file.

---

## Folder Structure

```
metrics/
├── README.md                                    # This README file providing an overview of the metrics directory.
├── gaissa/
│   ├── gaissa_log.txt                           # Log file for tracking GAiSSA (Green AI Sustainable Software Assessment) metrics.
│   ├── gaissa_label.pdf                         # PDF file containing the label for the GAiSSA metrics.
│   └── gaissa_mobilenet_exp_batch_62.csv        # CSV file containing detailed metrics from the GAiSSA tracker for experiment batch 62.
|
├── mobilenet_exp_batch_62_emissions.csv         # CO2 emission metrics for final MobileNet experiment with batch size 62.
└── {optimizer}_{batch_size}_emissions.csv       # CO2 emission metrics for MobileNet experiment with different optimizer and batch sizes.
```

---

## File Descriptions

### 1. `gaissa/`
This folder contains files generated by GAiSSA, which is used to monitor and log sustainable AI practices, such as energy usage and carbon emissions during model training.

- **`gaissa_log.txt`**: A log file detailing the GAiSSA code execution and metrics tracking for the dog breed classification project. (Includes model architecture, energy consumption, and other relevant data.)
- **`gaissa_mobilenet_exp_batch_62.csv`**: CSV file containing specific GAiSSA metrics for the MobileNet experiment with a batch size of 62 (final model), including energy usage, emission data, and model training times.
- **`gaissa_label.pdf`**: A PDF file providing a label for the GAiSSA metrics.

### 2. `mobilenet_exp_batch_62_emissions.csv`
Similar to the batch 32 emissions file, this file contains CO2 emission metrics for the MobileNet experiment with batch size 62. It provides insights into the environmental impact of training with a larger batch size, helping in the evaluation of model efficiency.

### 3. `{optimizer}_{batch_size}_emissions.csv`
This file contains the CO2 emission metrics tracked during the training of the MobileNet model with different optimizers and batch sizes. The file name indicates the hyperparameter values. It records details such as the amount of energy consumed, duration of the training process, and the estimated carbon footprint generated during the experiment.

---

## How to Use the Metrics

- **Energy and Emissions Monitoring**: The emission CSV files provide insight into the environmental impact of different training configurations. These metrics can be used for optimizing the training process to minimize energy consumption and carbon emissions.
- **GAiSSA Logs**: The files in the `gaissa/` folder provide more detailed metrics specifically tailored for sustainable AI monitoring. Use these logs for detailed analysis of the energy efficiency of the experiments.
