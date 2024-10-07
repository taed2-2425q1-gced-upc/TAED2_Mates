# Mates

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project.

## Project Organization

```
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- Top level description of the project
├── .dvc               <- DVC configuration files
├── data
│   ├── output         <- Final predictions and other outputs
│   ├── processed      <- Processed data ready for modeling
│   └── raw            <- Raw data coming from the source
│
├── docs               <- Documentation for the project
│
├── models             <- Trained and serialized models
│
├── notebooks          <- Jupyter notebooks for exploratory analysis
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mates and configuration for tools like black
│
├── metrics            <- Metrics for model evaluation and comparison
│
├── mlruns             <- MLflow tracking logs
│
├── reports            <- Generated analysis
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── setup.cfg          <- Configuration file for flake8
│
└── mates   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mates a Python module
    │
    ├── config.py               <- Store settings and configurations
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── prepare.py          <- Code to prepare data for modeling
    │   └── train.py            <- Code to train models
    │
```

--------

## Project Description

Strong software engineering practices are essential for ensuring the reliability and scalability of modern data science and machine learning projects. This project aims at providing hands-on experience in software engineering and good software practices, specifically regarding ML projects.

To put this into practice, our team has chosen to fine-tune an existing model in the task of dog breed classification. For this, the MobileNetV2 model was used as a based model.

This project will not only strengthen our technical proficiency in model fine-tuning and machine learning workflows but will also provide valuable experience in applying high-quality standard software engineering practices. By integrating principles such as version control, continuous integration, and code modularity, we aim to deliver a scalable and maintainable solution.