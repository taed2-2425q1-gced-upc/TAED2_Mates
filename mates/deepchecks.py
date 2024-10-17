from pathlib import Path
import numpy as np
import typer
from loguru import logger
from PIL import Image
from deepchecks.vision import VisionData, BatchOutputFormat
from deepchecks.vision.suites import data_integrity, train_test_validation
from sklearn.model_selection import train_test_split

from mates.config import RAW_DATA_DIR, REPORTS_DIR, IMG_SIZE
from mates.features import read_data, load_params

app = typer.Typer()


@app.command()
def custom_generator(X, y, batch_size, target_size=(IMG_SIZE, IMG_SIZE)):
    """"""
    n = len(X)
    for i in range(0, n, batch_size):
        images_batch = []
        labels_batch = []
        
        for j in range(i, min(i + batch_size, n)):
            img = Image.open(X[j]).resize(target_size)
            # values must be between 0 and 255
            img = np.array(img, dtype=np.uint8)
            label = np.where(y[j] == 1)[0][0]
            images_batch.append(img)
            labels_batch.append(label)
        
        images_batch = np.array(images_batch)
        labels_batch = np.array(labels_batch)
        
        yield BatchOutputFormat(images=images_batch, labels=labels_batch)


@app.command()
def create_vision_data(generator, task_type):
    """"""
    return VisionData(generator, task_type=task_type, reshuffle_data=False)


@app.command()
def run_checks(train_ds, val_ds, reports_dir):
    """"""

    reports_dir.mkdir(parents=True, exist_ok=True)

    suite = data_integrity()
    suite.add(train_test_validation())
    result = suite.run(train_ds, val_ds)
    result.save_as_html(str(reports_dir / "train_val_split_check.html"))


@app.command()
def train_val_split_check():
    """"""
    logger.info("Running checks on train-validation split...")
    params = load_params("prepare")

    X, y, _ = read_data(dir_path=RAW_DATA_DIR, train_data=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                      test_size=params["split_size"],
                                                      random_state=params["seed"])
        
    train_ds = create_vision_data(custom_generator(X_train, y_train, params["batch_size"]), 'classification')
    val_ds = create_vision_data(custom_generator(X_val, y_val, params["batch_size"]), 'classification')
    run_checks(train_ds, val_ds, REPORTS_DIR / "deepchecks")
    
    logger.success("Checks completed.")


if __name__ == "__main__":
    train_val_split_check() 




