from pathlib import Path


from mates.features import load_params, load_model
from mates.config import PROCESSED_DATA_DIR


app = typer.Typer()


@app.command()
def predict(
):
    params = load_params("predict")
    model = load_model(params["model_name"])
    y_pred = model.predict()


if __name__ == "__main__":
    app()
