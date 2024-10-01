import typer
import pickle as pk
from sklearn.model_selection import train_test_split


from mates.features import load_params, read_data
from mates.config import PROCESSED_DATA_DIR


app = typer.Typer()


@app.command()
def process_data(
): 
    """
    """
    params = load_params("prepare")

    X, y, encoding_labels = read_data(train_data=params["is_train"])
    
    if params["is_train"]:
        output_shape = len(encoding_labels)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=params["split_size"], random_state=params["seed"])

        if params["save_processed"]:
            with open(PROCESSED_DATA_DIR / 'output_shape.pkl', 'wb') as f:
                pk.dump(output_shape, f)

            with open(PROCESSED_DATA_DIR / 'X_train.pkl', 'wb') as f:
                pk.dump(X_train, f)
            with open(PROCESSED_DATA_DIR / 'y_train.pkl', 'wb') as f:
                pk.dump(y_train, f)
            with open(PROCESSED_DATA_DIR / 'X_valid.pkl', 'wb') as f:
                pk.dump(X_val, f)
            with open(PROCESSED_DATA_DIR / 'y_valid.pkl', 'wb') as f:
                pk.dump(y_val, f)

    print("Data processed!")

if __name__ == "__main__":
    process_data()