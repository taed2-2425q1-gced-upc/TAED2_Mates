import numpy as np
from pathlib import Path
from PIL import Image
import pickle as pk
import pytest
from mates.modeling.predict import predict_single
from mates.config import PROCESSED_DATA_DIR
from mates.features import load_processed_data, load_model

@pytest.fixture
def dbc_model():
    """Load model to test."""
    return load_model("mobilenet_exp_batch_62")

def test_dbc_model(dbc_model):
    """
    Test for dog breed classification model with predict_single function
    """
    
    dog_breeds = [
        "affenpinscher", "afghan_hound", "african_hunting_dog", "airedale", 
        "american_staffordshire_terrier", "appenzeller", "australian_terrier", 
        "basenji", "basset", "beagle", "bedlington_terrier", 
        "bernese_mountain_dog", "black-and-tan_coonhound", "blenheim_spaniel", 
        "bloodhound", "bluetick", "border_collie", "border_terrier", 
        "borzoi", "boston_bull", "bouvier_des_flandres", "boxer", 
        "brabancon_griffon", "briard", "brittany_spaniel", "bull_mastiff", 
        "cairn", "cardigan", "chesapeake_bay_retriever", "chihuahua", 
        "chow", "clumber", "cocker_spaniel", "collie", 
        "curly-coated_retriever", "dandie_dinmont", "dhole", "dingo", 
        "doberman", "english_foxhound", "english_setter", "english_springer", 
        "entlebucher", "eskimo_dog", "flat-coated_retriever", "french_bulldog", 
        "german_shepherd", "german_short-haired_pointer", "giant_schnauzer", 
        "golden_retriever", "gordon_setter", "great_dane", "great_pyrenees", 
        "greater_swiss_mountain_dog", "groenendael", "ibizan_hound", 
        "irish_setter", "irish_terrier", "irish_water_spaniel", 
        "irish_wolfhound", "italian_greyhound", "japanese_spaniel", 
        "keeshond", "kelpie", "kerry_blue_terrier", "komondor", 
        "kuvasz", "labrador_retriever", "lakeland_terrier", "leonberg", 
        "lhasa", "malamute", "malinois", "maltese_dog", 
        "mexican_hairless", "miniature_pinscher", "miniature_poodle", 
        "miniature_schnauzer", "newfoundland", "norfolk_terrier", 
        "norwegian_elkhound", "norwich_terrier", "old_english_sheepdog", 
        "otterhound", "papillon", "pekinese", "pembroke", 
        "pomeranian", "pug", "redbone", "rhodesian_ridgeback", 
        "rottweiler", "saint_bernard", "saluki", "samoyed", 
        "schipperke", "scotch_terrier", "scottish_deerhound", 
        "sealyham_terrier", "shetland_sheepdog", "shih-tzu", 
        "siberian_husky", "silky_terrier", "soft-coated_wheaten_terrier", 
        "staffordshire_bullterrier", "standard_poodle", "standard_schnauzer", 
        "sussex_spaniel", "tibetan_mastiff", "tibetan_terrier", 
        "toy_poodle", "toy_terrier", "vizsla", "walker_hound", 
        "weimaraner", "welsh_springer_spaniel", "west_highland_white_terrier", 
        "whippet", "wire-haired_fox_terrier", "yorkshire_terrier"
    ]
    
    # Create a mapping
    dog_breed_mapping = {idx: breed for idx, breed in enumerate(dog_breeds)}

    # Load validation data
    with open(PROCESSED_DATA_DIR / 'y_valid.pkl', 'rb') as f:
        y_valid = pk.load(f)
    with open(PROCESSED_DATA_DIR / 'x_valid.pkl', 'rb') as f:
        x_valid = pk.load(f)

    _, valid_data, _ = load_processed_data(32)
    expected = [dog_breed_mapping[np.argmax(y)] for y in y_valid]

    # Initialize counters
    correct_predictions = 0
    num_tests = min(len(x_valid), 1000)

    for i in range(num_tests):
        # Open the image once and convert to RGB
        img_path = Path(x_valid[i])
        with Image.open(img_path) as image:
            # Predict the breed
            predicted_breed = predict_single(dbc_model, dog_breeds, image)
            assert predicted_breed in dog_breeds
            
            if predicted_breed == expected[i]:
                correct_predictions += 1

    accuracy = correct_predictions / num_tests
    assert accuracy >= 0.8, f"Expected accuracy > 0.8, but got {accuracy:.4f}"
