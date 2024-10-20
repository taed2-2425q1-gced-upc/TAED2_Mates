""" Module to test function predict_test from mates.modeling.predict """
import os
import pandas as pd
from mates.config import OUTPUT_DATA_DIR, RAW_DATA_DIR
from mates.modeling.predict import predict_test


def test_predict():
    """
    Test for function predict_test, which predicts the dog breed 
    for all images in the raw/tests directory
    """

    predict_test()

    # Asert
    output_csv_path = os.path.join(OUTPUT_DATA_DIR, 'predictions_test.csv')

    # Read the generated CSV
    generated_df = pd.read_csv(output_csv_path)
    image_files = os.listdir(os.path.join(RAW_DATA_DIR, 'test'))
    num_images = len(image_files)
    assert len(generated_df) == num_images, \
        f"Expected {num_images} predictions, but got {len(generated_df)}."

    assert generated_df.shape[1] == 2, f"Expected 2 columns, but got {generated_df.shape[1]}."
    # Assert all breeds in the second column belong to the predefined list
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

    assert all(generated_df['breed'].isin(dog_breeds)), \
        "One or more breeds in the predictions are not in the predefined dog breeds list."
