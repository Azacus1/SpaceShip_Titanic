import pandas as pd
from scripts.data_preprocessing import preprocess_data

def test_preprocessing():
    preprocess_data('data/raw/train.csv', 'data/processed/train_cleaned.csv')
    processed_data = pd.read_csv('data/processed/train_cleaned.csv')
    assert not processed_data.isnull().sum().any(), "Preprocessed data contains null values!"
