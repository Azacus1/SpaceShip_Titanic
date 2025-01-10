import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(input_path, output_path):
    # Load data
    data = pd.read_csv(input_path)

    # Fill missing values
    data['HomePlanet'].fillna('Unknown', inplace=True)
    data['Destination'].fillna('Unknown', inplace=True)
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['VIP'].fillna(False, inplace=True)

    # Extract features from Cabin
    data[['Deck', 'Room', 'Side']] = data['Cabin'].str.split('/', expand=True)
    for col in ['Deck', 'Room', 'Side']:
        data[col].fillna('Unknown', inplace=True)

    # Encode categorical variables
    categorical_cols = ['HomePlanet', 'Destination', 'Deck', 'Side']
    for col in categorical_cols:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])

    # Drop unnecessary columns
    data.drop(columns=['Cabin', 'Name'], inplace=True)

    # Save processed data
    data.to_csv(output_path, index=False)



if __name__ == "__main__":
    preprocess_data('data/raw/train.csv', 'data/processed/train_cleaned.csv')
