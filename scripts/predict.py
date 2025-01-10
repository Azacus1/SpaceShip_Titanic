import pandas as pd
import pickle

def preprocess_features(data, is_train=False):
    # Extract features from Cabin if present
    if 'Cabin' in data.columns:
        data[['Deck', 'Room', 'Side']] = data['Cabin'].str.split('/', expand=True)
    
    # Handle missing values
    data['CryoSleep'] = data['CryoSleep'].map({'True': 1, 'False': 0})
    data['VIP'] = data['VIP'].map({'True': 1, 'False': 0})
    data['CryoSleep'].fillna(0, inplace=True)
    data['VIP'].fillna(0, inplace=True)
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['HomePlanet'].fillna('Unknown', inplace=True)
    data['Destination'].fillna('Unknown', inplace=True)
    data[['Deck', 'Room', 'Side']] = data[['Deck', 'Room', 'Side']].fillna('Unknown')

    # Convert 'Room' to numeric (e.g., hash encoding)
    if 'Room' in data.columns:
        data['Room'] = data['Room'].apply(lambda x: hash(x) if x != 'Unknown' else -1)

    # Encode categorical columns
    categorical_cols = ['HomePlanet', 'Destination', 'Deck', 'Side']
    for col in categorical_cols:
        data[col] = data[col].astype('category').cat.codes

    # Drop irrelevant columns
    if not is_train:
        data = data.drop(columns=['PassengerId'], errors='ignore')
    data = data.drop(columns=['Cabin', 'Name'], errors='ignore')

    return data

def generate_submission(model_path, test_path, submission_path):
    # Load model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # Load and preprocess test data
    test_data = pd.read_csv(test_path)
    passenger_ids = test_data['PassengerId']
    test_data = preprocess_features(test_data, is_train=False)

    # Align columns with training data (ensure same order and no mismatched columns)
    expected_features = model.get_booster().feature_names
    test_data = test_data[expected_features]

    # Predict on test data
    predictions = model.predict(test_data)

    # Convert predictions to True/False
    predictions = predictions.astype(bool)

    # Create submission file
    submission = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Transported': predictions
    })
    submission.to_csv(submission_path, index=False)
    print(f"Submission file saved to {submission_path}")

# Example usage
if __name__ == "__main__":
    generate_submission('models/xgboost_model.pkl', 'data/raw/test.csv', 'submissions/submission.csv')
