import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

def preprocess_features(data, is_train=False):
    # Extract features from Cabin if present
    if 'Cabin' in data.columns:
        data[['Deck', 'Room', 'Side']] = data['Cabin'].str.split('/', expand=True)
    
    # Fill missing values
    data['CryoSleep'] = data['CryoSleep'].map({'True': 1, 'False': 0})
    data['VIP'] = data['VIP'].map({'True': 1, 'False': 0})
    data['CryoSleep'].fillna(0, inplace=True)
    data['VIP'].fillna(0, inplace=True)
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['HomePlanet'].fillna('Unknown', inplace=True)
    data['Destination'].fillna('Unknown', inplace=True)
    data[['Deck', 'Room', 'Side']] = data[['Deck', 'Room', 'Side']].fillna('Unknown')

    # Encode categorical columns
    categorical_cols = ['HomePlanet', 'Destination', 'Deck', 'Side']
    for col in categorical_cols:
        data[col] = data[col].astype('category').cat.codes

    # Drop irrelevant columns
    if not is_train:
        data = data.drop(columns=['PassengerId'], errors='ignore')
    data = data.drop(columns=['Cabin', 'Name'], errors='ignore')

    return data

def train_model(input_path, model_path):
    # Load preprocessed data
    data = pd.read_csv(input_path)
    
    # Preprocess features
    data = preprocess_features(data)

    # Split features and target
    X = data.drop(columns=['Transported'])
    y = data['Transported'].astype(int)

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost model
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', enable_categorical=True, random_state=42)
    model.fit(X_train, y_train)

    # Validate the model
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f'Validation Accuracy: {accuracy}')

    # Save the trained model
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

# Example usage
if __name__ == "__main__":
    train_model('data/processed/train_cleaned.csv', 'models/xgboost_model.pkl')
