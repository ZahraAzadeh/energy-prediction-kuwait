import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Ensure outputs folder exists
os.makedirs("models", exist_ok=True)


def load_data():
    """
    Load merged dataset for training
    """
    path = "data/merged_dataset.csv"
    try:
        df = pd.read_csv(path)
        print("✅ merged_dataset.csv loaded")
        return df
    except FileNotFoundError:
        print(f"⚠️ {path} not found. Please run preprocess.py first.")
        return None


def preprocess_data(df):
    """
    Prepare features and target
    Assumes the target column is 'energy_consumption' (adjust as needed)
    """
    if 'energy_consumption' not in df.columns:
        print("⚠️ Please ensure 'energy_consumption' column exists in merged_dataset.csv")
        return None, None

    # Features: all columns except target
    X = df.drop(columns=['energy_consumption'])

    # Target
    y = df['energy_consumption']

    # Optional: handle categorical or missing data here
    X = pd.get_dummies(X)  # convert categorical columns to one-hot
    X.fillna(0, inplace=True)  # simple missing value handling

    return X, y


def train_model(X, y):
    """
    Train Random Forest Regressor
    """
    # Split data into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train model
    model.fit(X_train, y_train)
    print("✅ Model trained")

    # Evaluate
    y_pred = model.predict(X_test)
    print("Model Evaluation:")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")

    # Save model
    model_path = "models/random_forest_model.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Model saved at {model_path}")


if __name__ == "__main__":
    df = load_data()
    if df is not None:
        X, y = preprocess_data(df)
        if X is not None and y is not None:
            train_model(X, y)
