import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = "data/merged_dataset.csv"
MODEL_PATH = "models/random_forest_model.pkl"

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data(path):
    if not os.path.exists(path):
        st.error(f"{path} not found. Please run preprocess.py first.")
        return None
    df = pd.read_csv(path)
    return df

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"{path} not found. Please run train_model.py first.")
        return None
    model = joblib.load(path)
    return model

# -----------------------------
# Main app
# -----------------------------
def main():
    st.title("🌡️ Kuwait Energy Consumption Prediction")
    st.write(
        """
        This dashboard predicts electricity consumption in Kuwait based on weather conditions
        (temperature, humidity) using a trained Random Forest model.
        """
    )

    df = load_data(DATA_PATH)
    model = load_model(MODEL_PATH)

    if df is None or model is None:
        return

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))

    st.subheader("Predict Energy Consumption")
    # Let user select a row to predict
    row_index = st.number_input("Select row index for prediction", min_value=0, max_value=len(df)-1, value=0, step=1)
    input_data = df.drop(columns=['energy_consumption']).iloc[[row_index]]

    # Make prediction
    prediction = model.predict(input_data)[0]
    actual = df['energy_consumption'].iloc[row_index]

    st.write(f"**Predicted Energy Consumption:** {prediction:.2f} kWh")
    st.write(f"**Actual Energy Consumption:** {actual:.2f} kWh")

    st.subheader("Plot Energy Consumption")
    st.line_chart(df[['energy_consumption']])

if __name__ == "__main__":
    main()
