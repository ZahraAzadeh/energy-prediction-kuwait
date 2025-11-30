import pandas as pd
import os

# Ensure data folder exists
os.makedirs("data", exist_ok=True)

def load_weather_data():
    """
    Load weather_kuwait.csv
    """
    path = "data/weather_kuwait.csv"
    if not os.path.exists(path):
        print(f"⚠️  {path} not found. Run collect_weather.py first.")
        return None
    df_weather = pd.read_csv(path)
    print("✅ weather_kuwait.csv loaded")
    return df_weather

def load_energy_data():
    """
    Load Gulf energy consumption dataset
    """
    path = "data/energy_gulf.csv"
    if not os.path.exists(path):
        print(f"⚠️  {path} not found. Please place Gulf energy dataset in data/")
        return None
    df_energy = pd.read_csv(path)
    print("✅ energy_gulf.csv loaded")
    return df_energy

def merge_datasets(df_weather, df_energy):
    """
    Merge weather and energy datasets into merged_dataset.csv
    """
    if df_weather is None or df_energy is None:
        print("Error: Cannot merge datasets, one is missing")
        return None

    # Attempt to merge on date/time columns if available
    if "time" in df_weather.columns and "date" in df_energy.columns:
        df_merged = pd.merge(df_energy, df_weather, left_on="date", right_on="time", how="left")
    else:
        # If no matching column, concatenate side by side
        df_merged = pd.concat([df_energy.reset_index(drop=True), df_weather.reset_index(drop=True)], axis=1)

    output_path = "data/merged_dataset.csv"
    df_merged.to_csv(output_path, index=False)
    print(f"✅ merged_dataset.csv created at {output_path}")
    return df_merged

if __name__ == "__main__":
    weather = load_weather_data()
    energy = load_energy_data()
    merge_datasets(weather, energy)
