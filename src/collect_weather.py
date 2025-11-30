import pandas as pd
import requests
import os

# Ensure data folder exists
os.makedirs("data", exist_ok=True)

def fetch_weather():
    """
    Fetch daily temperature and hourly humidity for Kuwait City
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 29.3759,
        "longitude": 47.9774,
        "daily": "temperature_2m_max,temperature_2m_min",
        "hourly": "relativehumidity_2m",
        "timezone": "Asia/Kuwait"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"Error: API request failed with status code {response.status_code}")
        print("Response:", response.text)
        return None

    data = response.json()

    if "daily" not in data or "hourly" not in data:
        print("Error: Missing 'daily' or 'hourly' data in API response")
        print(data)
        return None

    # Daily temperatures
    df_daily = pd.DataFrame(data["daily"])

    # Hourly humidity → aggregate to daily mean
    df_hourly = pd.DataFrame(data["hourly"])
    df_hourly['time'] = pd.to_datetime(df_hourly['time'])
    df_hourly['date'] = df_hourly['time'].dt.date
    df_humidity_daily = df_hourly.groupby('date')['relativehumidity_2m'].mean().reset_index()
    df_humidity_daily.rename(columns={'relativehumidity_2m':'humidity_mean'}, inplace=True)

    # Merge daily temp + mean humidity
    df_weather = pd.merge(df_daily, df_humidity_daily, left_on='time', right_on='date', how='left')
    df_weather.drop(columns=['date'], inplace=True)

    # Save CSV
    output_path = "data/weather_kuwait.csv"
    df_weather.to_csv(output_path, index=False)
    print(f"✅ weather_kuwait.csv created at {output_path}")

    return df_weather

if __name__ == "__main__":
    fetch_weather()
