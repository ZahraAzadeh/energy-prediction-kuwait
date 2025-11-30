# Energy Consumption Prediction in Kuwait

This project predicts electricity consumption in Kuwait based on weather conditions, particularly during heatwaves. Since Kuwait's electricity open data is limited, the model combines climate data from Kuwait City with regional energy consumption patterns from Gulf countries with similar temperature behavior.

## Features
- Daily temperature, humidity, wind speed, solar radiation
- Time features: hour, day of week, month
- Electricity consumption (kWh) as target
- Residential and commercial building consumption

## Model
- XGBoost / Random Forest used for accurate prediction
- Trained on combined datasets to generalize for Kuwait's climate

## Dashboard
- Built with Streamlit
- Interactive visualization of actual vs predicted consumption
- Recommendations for reducing energy usage during peak heat
  - Optimal AC scheduling
  - Load shifting

## How to Run
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run the Streamlit dashboard:
   ```
   streamlit run src/dashboard_app.py
   ```

## Data Sources
- Weather: Open-Meteo or NASA POWER API
- Energy: Gulf region datasets (UAE, Qatar, Saudi Arabia) from Kaggle / World Bank