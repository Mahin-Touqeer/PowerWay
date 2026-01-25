import pandas as pd
import numpy as np
from collections import deque
import joblib

# ======================================================
#  CORE FORECASTING ENGINE
#  (Backend Logic)
# ======================================================
class WindForecaster:
    def __init__(self, model_path, scaler_path, initial_lags):
        """
        Initialize the forecaster with the trained model, scaler, and historical state.
        """
        # 1. Load Model
        try:
            self.model = joblib.load(model_path)
            print(f" Model loaded from {model_path}")
        except FileNotFoundError:
            print(f" Error: Model file '{model_path}' not found.")
            self.model = None

        # 2. Load Scaler
        try:
            self.scaler = joblib.load(scaler_path)
            print(f"Scaler loaded from {scaler_path}")
        except FileNotFoundError:
            print(f"Error: Scaler file '{scaler_path}' not found.")
            self.scaler = None

        # 3. Initialize the state (Lag Buffer)
        self.lag_buffer = deque(initial_lags, maxlen=5)

    def _normalize_input(self, temp, hum, wind):
        """
        Uses the loaded MinMaxScaler to transform inputs.
        Scaler Expects: [Wind_speed, Humidity, Temperature]
        """
        if self.scaler is None:
            return 0, 0, 0
            
        # Create a DataFrame with the exact column names/order the scaler was trained on
        input_df = pd.DataFrame([{
            'Wind_speed': wind,
            'Humidity': hum,
            'Temperature': temp
        }])
        
        # Transform the data
        scaled_values = self.scaler.transform(input_df)[0]
        
        # Unpack based on scaler order [Wind, Hum, Temp]
        norm_wind = scaled_values[0]
        norm_hum  = scaled_values[1]
        norm_temp = scaled_values[2]
        
        return norm_wind, norm_temp, norm_hum

    def _build_features(self, norm_wind, norm_temp, norm_hum):
        """
        Constructs the exact DataFrame row expected by the XGBoost model.
        Features: 5 Lags + 3 Weather Variables
        """
        return pd.DataFrame([{
            "wp_lag_1": self.lag_buffer[0], # Most recent lag
            "wp_lag_2": self.lag_buffer[1],
            "wp_lag_3": self.lag_buffer[2],
            "wp_lag_4": self.lag_buffer[3],
            "wp_lag_5": self.lag_buffer[4], # Oldest lag
            "Wind_speed": norm_wind,
            "Temperature": norm_temp,
            "Humidity": norm_hum
        }])

    def predict(self, weather_data, actual_power=None):
        """
        Main interface function to generate a forecast.
        
        Args:
            weather_data (list/array): [Temperature, Humidity, Wind_Speed]
            actual_power (float, optional): The actual power reading from the PREVIOUS step 
                                            (if available via SCADA) to correct the lag buffer.
        """
        if self.model is None or self.scaler is None:
            print("⚠️ Warning: Model or Scaler not loaded.")
            return 0.0

        # 1. Unpack the input array
        # Expecting: [Temp, Humidity, Wind]
        temp = weather_data[0]
        humidity = weather_data[1]
        wind_speed = weather_data[2]

        # 2. Normalize inputs using the scaler
        norm_wind, norm_temp, norm_hum = self._normalize_input(temp, humidity, wind_speed)
        
        # 3. Build the feature vector
        X = self._build_features(norm_wind, norm_temp, norm_hum)

        # 4. Generate Prediction
        prediction = self.model.predict(X)[0]
        prediction = max(prediction, 0.0) # Enforce physical constraint

        # 5. Update State (Autoregression Logic)
        if actual_power is not None:
            self.lag_buffer.appendleft(actual_power)
        else:
            self.lag_buffer.appendleft(prediction)

        return prediction