import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# --- 1. MODEL DEFINITION (Must match training exactly) ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        last_step_out = out[:, -1, :] 
        prediction = self.fc(last_step_out)
        return self.sigmoid(prediction)

# --- 2. INFERENCE WRAPPER ---
class SolarInference:
    def __init__(self, model_path, scaler_path, ghi_map_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Artifacts
        self.scaler = joblib.load(scaler_path)
        self.max_ghi_map = joblib.load(ghi_map_path)
        
        # Initialize Model
        # input_dim is 14 based on your feature_cols
        self.model = LSTMModel(input_size=14).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.lookback = 12 # 1 hour of 5-min intervals
        self.feature_cols = [
            'DHI', 'DNI', 'GHI', 'Wind_speed', 'Humidity', 'Temperature',
            'season_sin', 'season_cos', 'weekday_sin', 'weekday_cos',
            'PV_potential', 'GHI_diff_5m', 'GHI_rolling_std_30m', 'Clearness_Index'
        ]
        self.minmax_cols = [
            'DHI', 'DNI', 'GHI', 'Wind_speed', 'Humidity', 'Temperature',
            'PV_potential', 'GHI_diff_5m', 'GHI_rolling_std_30m', 'Clearness_Index'
        ]

    def preprocess(self, df_raw):
        """
        Expects a DataFrame with columns: 
        ['Time', 'DHI', 'DNI', 'GHI', 'Wind_speed', 'Humidity', 'Temperature', 'Season', 'Day_of_the_week']
        Needs at least 12 rows to perform rolling calculations.
        """
        df = df_raw.copy()
        df['Time'] = pd.to_datetime(df['Time'])
        
        # 1. Cyclical Features
        df["season_sin"] = np.sin(2 * np.pi * df["Season"] / 4)
        df["season_cos"] = np.cos(2 * np.pi * df["Season"] / 4)
        df["weekday_sin"] = np.sin(2 * np.pi * df["Day_of_the_week"] / 7)
        df["weekday_cos"] = np.cos(2 * np.pi * df["Day_of_the_week"] / 7)
        
        # 2. Potential & Rolling Features
        df["PV_potential"] = df["GHI"] * df["Temperature"]
        df['GHI_diff_5m'] = df['GHI'].diff(periods=1)
        df['GHI_rolling_std_30m'] = df['GHI'].rolling(window=6).std()
        
        # 3. Clearness Index using the saved map
        df['hour_min'] = df['Time'].dt.strftime('%H:%M')
        df['Clearness_Index'] = df.apply(
            lambda row: row['GHI'] / (self.max_ghi_map.get(row['hour_min'], 1.0) + 1e-9), axis=1
        )
        
        # Drop rows with NaNs (first 6 rows usually due to rolling std)
        df = df.dropna().reset_index(drop=True)
        
        # 4. Scaling
        df[self.minmax_cols] = self.scaler.transform(df[self.minmax_cols])
        
        return df[self.feature_cols]

    def predict(self, raw_data_list):
        """
        Takes a list of dictionaries (from JSON).
        Returns a single prediction for the next time step.
        """
        df_input = pd.DataFrame(raw_data_list)
        
        # Ensure we have enough data
        if len(df_input) < (self.lookback + 6):
            raise ValueError(f"Need at least {self.lookback + 6} rows of data for feature engineering.")

        processed_features = self.preprocess(df_input)
        
        # Grab the last 'lookback' steps
        input_sequence = processed_features.tail(self.lookback).values.astype(np.float32)
        
        # Convert to Tensor (Batch, Seq, Feature)
        input_tensor = torch.tensor(input_sequence).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(input_tensor)
        
        return prediction.item()

# --- 3. EXAMPLE USAGE ---
if __name__ == "__main__":
    predictor = SolarInference(
        model_path='best_solar_model.pth',
        scaler_path='scaler.gz',
        ghi_map_path='max_ghi_map.gz'
    )
    
    # Mock data for testing (In web app, this comes from your DB or API request)
    # You need 18+ rows to account for the rolling window + lookback steps
    # ...
    # print(f"Next Predicted Value: {predictor.predict(mock_data)}")